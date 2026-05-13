"""
End-to-end pipeline for processing BICAM congressional hearing transcripts.

Steps:
1. Load hearings data, filter to 115th-118th Congress (House only)
2. Load transcripts and inspect format
3. Process transcripts: speaker segmentation -> sentence splitting
4. Enrich with political metadata using BICAM's hearing-member mappings
5. Create a representative sample for silver labeling (~10K sentences)

Usage:
    cd <project_root>
    .venv/bin/python -m src.pipeline
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from src.data import (
    HOUSE_MAJORITY,
    TEXT_COLUMN,
    UNIFIED_GOVERNMENT,
    build_hearing_member_map,
    build_member_lookup,
    build_member_lookup_from_hearing_members,
    filter_hearings_by_congress,
    get_majority_status,
    load_hearings,
    load_hearings_committees,
    load_hearings_dates,
    load_hearings_members,
    load_hearings_texts_chunked,
    load_members,
    load_members_terms,
    resolve_hearing_dates,
)
from src.elections import load_elections_data
from src.leadership import prepare_leadership_enrichment
from src.preprocess import (
    download_nltk_deps,
    extract_name_parts,
    is_likely_witness,
    match_speaker_to_member,
    process_single_hearing,
)
from src.voteview import prepare_voteview_enrichment

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step1_explore_data():
    """Step 1: Load, explore, and filter hearings metadata."""
    logger.info("STEP 1: Loading and exploring hearings data")

    hearings_df = load_hearings()
    logger.info("Total hearings in dataset: %d", len(hearings_df))
    logger.info("Columns: %s", list(hearings_df.columns))
    logger.info("Congress range: %d - %d", hearings_df["congress"].min(), hearings_df["congress"].max())

    # Filter to target period (House only, 115th+)
    new_era = filter_hearings_by_congress(hearings_df, min_congress=115, chamber="house")
    logger.info("House hearings from 115th Congress onwards: %d", len(new_era))
    logger.info("Breakdown by congress:\n%s", new_era["congress"].value_counts().sort_index().to_string())

    return hearings_df, new_era


def step2_load_transcripts(new_era):
    """Step 2: Load transcripts for target hearings."""
    logger.info("STEP 2: Loading transcripts for target hearings")

    target_ids = new_era["hearing_id"].tolist()
    logger.info("Loading transcripts for %d hearings (reading ~5.4GB file in chunks)...", len(target_ids))

    texts_df = load_hearings_texts_chunked(target_ids)
    logger.info("Loaded %d transcript records", len(texts_df))
    logger.info("Columns: %s", list(texts_df.columns))

    # Check text availability
    has_text = texts_df[TEXT_COLUMN].notna() & (texts_df[TEXT_COLUMN].str.len() > 100)
    logger.info("Records with substantial text: %d / %d", has_text.sum(), len(texts_df))

    # Show a sample
    sample_row = texts_df[has_text].iloc[0]
    text = str(sample_row[TEXT_COLUMN])
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if (
            line.strip().startswith(("Chairman", "Mr.", "Mrs.", "Ms."))
            and "...." not in line
            and len(line.strip()) > 20
        ):
            start = max(0, i)
            end = min(len(lines), i + 15)
            sample_lines = "\n".join(f"  {lines[j]}" for j in range(start, end))
            logger.debug("Sample transcript (hearing_id=%s):\n%s", sample_row["hearing_id"], sample_lines)
            break

    return texts_df


def step3_process_transcripts(new_era, texts_df):
    """Step 3: Process all transcripts into sentence-level records."""
    logger.info("STEP 3: Processing transcripts into sentence-level format")

    download_nltk_deps()

    # Only process hearings with actual text
    has_text = texts_df[TEXT_COLUMN].notna() & (texts_df[TEXT_COLUMN].str.len() > 100)
    processable = texts_df[has_text]
    logger.info("Processing %d hearings with text...", len(processable))

    all_records = []
    failed_hearings = []
    empty_hearings = 0

    for _, row in tqdm(processable.iterrows(), total=len(processable), desc="Processing hearings"):
        hearing_id = row["hearing_id"]
        text = str(row[TEXT_COLUMN])

        try:
            records = process_single_hearing(hearing_id, text)
            if records:
                all_records.extend(records)
            else:
                empty_hearings += 1
        except Exception as e:
            failed_hearings.append({"hearing_id": hearing_id, "error": str(e)})

    sentences_df = pd.DataFrame(all_records)
    logger.info("Total sentences extracted: %d", len(sentences_df))
    logger.info("Hearings with no speakers found: %d", empty_hearings)
    logger.info("Failed hearings: %d", len(failed_hearings))

    if not sentences_df.empty:
        logger.info("Unique speakers: %d", sentences_df["speaker"].nunique())
        logger.info("Unique hearings with sentences: %d", sentences_df["hearing_id"].nunique())

    return sentences_df, failed_hearings


def _vectorized_merge_match(pairs_df, matchable_df, left_name_col, match_type_label):
    """
    Vectorized merge-based matching of speaker names to member records.

    Tries matching on the full speaker_last_name first. For unmatched rows where
    speaker_last_word differs, retries with the last-word-only fallback.

    Args:
        pairs_df: DataFrame with columns [hearing_id|congress, speaker_last_name,
                  speaker_last_word, ...] — the pairs to match.
        matchable_df: DataFrame with [join_key, last_name_upper, bioguide_id, ...] —
                      the candidates, pre-filtered to unique name matches only.
        left_name_col: The join key besides the name ("hearing_id" or "congress").
        match_type_label: Label for the match_type column (e.g. "hearing_member_exact").

    Returns:
        pairs_df with added columns: bioguide_id, last_name, first_name, party,
        state, match_score, match_type.
    """
    member_cols = ["last_name_upper", "bioguide_id", "last_name", "first_name", "party", "state"]
    merge_cols = [left_name_col] + [c for c in member_cols if c in matchable_df.columns]

    # First pass: match on full speaker_last_name
    retVal = pairs_df.merge(
        matchable_df[merge_cols],
        left_on=[left_name_col, "speaker_last_name"],
        right_on=[left_name_col, "last_name_upper"],
        how="left",
        suffixes=("", "_full"),
    )

    # Second pass: for unmatched multi-word names, try speaker_last_word
    unmatched = retVal["bioguide_id"].isna()
    has_fallback = retVal["speaker_last_word"] != retVal["speaker_last_name"]
    fallback_mask = unmatched & has_fallback

    if fallback_mask.any():
        fallback_pairs = retVal.loc[fallback_mask, [left_name_col, "speaker_last_word"]].copy()
        fallback_matched = fallback_pairs.merge(
            matchable_df[merge_cols],
            left_on=[left_name_col, "speaker_last_word"],
            right_on=[left_name_col, "last_name_upper"],
            how="left",
        )
        for col in ["bioguide_id", "last_name", "first_name", "party", "state"]:
            if col in fallback_matched.columns:
                retVal.loc[fallback_mask, col] = fallback_matched[col].values

    retVal["match_type"] = None
    retVal.loc[retVal["bioguide_id"].notna(), "match_type"] = match_type_label
    retVal["match_score"] = None
    retVal.loc[retVal["bioguide_id"].notna(), "match_score"] = 100

    # Clean up merge artifacts
    retVal = retVal.drop(columns=["last_name_upper"], errors="ignore")

    return retVal


def _build_unique_matchable(lookup_df, group_cols):
    """
    From a lookup DataFrame, keep only entries where the name is unambiguous
    within the group (e.g. one SMITH per hearing or per congress).
    """
    counts = lookup_df.groupby([*group_cols, "last_name_upper"]).size().reset_index(name="_count")
    unique = counts[counts["_count"] == 1][[*group_cols, "last_name_upper"]]
    deduped = lookup_df.drop_duplicates(subset=[*group_cols, "last_name_upper"])
    retVal = deduped.merge(unique, on=[*group_cols, "last_name_upper"])
    return retVal


def step4_enrich_metadata(sentences_df, new_era):
    """Step 4: Enrich sentence records with political metadata."""
    logger.info("STEP 4: Enriching with political metadata")

    if sentences_df.empty:
        logger.warning("No sentences to enrich.")
        return sentences_df

    # Load hearing-level metadata
    hearings_committees = load_hearings_committees()
    hearings_dates = load_hearings_dates()

    # Merge congress info from hearings
    hearing_info = new_era[["hearing_id", "congress", "chamber"]].copy()
    if "title" in new_era.columns:
        hearing_info["title"] = new_era["title"]

    sentences_df = sentences_df.merge(hearing_info, on="hearing_id", how="left")

    # Merge committee info (take first committee per hearing)
    first_committee = hearings_committees.drop_duplicates(subset="hearing_id", keep="first")
    sentences_df = sentences_df.merge(
        first_committee[["hearing_id", "committee_code", "committee_name"]], on="hearing_id", how="left"
    )

    # Merge date info (resolve correct date per hearing using congress date ranges)
    resolved_dates = resolve_hearing_dates(hearings_dates, new_era)
    sentences_df = sentences_df.merge(resolved_dates[["hearing_id", "hearing_date"]], on="hearing_id", how="left")

    # Recompute name columns from speaker string (handles cached data from old pipeline)
    name_parts = sentences_df["speaker"].apply(extract_name_parts)
    sentences_df["speaker_last_name"] = name_parts.apply(lambda x: x[0])
    sentences_df["speaker_last_word"] = name_parts.apply(lambda x: x[1])

    # Filter out likely witnesses (title-based heuristic)
    sentences_df["is_witness"] = sentences_df["speaker"].apply(is_likely_witness)
    n_witness = sentences_df["is_witness"].sum()
    n_total = len(sentences_df)
    logger.info("Identified %d witness sentences (%.1f%%) -- filtering out", n_witness, n_witness / n_total * 100)

    legislators_df = sentences_df[~sentences_df["is_witness"]].copy()
    logger.info("Legislator sentences remaining: %d", len(legislators_df))

    # Build member lookup from terms (primary) + hearing_members (fills gaps, esp. 118th)
    members_df = load_members()
    terms_df = load_members_terms()
    member_lookup_terms = build_member_lookup(members_df, terms_df)

    hearing_members = load_hearings_members()
    member_lookup_hm = build_member_lookup_from_hearing_members(hearing_members, members_df, new_era)

    # Combine both sources, deduplicate
    member_lookup = pd.concat([member_lookup_terms, member_lookup_hm], ignore_index=True)
    member_lookup = member_lookup.drop_duplicates(subset=["bioguide_id", "congress"])
    logger.info(
        "Member lookup table: %d entries (terms: %d, hearing_members added: %d)",
        len(member_lookup),
        len(member_lookup_terms),
        len(member_lookup) - len(member_lookup_terms),
    )
    for c in sorted(member_lookup["congress"].dropna().unique()):
        n = member_lookup[member_lookup["congress"] == c]["bioguide_id"].nunique()
        logger.info("  Congress %d: %d unique members", int(c), n)

    # Build hearing-member map for direct hearing-level matching
    hearing_member_map = build_hearing_member_map(hearing_members, members_df)

    logger.info("Matching speakers to members database...")

    # Strategy: match unique (hearing_id, speaker_last_name, speaker_last_word) pairs,
    # then merge results back. This reduces millions of lookups to thousands.
    unique_pairs = legislators_df[
        ["hearing_id", "speaker_last_name", "speaker_last_word", "congress"]
    ].drop_duplicates()
    logger.info("Unique (hearing, speaker) pairs to match: %d", len(unique_pairs))

    # --- Step A: hearing_member_exact (hearing_id + name) ---
    hm_matchable = _build_unique_matchable(hearing_member_map, ["hearing_id"])
    matched = _vectorized_merge_match(unique_pairs, hm_matchable, "hearing_id", "hearing_member_exact")

    # --- Step B: congress_exact (congress + name) ---
    unmatched_mask = matched["bioguide_id"].isna()
    unmatched_b = matched[unmatched_mask][["hearing_id", "speaker_last_name", "speaker_last_word", "congress"]].copy()

    if not unmatched_b.empty:
        ml_matchable = _build_unique_matchable(member_lookup, ["congress"])
        congress_matched = _vectorized_merge_match(unmatched_b, ml_matchable, "congress", "congress_exact")

        # Write back into matched
        update_cols = ["bioguide_id", "last_name", "first_name", "party", "state", "match_score", "match_type"]
        for col in update_cols:
            if col not in matched.columns:
                matched[col] = None
        matched.loc[unmatched_mask, update_cols] = congress_matched[update_cols].values

    # --- Step C: fuzzy matching ---
    unmatched_mask_c = matched["bioguide_id"].isna()
    still_unmatched = matched[unmatched_mask_c]
    fuzzy_pairs = still_unmatched[["speaker_last_name", "speaker_last_word", "congress"]].drop_duplicates()
    logger.info("Fuzzy matching %d unique (name, congress) pairs...", len(fuzzy_pairs))

    fuzzy_results = {}
    for _, fp in tqdm(fuzzy_pairs.iterrows(), total=len(fuzzy_pairs), desc="Fuzzy matching"):
        result = match_speaker_to_member(
            fp["speaker_last_name"],
            member_lookup,
            fp["congress"],
            speaker_last_word=fp["speaker_last_word"],
        )
        fuzzy_results[(fp["speaker_last_name"], fp["congress"])] = result

    for idx, row in still_unmatched.iterrows():
        key = (row["speaker_last_name"], row["congress"])
        result = fuzzy_results.get(key)
        if result:
            for col, val in [
                ("bioguide_id", result["bioguide_id"]),
                ("last_name", result["matched_name"]),
                ("first_name", result["first_name"]),
                ("party", result["party"]),
                ("state", result["state"]),
                ("match_score", result["match_score"]),
                ("match_type", result["match_type"]),
            ]:
                matched.at[idx, col] = val

    # Note: "The Chairman" resolution would require committee chair role data
    # that BICAM's hearing_members.csv doesn't provide. These ~160K sentences
    # remain unmatched but are excluded by the post-match witness filter below
    # since they have speaker_last_name == "CHAIRMAN".

    # Merge match results back into legislators_df
    legislators_df = legislators_df.merge(
        matched[
            [
                "hearing_id",
                "speaker_last_name",
                "bioguide_id",
                "first_name",
                "party",
                "state",
                "match_score",
                "match_type",
            ]
        ].rename(columns={"first_name": "member_first_name", "state": "member_state"}),
        on=["hearing_id", "speaker_last_name"],
        how="left",
    )

    # --- Post-match witness filtering ---
    # Speakers who passed title-based witness filter but don't match ANY member
    # are very likely witnesses (e.g. "Mr. Wray", "Mr. Dodaro").
    # Only reclassify single-word names (multi-word names might just be matching failures).
    pre_filter_count = len(legislators_df)
    is_single_word = legislators_df["speaker_last_name"] == legislators_df["speaker_last_word"]
    is_unmatched = legislators_df["bioguide_id"].isna()
    is_not_chair = ~legislators_df["speaker_last_name"].isin({"CHAIRMAN", "CHAIRWOMAN", "CHAIRPERSON"})
    witness_reclassified = is_single_word & is_unmatched & is_not_chair
    n_reclassified = witness_reclassified.sum()
    logger.info(
        "Post-match witness reclassification: %d sentences (%.1f%% of legislators)",
        n_reclassified,
        n_reclassified / pre_filter_count * 100,
    )
    legislators_df = legislators_df[~witness_reclassified].copy()

    # Report match rate
    matched_count = legislators_df["bioguide_id"].notna().sum()
    total = len(legislators_df)
    logger.info("Match rate: %d/%d (%.1f%%)", matched_count, total, matched_count / total * 100)
    logger.info("Match type breakdown:\n%s", legislators_df["match_type"].value_counts().to_string())

    # Add majority/minority status (vectorized)
    majority_map = {
        (p, c): get_majority_status(p, c)
        for p in ["Republican", "Democratic", "Democrat", "Independent", "Libertarian"]
        for c in HOUSE_MAJORITY
    }

    legislators_df["minority"] = legislators_df.apply(
        lambda r: majority_map.get((r["party"], r["congress"])) if pd.notna(r["party"]) else None, axis=1
    )

    legislators_df["unified"] = legislators_df["congress"].map(UNIFIED_GOVERNMENT)
    legislators_df["minuni"] = legislators_df["minority"] * legislators_df["unified"]

    legislators_df["dem"] = legislators_df["party"].apply(
        lambda p: 1 if pd.notna(p) and str(p).strip() in ("Democratic", "Democrat") else (0 if pd.notna(p) else None)
    )

    logger.info("Party breakdown:\n%s", legislators_df["party"].value_counts().to_string())
    logger.info("Minority status (1=minority, 0=majority):\n%s", legislators_df["minority"].value_counts().to_string())
    logger.info("Unified government status:\n%s", legislators_df["unified"].value_counts().to_string())

    # Drop helper columns
    legislators_df = legislators_df.drop(columns=["is_witness", "speaker_last_word"], errors="ignore")

    # --- Voteview enrichment: DW-NOMINATE, seniority, gender ---
    target_congresses = sorted(legislators_df["congress"].dropna().unique().astype(int).tolist())
    voteview_df = prepare_voteview_enrichment(target_congresses=target_congresses)
    pre_voteview = len(legislators_df)
    legislators_df = legislators_df.merge(voteview_df, on=["bioguide_id", "congress"], how="left")

    # Report coverage (only for rows that have a bioguide_id — unmatched speakers can't be enriched)
    matched_mask = legislators_df["bioguide_id"].notna()
    n_matched = matched_mask.sum()
    n_with_nominate = (matched_mask & legislators_df["nominate_dim1"].notna()).sum()
    logger.info(
        "Voteview enrichment: %d/%d matched legislators have NOMINATE scores (%.1f%%)",
        n_with_nominate,
        n_matched,
        n_with_nominate / n_matched * 100 if n_matched > 0 else 0,
    )
    n_with_seniority = (matched_mask & legislators_df["seniority"].notna()).sum()
    logger.info(
        "Voteview enrichment: %d/%d matched legislators have seniority data (%.1f%%)",
        n_with_seniority,
        n_matched,
        n_with_seniority / n_matched * 100 if n_matched > 0 else 0,
    )
    if "gender" in legislators_df.columns:
        n_with_gender = (matched_mask & legislators_df["gender"].notna()).sum()
        logger.info(
            "Voteview enrichment: %d/%d matched legislators have gender data (%.1f%%)",
            n_with_gender,
            n_matched,
            n_with_gender / n_matched * 100 if n_matched > 0 else 0,
        )
    assert len(legislators_df) == pre_voteview, "Voteview merge changed row count — check for duplicate keys"

    # --- Committee leadership enrichment: chair/ranking member ---
    pre_leadership = len(legislators_df)
    legislators_df = prepare_leadership_enrichment(legislators_df)
    assert len(legislators_df) == pre_leadership, "Leadership merge changed row count — check for duplicate keys"

    # --- Election vote share enrichment ---
    pre_elections = len(legislators_df)
    elections_df = load_elections_data(target_congresses=target_congresses)
    legislators_df = legislators_df.merge(elections_df, on=["state_abbrev", "district_code", "congress"], how="left")

    n_with_vote = (matched_mask & legislators_df["vote_pct"].notna()).sum()
    logger.info(
        "Elections enrichment: %d/%d matched legislators have vote share data (%.1f%%)",
        n_with_vote,
        n_matched,
        n_with_vote / n_matched * 100 if n_matched > 0 else 0,
    )
    assert len(legislators_df) == pre_elections, "Elections merge changed row count"

    return legislators_df


def step5_create_sample(legislators_df, sample_size=10000, seed=42):
    """Step 5: Create a representative sample for silver labeling."""
    logger.info("STEP 5: Creating representative sample for silver labeling")

    if legislators_df.empty:
        logger.warning("No data to sample from.")
        return pd.DataFrame()

    # Only sample from matched legislators
    matched = legislators_df[legislators_df["bioguide_id"].notna()].copy()
    logger.info("Matched legislator sentences available: %d", len(matched))

    if len(matched) <= sample_size:
        logger.info("Dataset smaller than target sample (%d), using all data", sample_size)
        sample = matched.copy()
    else:
        # Stratified sample by congress
        congresses = sorted(matched["congress"].unique())
        per_congress = sample_size // len(congresses)
        parts = []
        for c in congresses:
            congress_data = matched[matched["congress"] == c]
            n = min(len(congress_data), per_congress)
            parts.append(congress_data.sample(n, random_state=seed))

        sample = pd.concat(parts, ignore_index=True)

        # Top up if we didn't hit the target
        if len(sample) < sample_size:
            remaining = matched[~matched.index.isin(sample.index)]
            extra = remaining.sample(min(len(remaining), sample_size - len(sample)), random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=True)

    logger.info("Sample size: %d", len(sample))
    logger.info("Congress distribution:\n%s", sample["congress"].value_counts().sort_index().to_string())
    logger.info("Party distribution:\n%s", sample["party"].value_counts().to_string())
    logger.info("Minority distribution:\n%s", sample["minority"].value_counts().to_string())

    return sample


def run_pipeline():
    """Run the full pipeline end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    ensure_output_dir()

    # Step 1: Explore and filter
    _hearings_df, new_era = step1_explore_data()

    # Step 2-3: Load and process transcripts (skip if raw CSV already exists)
    raw_path = os.path.join(OUTPUT_DIR, "sentences_raw.csv")
    if os.path.exists(raw_path):
        logger.info("Found existing raw sentences at %s, loading...", raw_path)
        sentences_df = pd.read_csv(raw_path)
        logger.info("Loaded %d sentences from cache", len(sentences_df))
    else:
        # Step 2: Load transcripts
        texts_df = step2_load_transcripts(new_era)

        # Step 3: Process transcripts
        sentences_df, _failed = step3_process_transcripts(new_era, texts_df)

        if sentences_df.empty:
            logger.error("No sentences extracted. Check transcript format.")
            return

        # Save raw sentences
        sentences_df.to_csv(raw_path, index=False)
        logger.info("Raw sentences saved to: %s", raw_path)

    # Step 4: Enrich with metadata
    legislators_df = step4_enrich_metadata(sentences_df, new_era)

    # Save enriched data
    enriched_path = os.path.join(OUTPUT_DIR, "sentences_enriched.csv")
    legislators_df.to_csv(enriched_path, index=False)
    logger.info("Enriched legislator sentences saved to: %s", enriched_path)

    # Step 5: Create sample
    sample = step5_create_sample(legislators_df)

    # Save sample
    sample_path = os.path.join(OUTPUT_DIR, "sample_for_labeling.csv")
    sample.to_csv(sample_path, index=False)
    logger.info("Silver labeling sample saved to: %s", sample_path)

    # Summary
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total hearings in target period: %d", len(new_era))
    logger.info("  Total sentences extracted: %d", len(sentences_df))
    logger.info("  Legislator sentences (filtered): %d", len(legislators_df))
    logger.info("  Matched to member database: %d", legislators_df["bioguide_id"].notna().sum())
    logger.info("  Silver labeling sample: %d", len(sample))
    logger.info("Output files in: %s/", OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()
