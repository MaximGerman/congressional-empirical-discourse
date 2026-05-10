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
    build_hearing_member_map,
    build_member_lookup,
    filter_hearings_by_congress,
    get_majority_status,
    load_hearings,
    load_hearings_committees,
    load_hearings_dates,
    load_hearings_members,
    load_hearings_texts_chunked,
    load_members,
    load_members_terms,
)
from src.preprocess import (
    download_nltk_deps,
    is_likely_witness,
    match_speaker_to_member,
    process_single_hearing,
)

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

    # Merge date info (take first date per hearing)
    first_date = hearings_dates.drop_duplicates(subset="hearing_id", keep="first")
    sentences_df = sentences_df.merge(first_date[["hearing_id", "hearing_date"]], on="hearing_id", how="left")

    # Filter out likely witnesses
    sentences_df["is_witness"] = sentences_df["speaker"].apply(is_likely_witness)
    n_witness = sentences_df["is_witness"].sum()
    n_total = len(sentences_df)
    logger.info("Identified %d witness sentences (%.1f%%) -- filtering out", n_witness, n_witness / n_total * 100)

    legislators_df = sentences_df[~sentences_df["is_witness"]].copy()
    logger.info("Legislator sentences remaining: %d", len(legislators_df))

    # Match speakers to members using vectorized merge + fuzzy fallback
    members_df = load_members()
    terms_df = load_members_terms()
    member_lookup = build_member_lookup(members_df, terms_df)
    logger.info("Member lookup table: %d entries", len(member_lookup))

    # Load BICAM's direct hearing-member mapping
    hearing_members = load_hearings_members()
    hearing_member_map = build_hearing_member_map(hearing_members, members_df)

    logger.info("Matching speakers to members database (vectorized)...")

    # Strategy: match unique (hearing_id, speaker_last_name) pairs, then merge back.
    # This reduces millions of lookups to thousands.

    unique_pairs = legislators_df[["hearing_id", "speaker_last_name", "congress"]].drop_duplicates()
    logger.info("Unique (hearing, speaker) pairs to match: %d", len(unique_pairs))

    # Step A: Merge against hearing_member_map (exact match on hearing_id + last_name)
    # Only keep hearing-member combos where exactly one member matches that last name
    hm_deduped = hearing_member_map.drop_duplicates(subset=["hearing_id", "last_name_upper"])
    hm_counts = hearing_member_map.groupby(["hearing_id", "last_name_upper"]).size().reset_index(name="_count")
    hm_unique = hm_counts[hm_counts["_count"] == 1][["hearing_id", "last_name_upper"]]
    hm_matchable = hm_deduped.merge(hm_unique, on=["hearing_id", "last_name_upper"])

    matched_via_hearing = unique_pairs.merge(
        hm_matchable[["hearing_id", "last_name_upper", "bioguide_id", "last_name", "first_name", "party", "state"]],
        left_on=["hearing_id", "speaker_last_name"],
        right_on=["hearing_id", "last_name_upper"],
        how="left",
    )
    matched_via_hearing["match_type"] = None
    matched_via_hearing.loc[matched_via_hearing["bioguide_id"].notna(), "match_type"] = "hearing_member_exact"
    matched_via_hearing.loc[matched_via_hearing["bioguide_id"].notna(), "match_score"] = 100

    # Step B: For unmatched pairs, try exact match against all members in that congress
    unmatched_mask = matched_via_hearing["bioguide_id"].isna()
    unmatched = matched_via_hearing[unmatched_mask][["hearing_id", "speaker_last_name", "congress"]].copy()

    if not unmatched.empty:
        # Exact match against member_lookup (all House members by congress)
        ml_counts = member_lookup.groupby(["congress", "last_name_upper"]).size().reset_index(name="_count")
        ml_unique = ml_counts[ml_counts["_count"] == 1][["congress", "last_name_upper"]]
        ml_deduped = member_lookup.drop_duplicates(subset=["congress", "last_name_upper"])
        ml_matchable = ml_deduped.merge(ml_unique, on=["congress", "last_name_upper"])

        congress_matched = unmatched.merge(
            ml_matchable[["congress", "last_name_upper", "bioguide_id", "last_name", "first_name", "party", "state"]],
            left_on=["congress", "speaker_last_name"],
            right_on=["congress", "last_name_upper"],
            how="left",
        )
        congress_matched["match_type"] = None
        congress_matched.loc[congress_matched["bioguide_id"].notna(), "match_type"] = "congress_exact"
        congress_matched["match_score"] = None
        congress_matched.loc[congress_matched["bioguide_id"].notna(), "match_score"] = 100

        # Step C: For still unmatched, try fuzzy matching (only unique last_name + congress combos)
        still_unmatched = congress_matched[congress_matched["bioguide_id"].isna()]
        fuzzy_pairs = still_unmatched[["speaker_last_name", "congress"]].drop_duplicates()
        logger.info("Fuzzy matching %d unique (name, congress) pairs...", len(fuzzy_pairs))

        fuzzy_results = {}
        for _, fp in tqdm(fuzzy_pairs.iterrows(), total=len(fuzzy_pairs), desc="Fuzzy matching"):
            result = match_speaker_to_member(fp["speaker_last_name"], member_lookup, fp["congress"])
            fuzzy_results[(fp["speaker_last_name"], fp["congress"])] = result

        # Apply fuzzy results back to congress_matched
        for idx, row in congress_matched[congress_matched["bioguide_id"].isna()].iterrows():
            key = (row["speaker_last_name"], row["congress"])
            result = fuzzy_results.get(key)
            if result:
                congress_matched.at[idx, "bioguide_id"] = result["bioguide_id"]
                congress_matched.at[idx, "last_name"] = result["matched_name"]
                congress_matched.at[idx, "first_name"] = result["first_name"]
                congress_matched.at[idx, "party"] = result["party"]
                congress_matched.at[idx, "state"] = result["state"]
                congress_matched.at[idx, "match_score"] = result["match_score"]
                congress_matched.at[idx, "match_type"] = result["match_type"]

        # Update matched_via_hearing with results from congress_matched
        update_cols = ["bioguide_id", "last_name", "first_name", "party", "state", "match_score", "match_type"]
        for col in update_cols:
            if col not in matched_via_hearing.columns:
                matched_via_hearing[col] = None
        matched_via_hearing.loc[unmatched_mask, update_cols] = congress_matched[update_cols].values

    # Merge match results back into legislators_df
    legislators_df = legislators_df.merge(
        matched_via_hearing[
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

    logger.info("Party breakdown:\n%s", legislators_df["party"].value_counts().to_string())
    logger.info("Minority status (1=minority, 0=majority):\n%s", legislators_df["minority"].value_counts().to_string())

    # Drop helper columns
    legislators_df = legislators_df.drop(columns=["is_witness"], errors="ignore")

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
