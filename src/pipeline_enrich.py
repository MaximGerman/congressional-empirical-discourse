import logging

import pandas as pd
from tqdm import tqdm

from src.data import (
    HOUSE_MAJORITY,
    UNIFIED_GOVERNMENT,
    build_hearing_member_map,
    build_member_lookup,
    build_member_lookup_from_hearing_members,
    get_majority_status,
    load_hearings_committees,
    load_hearings_dates,
    load_hearings_members,
    load_members,
    load_members_terms,
    resolve_hearing_dates,
)
from src.elections import load_elections_data
from src.leadership import prepare_leadership_enrichment
from src.preprocess import extract_name_parts, is_likely_witness, match_speaker_to_member
from src.voteview import prepare_voteview_enrichment

logger = logging.getLogger(__name__)


def _vectorized_merge_match(pairs_df, matchable_df, left_name_col, match_type_label):
    """
    Vectorized merge-based matching of speaker names to member records.

    Tries matching on the full speaker_last_name first. For unmatched rows where
    speaker_last_word differs, retries with the last-word-only fallback.
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


def _merge_hearing_metadata(sentences_df, new_era):
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

    return sentences_df


def _filter_witnesses(sentences_df):
    # Filter out likely witnesses (title-based heuristic)
    sentences_df["is_witness"] = sentences_df["speaker"].apply(is_likely_witness)
    n_witness = sentences_df["is_witness"].sum()
    n_total = len(sentences_df)
    logger.info(
        "Identified %d witness sentences (%.1f%%) -- filtering out",
        n_witness,
        n_witness / n_total * 100 if n_total > 0 else 0,
    )

    legislators_df = sentences_df[~sentences_df["is_witness"]].copy()
    logger.info("Legislator sentences remaining: %d", len(legislators_df))
    return legislators_df


def _match_speakers_to_members(legislators_df, new_era):
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
    pre_filter_count = len(legislators_df)
    is_single_word = legislators_df["speaker_last_name"] == legislators_df["speaker_last_word"]
    is_unmatched = legislators_df["bioguide_id"].isna()
    is_not_chair = ~legislators_df["speaker_last_name"].isin({"CHAIRMAN", "CHAIRWOMAN", "CHAIRPERSON"})
    witness_reclassified = is_single_word & is_unmatched & is_not_chair
    n_reclassified = witness_reclassified.sum()
    logger.info(
        "Post-match witness reclassification: %d sentences (%.1f%% of legislators)",
        n_reclassified,
        n_reclassified / pre_filter_count * 100 if pre_filter_count > 0 else 0,
    )
    legislators_df = legislators_df[~witness_reclassified].copy()

    # Report match rate
    matched_count = legislators_df["bioguide_id"].notna().sum()
    total = len(legislators_df)
    logger.info("Match rate: %d/%d (%.1f%%)", matched_count, total, matched_count / total * 100 if total > 0 else 0)
    logger.info("Match type breakdown:\n%s", legislators_df["match_type"].value_counts().to_string())

    return legislators_df


def _apply_enrichments(legislators_df):
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


def step4_enrich_metadata(sentences_df, new_era):
    """Step 4: Enrich sentence records with political metadata."""
    logger.info("STEP 4: Enriching with political metadata")

    if sentences_df.empty:
        logger.warning("No sentences to enrich.")
        return sentences_df

    sentences_df = _merge_hearing_metadata(sentences_df, new_era)
    legislators_df = _filter_witnesses(sentences_df)
    legislators_df = _match_speakers_to_members(legislators_df, new_era)
    legislators_df = _apply_enrichments(legislators_df)

    return legislators_df
