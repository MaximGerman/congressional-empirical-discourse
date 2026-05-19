import logging
import os

import pandas as pd
from tqdm import tqdm

from src.data import (
    HOUSE_MAJORITY,
    UNIFIED_GOVERNMENT,
    build_hearing_member_map,
    build_member_lookup,
    build_member_lookup_from_hearing_members,
    load_hearings_committees,
    load_hearings_dates,
    load_hearings_members,
    load_members,
    load_members_terms,
    resolve_hearing_dates,
)
from src.elections import load_elections_data
from src.leadership import prepare_leadership_enrichment
from src.preprocess import extract_name_parts, match_speaker_to_member
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
    assert len(retVal) == len(pairs_df), (
        f"_vectorized_merge_match primary merge produced {len(retVal)} rows "
        f"but expected {len(pairs_df)}. Check that matchable_df is deduplicated "
        f"on [{left_name_col}, last_name_upper] before calling this function."
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
        # Guard: merge must stay 1:1; if duplicates crept through, skip rather than misalign
        assert len(fallback_matched) == fallback_mask.sum(), (
            f"_vectorized_merge_match fallback produced {len(fallback_matched)} rows "
            f"but expected {fallback_mask.sum()}. Check that matchable_df is deduplicated "
            f"on [{left_name_col}, last_name_upper] before calling this function."
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

    # Fix missing committee names: raw dataset has duplicate/incorrect Senate mappings for House hearings
    hc = hearings_committees.merge(hearing_info[["hearing_id", "chamber"]], on="hearing_id", how="left")

    def matches_chamber(row):
        if pd.isna(row["chamber"]) or pd.isna(row["committee_code"]):
            return 0
        if row["chamber"] == "house" and str(row["committee_code"]).startswith("hs"):
            return 1
        if row["chamber"] == "senate" and str(row["committee_code"]).startswith("ss"):
            return 1
        return 0

    hc["_matches_chamber"] = hc.apply(matches_chamber, axis=1)
    hc["_has_name"] = hc["committee_name"].notna().astype(int)

    # Sort to prioritize chamber-matching codes and non-null names, then deduplicate
    hc = hc.sort_values(by=["_matches_chamber", "_has_name"], ascending=[False, False])
    first_committee = hc.drop_duplicates(subset="hearing_id", keep="first")

    # Merge committee info
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
    # Filter out likely witnesses (vectorized title-based heuristic)
    legislator_prefixes = (
        "Mr.",
        "Mrs.",
        "Ms.",
        "Chairman",
        "Chairwoman",
        "The Chairman",
        "The Chairwoman",
        "Senator",
        "Representative",
    )
    sentences_df["is_witness"] = ~sentences_df["speaker"].fillna("").str.startswith(legislator_prefixes)
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
    member_cols = ["bioguide_id", "last_name", "first_name", "party", "state"]
    member_info = members_df[member_cols].copy() if not members_df.empty else pd.DataFrame(columns=member_cols)
    member_info = member_info.drop_duplicates(subset=["bioguide_id"], keep="first")

    terms_df = load_members_terms()
    member_lookup_terms = build_member_lookup(members_df, terms_df)

    hearing_members = load_hearings_members()
    member_lookup_hm = build_member_lookup_from_hearing_members(hearing_members, members_df, new_era)

    # --- Gap 3 Remediation: Augment with Voteview (HSall_members.csv) ---
    vv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "external", "HSall_members.csv")
    if os.path.exists(vv_path):
        vv = pd.read_csv(vv_path)
        vv_house = vv[(vv["chamber"] == "House") & (vv["congress"].isin([115, 116, 117, 118]))].copy()
        vv_lookup = vv_house[["bioguide_id", "congress"]].merge(member_info, on="bioguide_id", how="inner")
        vv_lookup["last_name_upper"] = vv_lookup["last_name"].str.upper()
    else:
        vv_lookup = pd.DataFrame()

    # Combine all sources, deduplicate
    member_lookup = pd.concat([member_lookup_terms, member_lookup_hm, vv_lookup], ignore_index=True)
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

    # Strategy: match unique (hearing_id, speaker_last_name, speaker_last_word, committee_code) pairs,
    # then merge results back. This reduces millions of lookups to thousands.
    unique_pairs = legislators_df[
        ["hearing_id", "speaker_last_name", "speaker_last_word", "congress", "committee_code"]
    ].drop_duplicates()
    logger.info("Unique (hearing, speaker) pairs to match: %d", len(unique_pairs))

    # --- Step A: hearing_member_exact (hearing_id + name) ---
    hm_matchable = _build_unique_matchable(hearing_member_map, ["hearing_id"])
    matched = _vectorized_merge_match(unique_pairs, hm_matchable, "hearing_id", "hearing_member_exact")

    # --- Step 0: Resolve Anonymous Chairs (Gap 2) ---
    from src.leadership import load_committee_leaders, normalize_committee_code

    target_congresses = sorted(legislators_df["congress"].dropna().unique().astype(int).tolist())
    leaders_df = load_committee_leaders(target_congresses=target_congresses)
    chair_lookup = leaders_df[leaders_df["role"] == "chair"][["congress", "thomas_id", "bioguide_id"]].copy()
    chair_lookup = chair_lookup.drop_duplicates(subset=["congress", "thomas_id"], keep="first")

    is_anon_chair = (
        matched["speaker_last_name"].isin({"CHAIRMAN", "CHAIRWOMAN", "CHAIRPERSON"})
        & matched["bioguide_id"].isna()  # Only resolve unmatched chairs
    )
    if is_anon_chair.any():
        matched.loc[is_anon_chair, "_thomas_id"] = matched.loc[is_anon_chair, "committee_code"].apply(
            normalize_committee_code
        )
        chair_matches = matched[is_anon_chair].merge(
            chair_lookup,
            left_on=["congress", "_thomas_id"],
            right_on=["congress", "thomas_id"],
            how="left",
            suffixes=("", "_chair"),
        )
        chair_matches = chair_matches.drop(
            columns=["bioguide_id", "last_name", "first_name", "party", "state"], errors="ignore"
        )
        chair_matches = chair_matches.merge(
            member_info, left_on="bioguide_id_chair", right_on="bioguide_id", how="left"
        )
        chair_matches_index = chair_matches.set_index(matched[is_anon_chair].index)
        for col in ["bioguide_id", "last_name", "first_name", "party", "state"]:
            if col in chair_matches_index.columns:
                matched.loc[is_anon_chair, col] = chair_matches_index[col].values

        has_chair_match = is_anon_chair & matched["bioguide_id"].notna()
        matched.loc[has_chair_match, "match_score"] = 100
        matched.loc[has_chair_match, "match_type"] = "committee_chair_exact"

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

    # Merge match results back into legislators_df.
    # NaN != NaN in pandas merge keys, so hearings with no committee assignment would silently
    # lose all matches. Use a sentinel to make NaN committee_codes mergeable.
    _SENTINEL = "__NO_COMMITTEE__"
    legislators_df = legislators_df.copy()
    legislators_df["committee_code"] = legislators_df["committee_code"].fillna(_SENTINEL)
    matched = matched.copy()
    matched["committee_code"] = matched["committee_code"].fillna(_SENTINEL)

    legislators_df = legislators_df.merge(
        matched[
            [
                "hearing_id",
                "speaker_last_name",
                "committee_code",
                "bioguide_id",
                "first_name",
                "party",
                "state",
                "match_score",
                "match_type",
            ]
        ].rename(columns={"first_name": "member_first_name", "state": "member_state"}),
        on=["hearing_id", "speaker_last_name", "committee_code"],
        how="left",
    )
    # Restore NaN for the sentinel
    legislators_df["committee_code"] = legislators_df["committee_code"].replace(_SENTINEL, pd.NA)

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
    # Add majority/minority status (vectorized and robust to unlisted third parties)
    majority_party = legislators_df["congress"].map(HOUSE_MAJORITY)
    normalized_party = legislators_df["party"].str.strip().replace({"Democrat": "Democratic"})

    legislators_df["minority"] = (normalized_party != majority_party).astype("Int8")
    # Maintain nulls for rows with missing party or unknown congress majority
    null_mask = legislators_df["party"].isna() | majority_party.isna()
    legislators_df.loc[null_mask, "minority"] = pd.NA

    legislators_df["unified"] = legislators_df["congress"].map(UNIFIED_GOVERNMENT)
    legislators_df["minuni"] = (legislators_df["minority"] * legislators_df["unified"]).astype("Int8")

    # Add democrat flag (vectorized)
    legislators_df["dem"] = pd.Series(pd.NA, index=legislators_df.index, dtype="Int8")
    is_dem = normalized_party.isin({"Democratic", "Democrat"})
    legislators_df.loc[legislators_df["party"].notna(), "dem"] = is_dem[legislators_df["party"].notna()].astype("Int8")

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

    # Filter out non-House members: Senators who testified at House hearings,
    # territorial delegates, and members matched to wrong congresses.
    # These have a bioguide_id but no Voteview data (because we only loaded House members
    # for the target congresses). They have wrong minority status and missing covariates.
    has_bio = legislators_df["bioguide_id"].notna()
    has_voteview = legislators_df["nominate_dim1"].notna() | legislators_df["seniority"].notna()
    non_house = has_bio & ~has_voteview
    n_non_house = non_house.sum()
    if n_non_house > 0:
        non_house_bios = legislators_df.loc[non_house, "bioguide_id"].unique()
        logger.warning(
            "Filtering %d rows from %d non-House members (Senators/delegates/wrong-congress): %s",
            n_non_house,
            len(non_house_bios),
            list(non_house_bios[:10]),
        )
        legislators_df = legislators_df[~non_house].copy()

    matched_mask = legislators_df["bioguide_id"].notna()
    n_matched = matched_mask.sum()

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
