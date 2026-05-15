"""
Voteview data enrichment: DW-NOMINATE scores, gender, seniority, freshman status.

Data source: Voteview (voteview.com) HSall_members.csv — a single CSV with every
member of Congress, including NOMINATE scores, party, state, and bioguide ID.

This module downloads the file on first use and caches it in data/external/.
"""

import logging
import os
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

VOTEVIEW_MEMBERS_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "external")


def download_voteview_data(dest_dir=None):
    """
    Download HSall_members.csv from Voteview if not already present.

    Args:
        dest_dir: Directory to store the file. Defaults to data/external/.

    Returns:
        Path to the downloaded CSV file.
    """
    if dest_dir is None:
        dest_dir = DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, "HSall_members.csv")
    if os.path.exists(dest_path):
        logger.info("Voteview data already exists at %s", dest_path)
        return dest_path

    logger.info("Downloading Voteview members data from %s ...", VOTEVIEW_MEMBERS_URL)
    urllib.request.urlretrieve(VOTEVIEW_MEMBERS_URL, dest_path)
    logger.info("Downloaded Voteview data to %s", dest_path)
    return dest_path


def load_voteview_members(path=None, congress_range=None, chamber="House"):
    """
    Load Voteview member data, filtered to target congresses and chamber.

    Args:
        path: Path to HSall_members.csv. If None, downloads/uses default location.
        congress_range: Tuple of (min_congress, max_congress) inclusive.
                       If None, loads all congresses.
        chamber: Chamber to filter to ("House" or "Senate"). None for both.

    Returns:
        DataFrame with Voteview member data for the specified congresses.
    """
    if path is None:
        path = download_voteview_data()

    df = pd.read_csv(path)

    if chamber:
        df = df[df["chamber"] == chamber].copy()

    if congress_range:
        min_c, max_c = congress_range
        df = df[(df["congress"] >= min_c) & (df["congress"] <= max_c)].copy()

    return df


def compute_seniority(voteview_df, target_congresses=None):
    """
    Compute seniority for each member-congress pair.

    Seniority = number of congresses served in the House up to and including
    the current congress. A member serving their first term has seniority 1.

    Args:
        voteview_df: Full Voteview DataFrame (should include historical congresses
                     for accurate seniority computation, not just target ones).
        target_congresses: List of congresses to return seniority for.
                          If None, returns seniority for all congresses in the data.

    Returns:
        DataFrame with columns: bioguide_id, congress, seniority, freshman
    """
    house = voteview_df[voteview_df["chamber"] == "House"].copy()
    house_terms = house[["bioguide_id", "congress"]].drop_duplicates().sort_values(["bioguide_id", "congress"])

    # Cumulative count of House terms per member (1-indexed)
    house_terms["seniority"] = house_terms.groupby("bioguide_id").cumcount() + 1
    house_terms["freshman"] = (house_terms["seniority"] == 1).astype(int)
    house_terms["seniority_sq"] = house_terms["seniority"] ** 2

    if target_congresses is not None:
        house_terms = house_terms[house_terms["congress"].isin(target_congresses)]

    return house_terms[["bioguide_id", "congress", "seniority", "seniority_sq", "freshman"]]


def prepare_voteview_enrichment(path=None, target_congresses=None):
    """
    Prepare Voteview data for pipeline enrichment.

    Loads the full Voteview dataset (all congresses, for seniority computation),
    then returns a DataFrame with derived columns ready to merge on (bioguide_id, congress).

    Args:
        path: Path to HSall_members.csv. If None, downloads/uses default.
        target_congresses: List of target congress numbers (e.g., [115, 116, 117, 118]).

    Returns:
        DataFrame with columns: bioguide_id, congress, nominate_dim1, nominate_dim2,
                               abs_dwnom1, seniority, seniority_sq, freshman, and optionally gender/female.
    """
    if target_congresses is None:
        target_congresses = [115, 116, 117, 118]

    # Load ALL congresses for seniority computation (need full history)
    all_members = load_voteview_members(path=path, congress_range=None, chamber="House")
    logger.info("Loaded %d Voteview records (all House members, all congresses)", len(all_members))

    # Compute seniority using full history
    seniority_df = compute_seniority(all_members, target_congresses=target_congresses)

    # Filter to target congresses for the enrichment columns
    target_members = all_members[all_members["congress"].isin(target_congresses)].copy()
    logger.info("Voteview members in target congresses (%s): %d", target_congresses, len(target_members))

    # Select columns
    cols = ["bioguide_id", "congress", "nominate_dim1", "nominate_dim2", "state_abbrev", "district_code"]
    if "gender" in target_members.columns:
        cols.append("gender")
        logger.info("Gender column found in Voteview data")
    else:
        logger.warning("Gender column not found in Voteview data — gender will not be enriched")

    retVal = target_members[cols].copy()

    # Derived: absolute DW-NOMINATE dimension 1 (ideological extremity)
    retVal["abs_dwnom1"] = retVal["nominate_dim1"].abs()

    if "gender" in retVal.columns:
        retVal["female"] = (retVal["gender"] == "F").astype(int)

    # Merge seniority
    retVal = retVal.merge(seniority_df, on=["bioguide_id", "congress"], how="left")

    # Deduplicate: Voteview can have multiple rows per (bioguide_id, congress) for
    # members who served partial terms or had special elections. Keep first occurrence.
    n_before = len(retVal)
    retVal = retVal.drop_duplicates(subset=["bioguide_id", "congress"], keep="first")
    n_deduped = n_before - len(retVal)
    if n_deduped > 0:
        logger.info("Deduplicated %d duplicate (bioguide_id, congress) rows", n_deduped)

    # Rank-standardization (within congress)
    retVal["abs_dwnom1_rs"] = retVal.groupby("congress")["abs_dwnom1"].rank(pct=True)
    retVal["seniority_rs"] = retVal.groupby("congress")["seniority"].rank(pct=True)
    retVal["seniority_sq_rs"] = retVal["seniority_rs"] ** 2

    for c in sorted(target_congresses):
        n = retVal[retVal["congress"] == c]["bioguide_id"].nunique()
        logger.info("  Congress %d: %d members with Voteview data", c, n)

    n_missing_nom = retVal["nominate_dim1"].isna().sum()
    if n_missing_nom > 0:
        logger.warning("  %d rows with missing nominate_dim1", n_missing_nom)

    return retVal
