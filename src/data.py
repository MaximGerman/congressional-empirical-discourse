import os

import bicam
import pandas as pd

# House majority party by congress (for minority/majority coding)
HOUSE_MAJORITY = {
    115: "Republican",  # 2017-2019
    116: "Democratic",  # 2019-2021
    117: "Democratic",  # 2021-2023
    118: "Republican",  # 2023-2025
    119: "Republican",  # 2025-2027
}

# The text column in hearings_texts.csv
TEXT_COLUMN = "raw_text"


def download_all_data(confirm=True):
    """Downloads hearings and members datasets."""
    bicam.download_dataset("hearings", confirm=confirm)
    bicam.download_dataset("members", confirm=confirm)


def load_hearings():
    """Loads the main hearings dataframe (hearing_id, congress, chamber, etc.)."""
    return bicam.load_dataframe("hearings", "hearings.csv", download=True, confirm=True)


def load_hearings_metadata():
    """Loads the hearings metadata dataframe."""
    return bicam.load_dataframe("hearings", "hearings_metadata.csv", download=True, confirm=True)


def load_hearings_texts_chunked(target_ids, chunksize=500):
    """
    Load transcripts for specific hearing IDs by reading in chunks.
    The full file is ~5.4GB so we don't load it all at once.
    """
    path = os.path.join(os.path.expanduser("~"), ".bicam", "hearings", "hearings_texts.csv")
    target_set = set(target_ids)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        filtered = chunk[chunk["hearing_id"].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


def load_hearings_members():
    """Loads the hearing-member mapping (hearing_id -> bioguide_id)."""
    return bicam.load_dataframe("hearings", "hearings_members.csv", download=True, confirm=True)


def load_hearings_committees():
    """Loads hearing-committee mapping."""
    return bicam.load_dataframe("hearings", "hearings_committees.csv", download=True, confirm=True)


def load_hearings_dates():
    """Loads hearing dates."""
    return bicam.load_dataframe("hearings", "hearings_dates.csv", download=True, confirm=True)


def load_members():
    """Loads the members dataframe with name and party info."""
    return bicam.load_dataframe("members", "members.csv", download=True, confirm=True)


def load_members_terms():
    """Loads the members terms dataframe (congress-level membership info)."""
    return bicam.load_dataframe("members", "members_terms.csv", download=True, confirm=True)


def filter_hearings_by_congress(hearings_df, min_congress=115, chamber="house"):
    """Filter hearings to target congresses and chamber."""
    mask = hearings_df["congress"] >= min_congress
    if chamber:
        mask = mask & (hearings_df["chamber"] == chamber)
    return hearings_df[mask].copy()


def build_member_lookup(members_df, terms_df, target_congresses=None):
    """
    Build a lookup table mapping (last_name_upper, congress) -> member info.

    Returns a DataFrame with columns:
        bioguide_id, last_name, first_name, party, state, congress, last_name_upper
    """
    if target_congresses is None:
        target_congresses = list(HOUSE_MAJORITY.keys())

    # Filter terms to House members in target congresses
    house_terms = terms_df[(terms_df["chamber"] == "house") & (terms_df["congress"].isin(target_congresses))][
        ["bioguide_id", "congress", "state_code"]
    ].copy()

    # Merge with member info
    member_info = members_df[["bioguide_id", "last_name", "first_name", "party", "state"]].copy()
    retVal = house_terms.merge(member_info, on="bioguide_id", how="left")

    # Create uppercase last name for matching
    retVal["last_name_upper"] = retVal["last_name"].str.upper()

    return retVal


def build_hearing_member_map(hearings_members_df, members_df):
    """
    Build a map from (hearing_id, bioguide_id) to member info.
    Uses the BICAM hearings_members.csv which directly links hearings to members.
    """
    retVal = hearings_members_df.merge(
        members_df[["bioguide_id", "last_name", "first_name", "party", "state"]], on="bioguide_id", how="left"
    )
    retVal["last_name_upper"] = retVal["last_name"].str.upper()
    return retVal


def get_majority_status(party, congress):
    """
    Determine if a member is in the majority or minority party.
    Returns 1 for minority, 0 for majority (matching original paper convention).
    """
    majority_party = HOUSE_MAJORITY.get(congress)
    if majority_party is None or pd.isna(party):
        return None
    party_norm = party.strip()
    if party_norm in ("Democratic", "Democrat"):
        party_norm = "Democratic"
    if party_norm == majority_party:
        return 0  # majority
    return 1  # minority
