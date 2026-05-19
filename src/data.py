import logging
import os

import bicam
import pandas as pd

logger = logging.getLogger(__name__)

# House majority party by congress (for minority/majority coding)
HOUSE_MAJORITY = {
    115: "Republican",  # 2017-2019
    116: "Democratic",  # 2019-2021
    117: "Democratic",  # 2021-2023
    118: "Republican",  # 2023-2025
    119: "Republican",  # 2025-2027
}

# Unified government by congress (1 = unified, 0 = divided)
UNIFIED_GOVERNMENT = {
    115: 1,  # R House, R Senate, R Pres (Trump)
    116: 0,  # D House, R Senate, R Pres (Trump)
    117: 1,  # D House, D Senate, D Pres (Biden)
    118: 0,  # R House, D Senate, D Pres (Biden)
    119: 1,  # R House, R Senate, R Pres (Trump)
}

# Congress session date ranges (start date inclusive, end date exclusive)
CONGRESS_DATE_RANGES = {
    115: ("2017-01-03", "2019-01-03"),
    116: ("2019-01-03", "2021-01-03"),
    117: ("2021-01-03", "2023-01-03"),
    118: ("2023-01-03", "2025-01-03"),
    119: ("2025-01-03", "2027-01-03"),
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


def load_hearings_texts_chunked(target_ids, chunksize=10000):
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


def resolve_hearing_dates(hearings_dates, hearings_df):
    """
    Resolve the correct hearing date for each hearing by filtering to dates
    within the hearing's congress date range.

    BICAM's hearings_dates.csv contains multiple dates per hearing (historical
    proceedings, reissues, etc.). This function picks the date that falls within
    the congress's session period.

    Args:
        hearings_dates: DataFrame with columns [hearing_id, hearing_date]
        hearings_df: DataFrame with columns [hearing_id, congress]

    Returns:
        DataFrame with one row per hearing_id: [hearing_id, hearing_date]
    """
    df = hearings_dates.copy()
    df["hearing_date"] = pd.to_datetime(df["hearing_date"], errors="coerce")

    # Add congress info from hearings
    df = df.merge(hearings_df[["hearing_id", "congress"]].drop_duplicates(), on="hearing_id", how="left")

    # Build boolean mask: date falls within its congress range
    in_range = pd.Series(False, index=df.index)
    for congress, (start, end) in CONGRESS_DATE_RANGES.items():
        mask = (df["congress"] == congress) & (df["hearing_date"] >= start) & (df["hearing_date"] < end)
        in_range = in_range | mask

    # For multi-day hearings, pick the earliest in-range date (hearing start date).
    # This is a deliberate choice: the first session is when most opening statements
    # and substantive testimony occur.
    valid = df[in_range].sort_values("hearing_date").drop_duplicates(subset="hearing_id", keep="first")

    total_hearings = df["hearing_id"].nunique()
    n_in_range = len(valid)

    # Hearings with no in-range date: take the latest date as best guess
    missing_ids = set(df["hearing_id"]) - set(valid["hearing_id"])
    if missing_ids:
        fallback = (
            df[df["hearing_id"].isin(missing_ids)]
            .sort_values("hearing_date")
            .drop_duplicates(subset="hearing_id", keep="last")
        )
        valid = pd.concat([valid, fallback], ignore_index=True)

    logger.info(
        "Resolved hearing dates: %d/%d in-range, %d fallback (latest date used)",
        n_in_range,
        total_hearings,
        len(missing_ids),
    )

    return valid[["hearing_id", "hearing_date"]]


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


def build_member_lookup_from_hearing_members(hearings_members_df, members_df, hearings_df):
    """
    Build a member lookup from hearing_members data, augmenting coverage for congresses
    where members_terms.csv is incomplete (especially 118th Congress).

    Returns a DataFrame with the same schema as build_member_lookup():
        bioguide_id, last_name, first_name, party, state, congress, last_name_upper
    """
    # Join hearing_members with hearings to get congress numbers
    hm_with_congress = hearings_members_df.merge(hearings_df[["hearing_id", "congress"]], on="hearing_id", how="left")
    # Deduplicate: one entry per (bioguide_id, congress)
    hm_unique = hm_with_congress[["bioguide_id", "congress"]].drop_duplicates()

    # Merge with member info
    member_info = members_df[["bioguide_id", "last_name", "first_name", "party", "state"]].copy()
    retVal = hm_unique.merge(member_info, on="bioguide_id", how="left")
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


def optimize_dataframe(df):
    """
    Renames the columns and casts data types of the enriched DataFrame
    to optimize memory and speed for Streamlit and other downstream consumers.
    """
    df = df.copy()

    # Rename target_sentence to text if needed
    if "text" not in df.columns and "target_sentence" in df.columns:
        df = df.rename(columns={"target_sentence": "text"})

    # Define types to ensure consistency and minimize memory
    dtype = {
        "congress": "int16",
        "chamber": "category",
        "party": "category",
        "match_type": "category",
        "dem": "Int8",
        "minority": "Int8",
        "unified": "Int8",
        "minuni": "Int8",
        "freshman": "Int8",
        "chairspeech": "Int8",
        "rankmemspeech": "Int8",
        "leader": "Int8",
        "member_state": "category",
        "state_abbrev": "category",
        "female": "Int8",
        "district_code": "Int8",
        "seniority": "Int8",
        "seniority_sq": "Int16",
    }

    for col, dt in dtype.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except Exception as e:
                logger.warning("Could not cast column %s to %s: %s", col, dt, e)

    if "hearing_date" in df.columns:
        try:
            df["hearing_date"] = pd.to_datetime(df["hearing_date"])
        except Exception as e:
            logger.warning("Could not cast hearing_date to datetime: %s", e)

    return df
