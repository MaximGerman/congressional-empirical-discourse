import os

import pandas as pd
import pytest

from scripts.optimize_data import PARQUET_PATH


@pytest.mark.skipif(not os.path.exists(PARQUET_PATH), reason="Enriched data not found")
def test_enriched_data_schema():
    df = pd.read_parquet(PARQUET_PATH)

    expected_columns = [
        "congress",
        "chamber",
        "party",
        "bioguide_id",
        "text",
        "nominate_dim1",
        "seniority",
        "unified",
        "dem",
        "minority",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.skipif(not os.path.exists(PARQUET_PATH), reason="Enriched data not found")
def test_enriched_data_constraints():
    # Load a sample to keep it fast
    df = pd.read_parquet(PARQUET_PATH).sample(min(100000, 100000))

    # 1. Seniority >= 1
    if "seniority" in df.columns:
        assert (df["seniority"].dropna() >= 1).all(), "Found seniority < 1"

    # 2. Binary flags are 0 or 1
    binary_cols = ["dem", "minority", "unified", "freshman", "chairspeech", "rankmemspeech", "leader"]
    for col in binary_cols:
        if col in df.columns:
            valid_values = {0, 1}
            actual_values = set(df[col].dropna().unique())
            assert actual_values.issubset(valid_values), f"Column {col} has non-binary values: {actual_values}"

    # 3. Nominate scores for matched members
    if "nominate_dim1" in df.columns and "bioguide_id" in df.columns:
        matched = df[df["bioguide_id"].notna()]
        # Check if matched members have nominate scores (allow some nulls if Voteview coverage is < 100%)
        null_rate = matched["nominate_dim1"].isnull().mean()
        assert null_rate < 0.05, f"High null rate in nominate_dim1 for matched members: {null_rate:.2%}"

    # 4. Match score range
    if "match_score" in df.columns:
        valid_scores = df["match_score"].dropna()
        assert (valid_scores >= 0).all() and (valid_scores <= 100).all(), "match_score out of range [0, 100]"


@pytest.mark.skipif(not os.path.exists(PARQUET_PATH), reason="Enriched data not found")
def test_congress_ranges():
    df = pd.read_parquet(PARQUET_PATH, columns=["congress", "hearing_date"])
    df = df.dropna(subset=["hearing_date"])

    congress_ranges = {
        115: ("2017-01-03", "2019-01-03"),
        116: ("2019-01-03", "2021-01-03"),
        117: ("2021-01-03", "2023-01-03"),
        118: ("2023-01-03", "2025-01-03"),
    }

    for congress, (start, end) in congress_ranges.items():
        subset = df[df["congress"] == congress]
        if not subset.empty:
            assert (
                subset["hearing_date"] >= pd.to_datetime(start)
            ).all(), f"Congress {congress} has dates before {start}"
            # Allow some overlap for end dates as sessions can run slightly over or BICAM might have late filings
            assert (
                subset["hearing_date"] <= pd.to_datetime(end) + pd.Timedelta(days=90)
            ).all(), f"Congress {congress} has dates after {end}"
