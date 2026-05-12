import pandas as pd
import pytest

from src.voteview import compute_seniority, load_voteview_members, prepare_voteview_enrichment

# --- Fixtures ---


def _make_voteview_df(include_gender=True):
    """
    Synthetic Voteview data spanning multiple congresses with known properties.

    Members:
      - A000001 (Smith): served 113-117 in House (seniority 1-5 by congress)
      - B000002 (Jones): served 115-116 in House (seniority 1-2)
      - C000003 (Lee): served 117 in House only (freshman)
      - D000004 (Brown): served 115 in Senate (should be excluded from House computations)
    """
    rows = [
        # Smith: long-serving member
        {
            "congress": 113,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.3,
            "nominate_dim2": 0.1,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        {
            "congress": 114,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.32,
            "nominate_dim2": 0.12,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        {
            "congress": 115,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.35,
            "nominate_dim2": 0.15,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        {
            "congress": 116,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.38,
            "nominate_dim2": 0.18,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        {
            "congress": 117,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.40,
            "nominate_dim2": 0.20,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        # Jones: two terms
        {
            "congress": 115,
            "chamber": "House",
            "bioguide_id": "B000002",
            "nominate_dim1": 0.5,
            "nominate_dim2": -0.2,
            "party_code": 200,
            "bioname": "JONES, Mary",
        },
        {
            "congress": 116,
            "chamber": "House",
            "bioguide_id": "B000002",
            "nominate_dim1": 0.52,
            "nominate_dim2": -0.22,
            "party_code": 200,
            "bioname": "JONES, Mary",
        },
        # Lee: freshman in 117th
        {
            "congress": 117,
            "chamber": "House",
            "bioguide_id": "C000003",
            "nominate_dim1": 0.1,
            "nominate_dim2": 0.05,
            "party_code": 100,
            "bioname": "LEE, Pat",
        },
        # Brown: Senate member (should not appear in House seniority)
        {
            "congress": 115,
            "chamber": "Senate",
            "bioguide_id": "D000004",
            "nominate_dim1": -0.6,
            "nominate_dim2": 0.3,
            "party_code": 100,
            "bioname": "BROWN, Sam",
        },
    ]

    if include_gender:
        genders = ["M", "M", "M", "M", "M", "F", "F", "X", "M"]
        for i, row in enumerate(rows):
            row["gender"] = genders[i]

    return pd.DataFrame(rows)


@pytest.fixture
def voteview_df():
    return _make_voteview_df(include_gender=True)


@pytest.fixture
def voteview_df_no_gender():
    return _make_voteview_df(include_gender=False)


@pytest.fixture
def voteview_csv(tmp_path, voteview_df):
    """Write synthetic Voteview data to a CSV file."""
    path = tmp_path / "HSall_members.csv"
    voteview_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def voteview_csv_no_gender(tmp_path, voteview_df_no_gender):
    path = tmp_path / "HSall_members_no_gender.csv"
    voteview_df_no_gender.to_csv(path, index=False)
    return str(path)


# --- load_voteview_members ---


def test_load_voteview_members_all(voteview_csv):
    df = load_voteview_members(path=voteview_csv, congress_range=None, chamber=None)
    assert len(df) == 9  # all rows


def test_load_voteview_members_house_only(voteview_csv):
    df = load_voteview_members(path=voteview_csv, chamber="House")
    assert len(df) == 8  # excludes Senate Brown
    assert "D000004" not in df["bioguide_id"].values


def test_load_voteview_members_congress_range(voteview_csv):
    df = load_voteview_members(path=voteview_csv, congress_range=(115, 116), chamber="House")
    assert set(df["congress"].unique()) == {115, 116}
    assert len(df) == 4  # Smith 115+116, Jones 115+116


def test_load_voteview_members_empty_range(voteview_csv):
    df = load_voteview_members(path=voteview_csv, congress_range=(200, 201), chamber="House")
    assert len(df) == 0


# --- compute_seniority ---


def test_compute_seniority_all(voteview_df):
    result = compute_seniority(voteview_df)
    # Smith: 5 terms (113-117)
    smith = result[result["bioguide_id"] == "A000001"].sort_values("congress")
    assert list(smith["seniority"]) == [1, 2, 3, 4, 5]
    assert list(smith["freshman"]) == [1, 0, 0, 0, 0]

    # Jones: 2 terms (115-116)
    jones = result[result["bioguide_id"] == "B000002"].sort_values("congress")
    assert list(jones["seniority"]) == [1, 2]
    assert list(jones["freshman"]) == [1, 0]

    # Lee: 1 term (117)
    lee = result[result["bioguide_id"] == "C000003"]
    assert len(lee) == 1
    assert lee.iloc[0]["seniority"] == 1
    assert lee.iloc[0]["freshman"] == 1


def test_compute_seniority_excludes_senate(voteview_df):
    """Senate members should not appear in House seniority."""
    result = compute_seniority(voteview_df)
    assert "D000004" not in result["bioguide_id"].values


def test_compute_seniority_target_congresses(voteview_df):
    """When target_congresses is set, only those congresses are returned."""
    result = compute_seniority(voteview_df, target_congresses=[115, 116])
    assert set(result["congress"].unique()) == {115, 116}

    # Smith should have seniority 3 in 115th (113, 114, 115)
    smith_115 = result[(result["bioguide_id"] == "A000001") & (result["congress"] == 115)]
    assert smith_115.iloc[0]["seniority"] == 3
    assert smith_115.iloc[0]["freshman"] == 0

    # Jones should be freshman in 115th
    jones_115 = result[(result["bioguide_id"] == "B000002") & (result["congress"] == 115)]
    assert jones_115.iloc[0]["seniority"] == 1
    assert jones_115.iloc[0]["freshman"] == 1


def test_compute_seniority_empty_dataframe():
    df = pd.DataFrame(columns=["bioguide_id", "congress", "chamber"])
    result = compute_seniority(df)
    assert len(result) == 0
    assert list(result.columns) == ["bioguide_id", "congress", "seniority", "seniority_sq", "freshman"]


# --- prepare_voteview_enrichment ---


def test_prepare_voteview_enrichment_columns(voteview_csv):
    result = prepare_voteview_enrichment(path=voteview_csv, target_congresses=[115, 116, 117])
    expected_cols = {
        "bioguide_id",
        "congress",
        "nominate_dim1",
        "nominate_dim2",
        "abs_dwnom1",
        "gender",
        "female",
        "seniority",
        "seniority_sq",
        "freshman",
    }
    assert expected_cols == set(result.columns)


def test_prepare_voteview_enrichment_abs_dwnom1(voteview_csv):
    result = prepare_voteview_enrichment(path=voteview_csv, target_congresses=[115])
    smith = result[result["bioguide_id"] == "A000001"].iloc[0]
    assert smith["abs_dwnom1"] == pytest.approx(0.35)
    assert smith["nominate_dim1"] == pytest.approx(-0.35)

    jones = result[result["bioguide_id"] == "B000002"].iloc[0]
    assert jones["abs_dwnom1"] == pytest.approx(0.5)


def test_prepare_voteview_enrichment_seniority(voteview_csv):
    result = prepare_voteview_enrichment(path=voteview_csv, target_congresses=[115, 117])

    # Smith in 115th: 3rd term (113, 114, 115)
    smith_115 = result[(result["bioguide_id"] == "A000001") & (result["congress"] == 115)].iloc[0]
    assert smith_115["seniority"] == 3
    assert smith_115["freshman"] == 0

    # Smith in 117th: 5th term
    smith_117 = result[(result["bioguide_id"] == "A000001") & (result["congress"] == 117)].iloc[0]
    assert smith_117["seniority"] == 5

    # Lee in 117th: freshman
    lee = result[(result["bioguide_id"] == "C000003") & (result["congress"] == 117)].iloc[0]
    assert lee["seniority"] == 1
    assert lee["freshman"] == 1


def test_prepare_voteview_enrichment_no_gender(voteview_csv_no_gender):
    result = prepare_voteview_enrichment(path=voteview_csv_no_gender, target_congresses=[115])
    assert "gender" not in result.columns
    assert "female" not in result.columns
    # Other columns should still be present
    assert "nominate_dim1" in result.columns
    assert "seniority" in result.columns


def test_prepare_voteview_enrichment_with_gender(voteview_csv):
    result = prepare_voteview_enrichment(path=voteview_csv, target_congresses=[115])
    assert "gender" in result.columns
    assert "female" in result.columns
    smith = result[result["bioguide_id"] == "A000001"].iloc[0]
    assert smith["gender"] == "M"
    assert smith["female"] == 0
    jones = result[result["bioguide_id"] == "B000002"].iloc[0]
    assert jones["gender"] == "F"
    assert jones["female"] == 1


def test_prepare_voteview_enrichment_excludes_senate(voteview_csv):
    result = prepare_voteview_enrichment(path=voteview_csv, target_congresses=[115])
    assert "D000004" not in result["bioguide_id"].values


def test_prepare_voteview_enrichment_deduplicates(tmp_path):
    """Members with duplicate (bioguide_id, congress) rows should be deduplicated."""
    rows = [
        {
            "congress": 115,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.3,
            "nominate_dim2": 0.1,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
        # Duplicate row (e.g., special election + regular term)
        {
            "congress": 115,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": -0.3,
            "nominate_dim2": 0.1,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "dupes.csv"
    df.to_csv(path, index=False)

    result = prepare_voteview_enrichment(path=str(path), target_congresses=[115])
    assert len(result) == 1


def test_prepare_voteview_enrichment_missing_nominate(tmp_path):
    """Members with NaN NOMINATE scores should be preserved (not dropped)."""
    rows = [
        {
            "congress": 115,
            "chamber": "House",
            "bioguide_id": "A000001",
            "nominate_dim1": None,
            "nominate_dim2": None,
            "party_code": 100,
            "bioname": "SMITH, John",
        },
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "missing_nom.csv"
    df.to_csv(path, index=False)

    result = prepare_voteview_enrichment(path=str(path), target_congresses=[115])
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["nominate_dim1"])
    assert pd.isna(result.iloc[0]["abs_dwnom1"])
