import pandas as pd

from src.data import (
    build_hearing_member_map,
    build_member_lookup,
    build_member_lookup_from_hearing_members,
    filter_hearings_by_congress,
    get_majority_status,
    resolve_hearing_dates,
)


def test_filter_hearings_by_congress():
    df = pd.DataFrame(
        {
            "hearing_id": [1, 2, 3],
            "congress": [114, 115, 116],
            "chamber": ["house", "house", "senate"],
        }
    )

    filtered = filter_hearings_by_congress(df, min_congress=115, chamber="house")
    assert len(filtered) == 1
    assert filtered.iloc[0]["hearing_id"] == 2


def test_filter_hearings_by_congress_no_chamber():
    df = pd.DataFrame(
        {
            "hearing_id": [1, 2, 3],
            "congress": [114, 115, 116],
            "chamber": ["house", "house", "senate"],
        }
    )

    filtered = filter_hearings_by_congress(df, min_congress=115, chamber=None)
    assert len(filtered) == 2
    assert set(filtered["hearing_id"]) == {2, 3}


def test_build_member_lookup():
    members = pd.DataFrame(
        {
            "bioguide_id": ["A000001", "B000002"],
            "last_name": ["Smith", "Jones"],
            "first_name": ["John", "Mary"],
            "party": ["Republican", "Democratic"],
            "state": ["CA", "NY"],
        }
    )
    terms = pd.DataFrame(
        {
            "bioguide_id": ["A000001", "B000002", "B000002"],
            "congress": [115, 115, 116],
            "chamber": ["house", "senate", "house"],
            "state_code": ["CA", "NY", "NY"],
        }
    )

    lookup = build_member_lookup(members, terms, target_congresses=[115, 116])
    assert len(lookup) == 2  # B000002 senate term is filtered out

    smith = lookup[lookup["last_name_upper"] == "SMITH"].iloc[0]
    assert smith["congress"] == 115
    assert smith["bioguide_id"] == "A000001"

    jones = lookup[lookup["last_name_upper"] == "JONES"].iloc[0]
    assert jones["congress"] == 116
    assert jones["bioguide_id"] == "B000002"


def test_build_hearing_member_map():
    hearings_members = pd.DataFrame(
        {
            "hearing_id": [1, 1],
            "bioguide_id": ["A000001", "B000002"],
        }
    )
    members = pd.DataFrame(
        {
            "bioguide_id": ["A000001", "B000002"],
            "last_name": ["Smith", "Jones"],
            "first_name": ["John", "Mary"],
            "party": ["Republican", "Democratic"],
            "state": ["CA", "NY"],
        }
    )

    hm_map = build_hearing_member_map(hearings_members, members)
    assert len(hm_map) == 2
    assert "last_name_upper" in hm_map.columns
    assert set(hm_map["last_name_upper"]) == {"SMITH", "JONES"}


def test_build_member_lookup_from_hearing_members():
    hearings_members = pd.DataFrame(
        {
            "hearing_id": [10, 10, 20],
            "bioguide_id": ["A000001", "B000002", "A000001"],
        }
    )
    members = pd.DataFrame(
        {
            "bioguide_id": ["A000001", "B000002"],
            "last_name": ["Smith", "Jones"],
            "first_name": ["John", "Mary"],
            "party": ["Republican", "Democratic"],
            "state": ["CA", "NY"],
        }
    )
    hearings = pd.DataFrame(
        {
            "hearing_id": [10, 20],
            "congress": [118, 118],
            "chamber": ["house", "house"],
        }
    )

    lookup = build_member_lookup_from_hearing_members(hearings_members, members, hearings)
    # Should have 2 unique (bioguide_id, congress) pairs
    assert len(lookup) == 2
    assert set(lookup["bioguide_id"]) == {"A000001", "B000002"}
    assert all(lookup["congress"] == 118)
    assert "last_name_upper" in lookup.columns


def test_get_majority_status_majority():
    # 115th was Republican majority
    assert get_majority_status("Republican", 115) == 0


def test_get_majority_status_minority():
    # 115th was Republican majority
    assert get_majority_status("Democratic", 115) == 1


def test_get_majority_status_unknown_congress():
    assert get_majority_status("Republican", 999) is None


def test_get_majority_status_nan_party():
    assert get_majority_status(float("nan"), 115) is None


def test_get_majority_status_democrat_alias():
    # 116th was Democratic majority
    assert get_majority_status("Democrat", 116) == 0
    assert get_majority_status("Democratic", 116) == 0


# --- resolve_hearing_dates tests ---


def test_resolve_hearing_dates_picks_in_range_date():
    """When multiple dates exist, picks the one in the congress date range."""
    dates = pd.DataFrame(
        {
            "hearing_id": ["h1", "h1", "h1"],
            "hearing_date": ["2005-10-06", "2017-02-07", "2017-05-15"],
        }
    )
    hearings = pd.DataFrame({"hearing_id": ["h1"], "congress": [115]})

    result = resolve_hearing_dates(dates, hearings)
    assert len(result) == 1
    assert result.iloc[0]["hearing_id"] == "h1"
    # Should pick earliest in-range date (2017-02-07), not the old 2005 date
    assert result.iloc[0]["hearing_date"].year == 2017
    assert result.iloc[0]["hearing_date"].month == 2


def test_resolve_hearing_dates_single_valid_date():
    """Single date within range is returned as-is."""
    dates = pd.DataFrame({"hearing_id": ["h1"], "hearing_date": ["2020-03-15"]})
    hearings = pd.DataFrame({"hearing_id": ["h1"], "congress": [116]})

    result = resolve_hearing_dates(dates, hearings)
    assert len(result) == 1
    assert result.iloc[0]["hearing_date"].year == 2020


def test_resolve_hearing_dates_no_in_range_falls_back_to_latest():
    """When no date is in range, falls back to the latest date."""
    dates = pd.DataFrame(
        {
            "hearing_id": ["h1", "h1"],
            "hearing_date": ["1995-01-01", "2005-06-15"],
        }
    )
    hearings = pd.DataFrame({"hearing_id": ["h1"], "congress": [115]})

    result = resolve_hearing_dates(dates, hearings)
    assert len(result) == 1
    # Neither date is in 115th range (2017-2019), so pick the latest: 2005
    assert result.iloc[0]["hearing_date"].year == 2005


def test_resolve_hearing_dates_multiple_hearings():
    """Resolves dates independently for each hearing."""
    dates = pd.DataFrame(
        {
            "hearing_id": ["h1", "h1", "h2", "h2"],
            "hearing_date": ["2005-01-01", "2017-06-01", "2010-01-01", "2020-03-01"],
        }
    )
    hearings = pd.DataFrame(
        {
            "hearing_id": ["h1", "h2"],
            "congress": [115, 116],
        }
    )

    result = resolve_hearing_dates(dates, hearings)
    assert len(result) == 2
    h1 = result[result["hearing_id"] == "h1"].iloc[0]
    h2 = result[result["hearing_id"] == "h2"].iloc[0]
    assert h1["hearing_date"].year == 2017  # in 115th range, not 2005
    assert h2["hearing_date"].year == 2020  # in 116th range (2019-2021), not 2010


def test_resolve_hearing_dates_unknown_congress_falls_back():
    """Hearing with congress not in CONGRESS_DATE_RANGES falls back to latest date."""
    dates = pd.DataFrame(
        {
            "hearing_id": ["h1", "h1"],
            "hearing_date": ["2010-01-01", "2012-06-15"],
        }
    )
    hearings = pd.DataFrame({"hearing_id": ["h1"], "congress": [112]})

    result = resolve_hearing_dates(dates, hearings)
    assert len(result) == 1
    assert result.iloc[0]["hearing_date"].year == 2012
