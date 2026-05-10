import pandas as pd

from src.data import (
    build_hearing_member_map,
    build_member_lookup,
    filter_hearings_by_congress,
    get_majority_status,
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
