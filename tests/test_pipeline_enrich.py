import pandas as pd
import pytest

from src.pipeline_enrich import (
    _apply_enrichments,
    _build_unique_matchable,
    _filter_witnesses,
    _match_speakers_to_members,
    _vectorized_merge_match,
)


@pytest.fixture
def mock_legislators_df():
    """Simple sentences/legislators dataframe for matching tests."""
    return pd.DataFrame(
        {
            "hearing_id": ["H1", "H1", "H2", "H2", "H2"],
            "speaker": ["Mr. SMITH", "Chairman JONES", "Mr. SMITH", "Ms. DOE", "The Chairman"],
            "speaker_last_name": ["SMITH", "JONES", "SMITH", "DOE", "CHAIRMAN"],
            "speaker_last_word": ["SMITH", "JONES", "SMITH", "DOE", "CHAIRMAN"],
            "congress": [115, 115, 116, 116, 116],
            "committee_code": ["hsgo00", "hsgo00", "hsgo00", "hsas00", "hsas00"],
        }
    )


@pytest.fixture
def mock_member_lookup():
    """Mock member database for congress-level matching."""
    return pd.DataFrame(
        {
            "bioguide_id": ["S001", "J002", "D003"],
            "last_name": ["Smith", "Jones", "Doe"],
            "last_name_upper": ["SMITH", "JONES", "DOE"],
            "first_name": ["Adam", "Bob", "Jane"],
            "party": ["Republican", "Democratic", "Republican"],
            "state": ["CA", "NY", "TX"],
            "congress": [115, 115, 116],
        }
    )


@pytest.fixture
def mock_hearing_member_map():
    """Mock hearing-member map for direct hearing-level matching."""
    return pd.DataFrame(
        {
            "hearing_id": ["H1", "H1"],
            "bioguide_id": ["S001", "J002"],
            "last_name": ["Smith", "Jones"],
            "last_name_upper": ["SMITH", "JONES"],
            "party": ["Republican", "Democratic"],
        }
    )


def test_vectorized_merge_match_basic(mock_legislators_df, mock_member_lookup):
    # Prepare inputs
    pairs = mock_legislators_df[["congress", "speaker_last_name", "speaker_last_word"]].drop_duplicates()
    matchable = mock_member_lookup.copy()

    # Run matching on congress level
    result = _vectorized_merge_match(pairs, matchable, "congress", "congress_exact")

    assert "bioguide_id" in result.columns
    # SMITH 115 matches
    assert (
        result.loc[(result["speaker_last_name"] == "SMITH") & (result["congress"] == 115), "bioguide_id"].iloc[0]
        == "S001"
    )
    # DOE 116 matches
    assert (
        result.loc[(result["speaker_last_name"] == "DOE") & (result["congress"] == 116), "bioguide_id"].iloc[0]
        == "D003"
    )
    # SMITH 116 does NOT match (not in lookup)
    assert pd.isna(
        result.loc[(result["speaker_last_name"] == "SMITH") & (result["congress"] == 116), "bioguide_id"].iloc[0]
    )


def test_vectorized_merge_match_fallback():
    # Test case where speaker_last_name is "DE LA CRUZ" but last_name_upper is just "CRUZ"
    pairs = pd.DataFrame({"group_id": [1], "speaker_last_name": ["DE LA CRUZ"], "speaker_last_word": ["CRUZ"]})
    matchable = pd.DataFrame({"group_id": [1], "last_name_upper": ["CRUZ"], "bioguide_id": ["C123"]})

    result = _vectorized_merge_match(pairs, matchable, "group_id", "test_match")

    assert result["bioguide_id"].iloc[0] == "C123"
    assert result["match_type"].iloc[0] == "test_match"


def test_build_unique_matchable():
    # Ambiguous case: two SMITHS in same congress
    df = pd.DataFrame(
        {"congress": [115, 115, 115], "last_name_upper": ["SMITH", "SMITH", "JONES"], "bioguide_id": ["S1", "S2", "J1"]}
    )

    unique = _build_unique_matchable(df, ["congress"])

    assert len(unique) == 1
    assert unique["last_name_upper"].iloc[0] == "JONES"
    assert unique["bioguide_id"].iloc[0] == "J1"


def test_filter_witnesses():
    df = pd.DataFrame(
        {
            "speaker": ["Mr. SMITH", "Dr. EXPERT", "Ms. DOE", "Professor BRAIN"],
            "text": ["Hello", "I am an expert", "Hi", "Science is cool"],
        }
    )

    # We need to mock is_likely_witness or just test how it's called
    # Since it's a simple apply, we're testing the logic in src.preprocess indirectly
    result = _filter_witnesses(df)

    # Based on src.preprocess (which I should check), titles like Dr., Professor are likely witnesses
    # Actually let's check src/preprocess.py for the exact logic
    assert "is_witness" in result.columns
    assert len(result) == 2  # Mr. SMITH and Ms. DOE remain


def test_match_speakers_to_members_integration(
    monkeypatch, mock_legislators_df, mock_member_lookup, mock_hearing_member_map
):
    # Mock data loading functions in pipeline_enrich
    monkeypatch.setattr("src.pipeline_enrich.load_members", lambda: pd.DataFrame())
    monkeypatch.setattr("src.pipeline_enrich.load_members_terms", lambda: pd.DataFrame())
    monkeypatch.setattr("src.pipeline_enrich.build_member_lookup", lambda m, t: mock_member_lookup)
    monkeypatch.setattr("src.pipeline_enrich.load_hearings_members", lambda: pd.DataFrame())
    monkeypatch.setattr(
        "src.pipeline_enrich.build_member_lookup_from_hearing_members", lambda hm, m, era: pd.DataFrame()
    )
    monkeypatch.setattr("src.pipeline_enrich.build_hearing_member_map", lambda hm, m: mock_hearing_member_map)

    # We also need to mock match_speaker_to_member for fuzzy matching
    monkeypatch.setattr("src.pipeline_enrich.match_speaker_to_member", lambda *args, **kwargs: None)

    # new_era mock
    new_era = pd.DataFrame()

    result = _match_speakers_to_members(mock_legislators_df, new_era)

    assert "bioguide_id" in result.columns
    # Check H1 matches (hearing_member_exact)
    h1_smith = result[(result["hearing_id"] == "H1") & (result["speaker_last_name"] == "SMITH")]
    assert h1_smith["bioguide_id"].iloc[0] == "S001"
    assert h1_smith["match_type"].iloc[0] == "hearing_member_exact"

    # Check H2 DOE match (congress_exact)
    h2_doe = result[(result["hearing_id"] == "H2") & (result["speaker_last_name"] == "DOE")]
    # Wait, DOE is in congress 116 in mock_member_lookup, and H2 is congress 116
    assert h2_doe["bioguide_id"].iloc[0] == "D003"
    assert h2_doe["match_type"].iloc[0] == "congress_exact"


def test_apply_enrichments_integration(monkeypatch):
    df = pd.DataFrame(
        {
            "bioguide_id": ["S001", "J002"],
            "congress": [115, 116],
            "party": ["Republican", "Democratic"],
            "state_abbrev": ["CA", "NY"],
            "district_code": [1, 2],
        }
    )

    # Mock external enrichment functions
    monkeypatch.setattr(
        "src.pipeline_enrich.prepare_voteview_enrichment",
        lambda target_congresses: pd.DataFrame(
            {
                "bioguide_id": ["S001", "J002"],
                "congress": [115, 116],
                "nominate_dim1": [0.5, -0.5],
                "seniority": [10, 5],
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.prepare_leadership_enrichment", lambda df: df.assign(chairspeech=0, rankmemspeech=0)
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.load_elections_data",
        lambda target_congresses: pd.DataFrame(
            {"state_abbrev": ["CA", "NY"], "district_code": [1, 2], "congress": [115, 116], "vote_pct": [60.0, 55.0]}
        ),
    )

    result = _apply_enrichments(df)

    assert "nominate_dim1" in result.columns
    assert "vote_pct" in result.columns
    assert "minority" in result.columns
    # 115th R is majority -> minority=0
    assert result.loc[result["bioguide_id"] == "S001", "minority"].iloc[0] == 0
    # 116th D is majority -> minority=0
    assert result.loc[result["bioguide_id"] == "J002", "minority"].iloc[0] == 0


def test_match_speakers_multi_chair_edge_case(monkeypatch):
    """Test that a committee with multiple chairs doesn't cause a row explosion."""
    import pandas as pd

    from src.pipeline_enrich import _match_speakers_to_members

    mock_legislators_df = pd.DataFrame(
        {
            "hearing_id": ["H1"],
            "speaker": ["The Chairman"],
            "speaker_last_name": ["CHAIRMAN"],
            "speaker_last_word": ["CHAIRMAN"],
            "congress": [116],
            "committee_code": ["hsgo00"],
        }
    )

    # Mock multiple chairs for the same committee
    mock_leaders_df = pd.DataFrame(
        {
            "congress": [116, 116],
            "thomas_id": ["HSGO", "HSGO"],
            "bioguide_id": ["C001", "C002"],
            "role": ["chair", "chair"],
        }
    )

    monkeypatch.setattr(
        "src.pipeline_enrich.load_members",
        lambda: pd.DataFrame(
            {
                "bioguide_id": ["C001", "C002"],
                "last_name": ["Chair1", "Chair2"],
                "first_name": ["A", "B"],
                "party": ["R", "R"],
                "state": ["NY", "NY"],
            }
        ),
    )
    monkeypatch.setattr("src.pipeline_enrich.load_members_terms", lambda: pd.DataFrame())
    monkeypatch.setattr(
        "src.pipeline_enrich.build_member_lookup",
        lambda m, t: pd.DataFrame(columns=["bioguide_id", "congress", "last_name_upper", "party", "state"]),
    )
    monkeypatch.setattr("src.pipeline_enrich.load_hearings_members", lambda: pd.DataFrame())
    monkeypatch.setattr(
        "src.pipeline_enrich.build_member_lookup_from_hearing_members",
        lambda hm, m, era: pd.DataFrame(columns=["bioguide_id", "congress", "last_name_upper", "party", "state"]),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.build_hearing_member_map",
        lambda hm, m: pd.DataFrame(columns=["hearing_id", "bioguide_id", "last_name_upper"]),
    )
    monkeypatch.setattr("src.pipeline_enrich.match_speaker_to_member", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.leadership.load_committee_leaders", lambda target_congresses: mock_leaders_df)

    result = _match_speakers_to_members(mock_legislators_df, pd.DataFrame())

    # Assert row count hasn't exploded
    assert len(result) == 1
    # Assert we picked the first chair
    assert result.iloc[0]["bioguide_id"] == "C001"
