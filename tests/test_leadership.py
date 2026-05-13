import pandas as pd
import pytest

from src.leadership import (
    COMMITTEE_LEADERS,
    load_committee_leaders,
    normalize_committee_code,
    prepare_leadership_enrichment,
)

# --- normalize_committee_code ---


class TestNormalizeCommitteeCode:
    def test_parent_committee(self):
        assert normalize_committee_code("hsag00") == "HSAG"

    def test_another_parent(self):
        assert normalize_committee_code("hsba00") == "HSBA"

    def test_senate_committee(self):
        assert normalize_committee_code("ssaf00") == "SSAF"

    def test_subcommittee_returns_empty(self):
        assert normalize_committee_code("hsag15") == ""

    def test_subcommittee_two_digits(self):
        assert normalize_committee_code("hsap10") == ""

    def test_empty_string(self):
        assert normalize_committee_code("") == ""

    def test_too_short(self):
        assert normalize_committee_code("hs") == ""

    def test_none_value(self):
        assert normalize_committee_code(None) == ""

    def test_non_string(self):
        assert normalize_committee_code(12345) == ""

    def test_preserves_case(self):
        """Output should be uppercase regardless of input case."""
        assert normalize_committee_code("HsAg00") == "HSAG"


# --- load_committee_leaders ---


class TestLoadCommitteeLeaders:
    def test_returns_expected_columns(self):
        df = load_committee_leaders(target_congresses=[115])
        assert list(df.columns) == ["bioguide_id", "congress", "thomas_id", "role"]

    def test_filter_by_congress(self):
        df = load_committee_leaders(target_congresses=[115])
        assert set(df["congress"].unique()) == {115}

    def test_multiple_congresses(self):
        df = load_committee_leaders(target_congresses=[115, 116])
        assert set(df["congress"].unique()) == {115, 116}

    def test_all_congresses(self):
        df = load_committee_leaders()
        assert set(df["congress"].unique()) == {115, 116, 117, 118}

    def test_no_matching_congress(self):
        df = load_committee_leaders(target_congresses=[200])
        assert len(df) == 0

    def test_role_values(self):
        df = load_committee_leaders(target_congresses=[115])
        assert set(df["role"].unique()) == {"chair", "ranking_member"}

    def test_known_chair(self):
        """Conaway was chair of Agriculture in the 115th."""
        df = load_committee_leaders(target_congresses=[115])
        ag_chairs = df[(df["thomas_id"] == "HSAG") & (df["role"] == "chair")]
        assert len(ag_chairs) == 1
        assert ag_chairs.iloc[0]["bioguide_id"] == "C001062"

    def test_known_ranking_member(self):
        """Peterson was ranking member of Agriculture in the 115th."""
        df = load_committee_leaders(target_congresses=[115])
        ag_rm = df[(df["thomas_id"] == "HSAG") & (df["role"] == "ranking_member")]
        assert len(ag_rm) == 1
        assert ag_rm.iloc[0]["bioguide_id"] == "P000258"

    def test_mid_congress_change_both_included(self):
        """115th HSRU had two ranking members (Slaughter died, McGovern succeeded)."""
        df = load_committee_leaders(target_congresses=[115])
        ru_rm = df[(df["thomas_id"] == "HSRU") & (df["role"] == "ranking_member")]
        assert len(ru_rm) == 2
        assert set(ru_rm["bioguide_id"]) == {"S000480", "M000312"}

    def test_116th_oversight_chair_change(self):
        """116th HSGO had two chairs (Cummings died, Maloney succeeded)."""
        df = load_committee_leaders(target_congresses=[116])
        go_chairs = df[(df["thomas_id"] == "HSGO") & (df["role"] == "chair")]
        assert len(go_chairs) == 2
        assert set(go_chairs["bioguide_id"]) == {"C000984", "M000087"}

    def test_no_duplicate_single_leader_entries(self):
        """Each committee should have exactly 1 chair per congress (except mid-congress changes)."""
        df = load_committee_leaders()
        # Exclude known multi-leader entries
        # 115th HSRU has 2 ranking members, 116th HSGO has 2 chairs
        for (congress, thomas_id), group in df.groupby(["congress", "thomas_id"]):
            for role, role_group in group.groupby("role"):
                if (congress, thomas_id, role) in {
                    (115, "HSRU", "ranking_member"),
                    (116, "HSGO", "chair"),
                }:
                    assert len(role_group) == 2
                else:
                    assert len(role_group) == 1, (
                        f"Duplicate {role} for ({congress}, {thomas_id}): {role_group['bioguide_id'].tolist()}"
                    )

    def test_expected_committee_count_per_congress(self):
        """Each congress should have at least 20 committees with leaders."""
        df = load_committee_leaders()
        for congress in [115, 116, 117, 118]:
            n_committees = df[df["congress"] == congress]["thomas_id"].nunique()
            assert n_committees >= 20, f"Congress {congress} has only {n_committees} committees"


# --- prepare_leadership_enrichment ---


@pytest.fixture
def sample_pipeline_df():
    """Synthetic pipeline DataFrame with known committee assignments."""
    return pd.DataFrame(
        {
            "hearing_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
            "bioguide_id": [
                "C001062",  # Conaway — 115th HSAG chair
                "P000258",  # Peterson — 115th HSAG ranking member
                "A000055",  # Aderholt — regular member, not a leader
                "C001062",  # Conaway — but on HSBA, not chair there
                None,  # No bioguide — unmatched speaker
                "C001062",  # Conaway — but wrong congress (116th, he's ranking member)
            ],
            "congress": [115, 115, 115, 115, 115, 116],
            "committee_code": [
                "hsag00",
                "hsag00",
                "hsag00",
                "hsba00",
                "hsag00",
                "hsag00",
            ],
            "speaker": ["Mr. Conaway", "Mr. Peterson", "Mr. Aderholt", "Mr. Conaway", "Unknown", "Mr. Conaway"],
        }
    )


class TestPrepareLeadershipEnrichment:
    def test_adds_expected_columns(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        assert "chairspeech" in result.columns
        assert "rankmemspeech" in result.columns
        assert "leader" in result.columns

    def test_chair_flagged_correctly(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Conaway is 115th HSAG chair
        row = result.iloc[0]
        assert row["chairspeech"] == 1
        assert row["rankmemspeech"] == 0
        assert row["leader"] == 1

    def test_ranking_member_flagged_correctly(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Peterson is 115th HSAG ranking member
        row = result.iloc[1]
        assert row["chairspeech"] == 0
        assert row["rankmemspeech"] == 1
        assert row["leader"] == 1

    def test_regular_member_not_flagged(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Aderholt is a regular member
        row = result.iloc[2]
        assert row["chairspeech"] == 0
        assert row["rankmemspeech"] == 0
        assert row["leader"] == 0

    def test_wrong_committee_not_flagged(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Conaway is HSAG chair, not HSBA chair
        row = result.iloc[3]
        assert row["chairspeech"] == 0
        assert row["rankmemspeech"] == 0
        assert row["leader"] == 0

    def test_missing_bioguide_not_flagged(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Unmatched speaker (no bioguide_id)
        row = result.iloc[4]
        assert row["chairspeech"] == 0
        assert row["rankmemspeech"] == 0
        assert row["leader"] == 0

    def test_ranking_member_in_different_congress(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        # Conaway is 116th HSAG *ranking member* (not chair)
        row = result.iloc[5]
        assert row["chairspeech"] == 0
        assert row["rankmemspeech"] == 1
        assert row["leader"] == 1

    def test_preserves_row_count(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        assert len(result) == len(sample_pipeline_df)

    def test_preserves_existing_columns(self, sample_pipeline_df):
        result = prepare_leadership_enrichment(sample_pipeline_df)
        for col in ["hearing_id", "bioguide_id", "congress", "committee_code", "speaker"]:
            assert col in result.columns

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["hearing_id", "bioguide_id", "congress", "committee_code"])
        result = prepare_leadership_enrichment(empty)
        assert len(result) == 0
        assert "chairspeech" in result.columns
        assert "rankmemspeech" in result.columns
        assert "leader" in result.columns

    def test_subcommittee_code_no_match(self):
        """Subcommittee codes should not match any leader."""
        df = pd.DataFrame(
            {
                "hearing_id": ["h1"],
                "bioguide_id": ["C001062"],  # Conaway — HSAG chair
                "congress": [115],
                "committee_code": ["hsag15"],  # Subcommittee!
            }
        )
        result = prepare_leadership_enrichment(df)
        assert result.iloc[0]["chairspeech"] == 0
        assert result.iloc[0]["leader"] == 0

    def test_chair_and_ranking_member_mutually_exclusive(self, sample_pipeline_df):
        """No single row should have both chairspeech=1 and rankmemspeech=1."""
        result = prepare_leadership_enrichment(sample_pipeline_df)
        both = (result["chairspeech"] == 1) & (result["rankmemspeech"] == 1)
        assert not both.any()

    def test_leader_equals_chair_or_rm(self, sample_pipeline_df):
        """leader should be 1 whenever chairspeech=1 OR rankmemspeech=1."""
        result = prepare_leadership_enrichment(sample_pipeline_df)
        expected = ((result["chairspeech"] == 1) | (result["rankmemspeech"] == 1)).astype(int)
        pd.testing.assert_series_equal(result["leader"], expected, check_names=False)


# --- Cross-validation against known data ---


class TestCrossValidation:
    """Spot-check a few well-known committee leaders across congresses."""

    def test_nadler_judiciary_115_ranking(self):
        """Nadler was ranking member of Judiciary in 115th."""
        entry = COMMITTEE_LEADERS[(115, "HSJU")]
        assert entry["ranking_member"] == "N000002"

    def test_nadler_judiciary_116_chair(self):
        """Nadler became chair of Judiciary in 116th (Democratic majority)."""
        entry = COMMITTEE_LEADERS[(116, "HSJU")]
        assert entry["chair"] == "N000002"

    def test_waters_financial_services_116_chair(self):
        """Waters became chair of Financial Services in 116th."""
        entry = COMMITTEE_LEADERS[(116, "HSBA")]
        assert entry["chair"] == "W000187"

    def test_jordan_judiciary_118_chair(self):
        """Jordan became chair of Judiciary in 118th."""
        entry = COMMITTEE_LEADERS[(118, "HSJU")]
        assert entry["chair"] == "J000289"

    def test_majority_party_holds_chair(self):
        """Verify that in Republican-majority congresses (115, 118), known
        Republican chairs are in the chair slot, and in Democratic-majority
        congresses (116, 117), known Democratic chairs are in chair slot."""
        # 115th: Republican majority → Conaway (R) chairs Agriculture
        assert COMMITTEE_LEADERS[(115, "HSAG")]["chair"] == "C001062"
        # 116th: Democratic majority → Peterson (D) chairs Agriculture
        assert COMMITTEE_LEADERS[(116, "HSAG")]["chair"] == "P000258"
        # 117th: Democratic majority → David Scott (D) chairs Agriculture
        assert COMMITTEE_LEADERS[(117, "HSAG")]["chair"] == "S001157"
        # 118th: Republican majority → Glenn Thompson (R) chairs Agriculture
        assert COMMITTEE_LEADERS[(118, "HSAG")]["chair"] == "T000467"
