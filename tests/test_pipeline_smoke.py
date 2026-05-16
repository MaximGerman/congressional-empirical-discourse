import os

import pandas as pd
import pytest

from src.pipeline import run_pipeline


@pytest.fixture
def mock_pipeline_data(tmp_path, monkeypatch):
    """Setup mock environment and data for pipeline smoke test."""

    # Mock directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("src.pipeline.OUTPUT_DIR", str(data_dir))

    # Mock data loading in Step 1
    mock_hearings = pd.DataFrame(
        {
            "hearing_id": ["H1", "H2"],
            "congress": [115, 116],
            "chamber": ["house", "house"],
            "title": ["Test 1", "Test 2"],
        }
    )
    monkeypatch.setattr("src.pipeline.load_hearings", lambda: mock_hearings)

    # Mock Step 2: load transcripts (must be > 100 chars and have 2+ leading spaces)
    long_text_1 = "  Chairman SMITH. Sentence 1. " + "Sentence. " * 20 + "\n  Mr. JONES. Sentence 3."
    long_text_2 = "  Chairman DOE. Sentence 4. " + "Sentence. " * 20 + "\n  Mr. BROWN. Sentence 5."
    mock_texts = pd.DataFrame({"hearing_id": ["H1", "H2"], "raw_text": [long_text_1, long_text_2]})
    monkeypatch.setattr("src.pipeline.load_hearings_texts_chunked", lambda ids: mock_texts)

    # Mock Step 4 dependencies (to avoid actual network/large file loads)
    monkeypatch.setattr(
        "src.pipeline_enrich.load_hearings_committees",
        lambda: pd.DataFrame(columns=["hearing_id", "committee_code", "committee_name"]),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.load_hearings_dates",
        lambda: pd.DataFrame({"hearing_id": ["H1", "H2"], "hearing_date": ["2018-01-01", "2020-01-01"]}),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.resolve_hearing_dates",
        lambda d, e: pd.DataFrame({"hearing_id": ["H1", "H2"], "hearing_date": ["2018-01-01", "2020-01-01"]}),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.load_members",
        lambda: pd.DataFrame(
            {
                "bioguide_id": ["S1", "J1", "D1", "B1"],
                "last_name": ["Smith", "Jones", "Doe", "Brown"],
                "first_name": ["Adam", "Bob", "Jane", "Alice"],
                "party": ["Republican", "Democratic", "Republican", "Democratic"],
                "state": ["CA", "NY", "TX", "FL"],
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.load_members_terms",
        lambda: pd.DataFrame(
            {
                "bioguide_id": ["S1", "J1", "D1", "B1"],
                "congress": [115, 115, 116, 116],
                "chamber": ["house", "house", "house", "house"],
                "state_code": ["CA", "NY", "TX", "FL"],
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipeline_enrich.load_hearings_members", lambda: pd.DataFrame(columns=["hearing_id", "bioguide_id"])
    )

    # Mock external enrichments
    monkeypatch.setattr(
        "src.pipeline_enrich.prepare_voteview_enrichment",
        lambda target_congresses: pd.DataFrame(
            columns=["bioguide_id", "congress", "nominate_dim1", "seniority", "state_abbrev", "district_code"]
        ),
    )
    monkeypatch.setattr("src.pipeline_enrich.prepare_leadership_enrichment", lambda df: df)
    monkeypatch.setattr(
        "src.pipeline_enrich.load_elections_data",
        lambda target_congresses: pd.DataFrame(columns=["state_abbrev", "district_code", "congress", "vote_pct"]),
    )

    return data_dir


def test_run_pipeline_smoke(mock_pipeline_data):
    """Verify that the full pipeline runs and produces output files."""
    run_pipeline()

    # Check that output files were created
    assert os.path.exists(os.path.join(mock_pipeline_data, "sentences_raw.csv"))
    assert os.path.exists(os.path.join(mock_pipeline_data, "sentences_enriched.parquet"))
    assert os.path.exists(os.path.join(mock_pipeline_data, "sample_for_labeling.csv"))

    # Verify content of enriched data
    enriched_df = pd.read_parquet(os.path.join(mock_pipeline_data, "sentences_enriched.parquet"))
    assert len(enriched_df) > 0
    assert "speaker" in enriched_df.columns
    assert "bioguide_id" in enriched_df.columns

    # In our mock text, "Chairman SMITH" should match "S1"
    smith_rows = enriched_df[enriched_df["speaker"].str.contains("SMITH", na=False)]
    assert not smith_rows.empty
    # The matching logic is complex, but in this simple mock it should find a match
    # depending on how exact/congress matching behaves with mock data
