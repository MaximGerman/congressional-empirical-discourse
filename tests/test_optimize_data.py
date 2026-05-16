import os

import pandas as pd
import pytest

from scripts.optimize_data import convert_csv_to_parquet


@pytest.fixture
def mock_csv(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "test_data.csv"
    data = {
        "congress": [115, 116, 117],
        "chamber": ["House", "House", "House"],
        "party": ["Republican", "Democratic", "Republican"],
        "speaker": ["Smith", "Jones", "Doe"],
        "target_sentence": ["Sentence 1", "Sentence 2", "Sentence 3"],
        "hearing_date": ["2018-01-01", "2020-01-01", "2022-01-01"],
        "match_type": ["exact", "fuzzy", "exact"],
        "dem": [0, 1, 0],
        "minority": [0, 1, 0],
        "unified": [1, 0, 1],
        "freshman": [0, 0, 1],
        "member_state": ["CA", "NY", "TX"],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_convert_csv_to_parquet(mock_csv, tmp_path, monkeypatch):
    # Set the paths to use the temporary directory
    test_parquet = tmp_path / "test_data.parquet"

    # Monkeypatch the paths in optimize_data
    monkeypatch.setattr("scripts.optimize_data.CSV_PATH", str(mock_csv))
    monkeypatch.setattr("scripts.optimize_data.PARQUET_PATH", str(test_parquet))

    # Run the conversion
    success = convert_csv_to_parquet()

    assert success is True
    assert os.path.exists(test_parquet)

    # Verify the contents
    df_pq = pd.read_parquet(test_parquet)
    assert len(df_pq) == 3
    assert "text" in df_pq.columns  # Renamed from target_sentence
    assert df_pq["text"].tolist() == ["Sentence 1", "Sentence 2", "Sentence 3"]
    assert df_pq["congress"].dtype == "int16"
    assert df_pq["party"].dtype == "category"
    assert pd.api.types.is_datetime64_any_dtype(df_pq["hearing_date"])


def test_convert_csv_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.optimize_data.CSV_PATH", "non_existent.csv")
    success = convert_csv_to_parquet()
    assert success is False
