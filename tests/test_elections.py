import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.elections import load_elections_data, fetch_mit_election_data

def test_load_elections_data(tmp_path):
    """Test processing of MIT Election Data."""
    # Create a mock CSV with various scenarios
    # - Standard contested race (NY-14, 2018)
    # - Unopposed race (candidatevotes == totalvotes)
    # - Unopposed race with totalvotes == 0 or missing
    # - At-large district (district == 0)
    
    mock_data = pd.DataFrame({
        "year": [2016, 2018, 2018, 2020, 2022],
        "state_po": ["WY", "NY", "NY", "CA", "TX"],
        "district": ["0", "14", "14", "12", "5"],
        "candidatevotes": [100000, 100000, 20000, 150000, 100000],
        "totalvotes": [100000, 120000, 120000, 150000, 0]  # TX-5 has 0 total votes (unopposed)
    })
    
    mock_csv = tmp_path / "mock_elections.csv"
    mock_data.to_csv(mock_csv, index=False)
    
    # Run the function
    result = load_elections_data(path=str(mock_csv), target_congresses=[115, 116, 117, 118])
    
    # Assertions
    assert len(result) == 4  # WY 2016, NY 2018, CA 2020, TX 2022
    
    # 2016 -> 115th Congress
    wy_115 = result[(result["state_abbrev"] == "WY") & (result["congress"] == 115)].iloc[0]
    assert wy_115["district_code"] == 0
    assert wy_115["vote_pct"] == 100.0
    
    # 2018 -> 116th Congress
    ny_116 = result[(result["state_abbrev"] == "NY") & (result["congress"] == 116)].iloc[0]
    assert ny_116["district_code"] == 14
    # Max vote share in NY-14 should be the winner's (100000 / 120000 * 100 = 83.33)
    assert abs(ny_116["vote_pct"] - 83.333) < 0.01
    
    # 2022 -> 118th Congress (totalvotes = 0)
    tx_118 = result[(result["state_abbrev"] == "TX") & (result["congress"] == 118)].iloc[0]
    assert tx_118["vote_pct"] == 100.0  # Should be filled to 100.0

@patch("src.elections.requests.get")
@patch("src.elections.os.makedirs")
def test_fetch_mit_election_data_success(mock_makedirs, mock_get, tmp_path):
    """Test the fetcher function when the API returns 200."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"mock,data\n1,2\n"]
    mock_get.return_value = mock_response
    
    dest_path = tmp_path / "test_download.csv"
    
    with patch("src.elections.os.getenv", return_value="dummy_token"):
        result_path = fetch_mit_election_data(dest_path=str(dest_path))
        
    assert result_path == str(dest_path)
    assert os.path.exists(result_path)
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "X-Dataverse-key" in kwargs["headers"]
    assert kwargs["headers"]["X-Dataverse-key"] == "dummy_token"

@patch("src.elections.requests.get")
def test_fetch_mit_election_data_failure(mock_get, tmp_path):
    """Test the fetcher function handles errors properly."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Guestbook required"
    mock_get.return_value = mock_response
    
    dest_path = tmp_path / "test_download_fail.csv"
    
    with pytest.raises(RuntimeError, match="Guestbook required"):
        fetch_mit_election_data(dest_path=str(dest_path))
