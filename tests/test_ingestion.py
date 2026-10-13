import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.ingestion.ingest import download_and_extract_data, log_ingestion_metadata
import os

@patch('src.ingestion.ingest.requests.get')
@patch('src.ingestion.ingest.zipfile.ZipFile')
def test_download_and_extract_data(mock_zip, mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response
    
    # Run
    url = "http://fakeurl.com/data.zip"
    extract_to = str(tmp_path)
    
    # We won't actually extract because we mocked ZipFile, but we can verify it returns the right path
    csv_path = download_and_extract_data(url, extract_to)
    
    assert csv_path == os.path.join(extract_to, "dataset_diabetes", "diabetic_data.csv")
    mock_get.assert_called_once_with(url, stream=True)

@patch('src.ingestion.ingest.load_dataframe_to_postgres')
def test_log_ingestion_metadata(mock_load):
    log_ingestion_metadata("test_source", "test_file", "SUCCESS", 100, 10)
    mock_load.assert_called_once()
    
    args, kwargs = mock_load.call_args
    df = args[0]
    assert df["source_name"].iloc[0] == "test_source"
    assert df["status"].iloc[0] == "SUCCESS"
    assert df["row_count"].iloc[0] == 100
