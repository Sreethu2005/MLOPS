import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.validation.validate import validate_raw_data

@patch('src.validation.validate.get_db_connection')
@patch('src.validation.validate.pd.read_sql')
@patch('src.validation.validate.log_validation_results')
@patch('src.validation.validate.log_rejected_records')
def test_validate_raw_data_success(mock_log_rejected, mock_log_results, mock_read_sql, mock_get_db):
    # Mock dataframe with no missing encounter_ids and no duplicates
    mock_df = pd.DataFrame({
        "encounter_id": ["1", "2", "3"],
        "patient_nbr": ["p1", "p2", "p3"]
    })
    mock_read_sql.return_value = mock_df
    
    result = validate_raw_data()
    assert result is True
    
    # Ensure log_rejected_records was NOT called (no failures)
    mock_log_rejected.assert_not_called()

@patch('src.validation.validate.get_db_connection')
@patch('src.validation.validate.pd.read_sql')
@patch('src.validation.validate.log_validation_results')
@patch('src.validation.validate.log_rejected_records')
def test_validate_raw_data_failures(mock_log_rejected, mock_log_results, mock_read_sql, mock_get_db):
    # Mock dataframe with a missing encounter_id and a duplicate
    mock_df = pd.DataFrame({
        "encounter_id": ["1", None, "2", "2"],
        "patient_nbr": ["p1", "p2", "p3", "p4"]
    })
    mock_read_sql.return_value = mock_df
    
    result = validate_raw_data()
    assert result is True
    
    # log_rejected_records should be called twice (once for missing, once for duplicate)
    assert mock_log_rejected.call_count == 2
