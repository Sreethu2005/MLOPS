import pytest
from unittest.mock import patch
import pandas as pd
from src.transformation.transform import clean_and_transform_curated, transform_to_staging

@patch('src.transformation.transform.get_db_connection')
@patch('src.transformation.transform.pd.read_sql')
@patch('src.transformation.transform.load_dataframe_to_postgres')
def test_transform_to_staging(mock_load, mock_read_sql, mock_get_db):
    mock_df = pd.DataFrame({
        "encounter_id": ["1", "2", "3", "3"], # 3 is duplicate
        "patient_nbr": ["11", "22", "33", "33"],
        "race": ["Caucasian", "AfricanAmerican", "?", "?"],
        "gender": ["Male", "Female", "Unknown/Invalid", "Unknown/Invalid"],
        "age": ["[0-10)", "?", "[10-20)", "[10-20)"],
        "admission_type_id": ["1", "2", "3", "3"],
        "discharge_disposition_id": ["1", "1", "1", "1"],
        "admission_source_id": ["7", "7", "7", "7"],
        "time_in_hospital": ["1", "2", "?", "?"],
        "num_lab_procedures": ["41", "59", "11", "11"],
        "num_procedures": ["0", "0", "5", "5"],
        "num_medications": ["1", "18", "13", "13"],
        "number_outpatient": ["0", "0", "2", "2"],
        "number_emergency": ["0", "0", "0", "0"],
        "number_inpatient": ["0", "0", "1", "1"],
        "diag_1": ["250.83", "276", "682", "682"],
        "diag_2": ["?", "250.01", "428", "428"],
        "diag_3": ["?", "255", "36", "36"],
        "number_diagnoses": ["1", "9", "6", "6"],
        "readmitted": ["NO", "<30", ">30", ">30"]
    })
    mock_read_sql.return_value = mock_df
    
    transform_to_staging()
    
    # Check that load was called
    mock_load.assert_called_once()
    args, kwargs = mock_load.call_args
    result_df = args[0]
    
    # 3 unique encounter_ids
    assert len(result_df) == 3
    
    # Check column renames
    assert "age_group" in result_df.columns
    assert "length_of_stay" in result_df.columns
    
    # Check '?' converted to None/NaN
    assert pd.isna(result_df[result_df["encounter_id"] == 2]["age_group"].iloc[0])
