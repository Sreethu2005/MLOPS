# Validation Rules

| Rule ID | Field/Table | Validation | Reason | Action on failure | Severity |
|---|---|---|---|---|---|
| DQ001 | raw_hospital_data | Row count > 0 | Ensure data was ingested | Fail pipeline | Critical |
| DQ002 | encounter_id | Must not be null | Required for encounter-level tracking | Reject record | Critical |
| DQ003 | encounter_id | Must be unique | Duplicate encounters corrupt counts | Reject record | Critical |
| DQ004 | age_group | Must not be null in staging | Demographics required for analytics | Reject record | High |
