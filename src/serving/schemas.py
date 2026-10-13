"""
Pydantic schemas for FastAPI serving layer with validation and clear error feedback.
Academic Decision-Support Prototype Only.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from src.data.loader import categorize_icd9

ACADEMIC_DISCLAIMER = (
    "ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY: This risk score is generated for academic and research purposes. "
    "It is NOT a validated diagnostic or clinical triaging tool and must NOT be used for direct patient care."
)

class PatientEncounterInput(BaseModel):
    """
    Schema representing an acute hospital encounter for diabetes-related readmission screening.
    """
    race: Optional[str] = Field(default="Caucasian", description="Reported racial background (e.g. Caucasian, AfricanAmerican, Hispanic, Asian, Other)")
    gender: str = Field(..., description="Reported gender (Male or Female)")
    age: str = Field(..., description="Age bracket, e.g. [0-10), [10-20), ..., [80-90), [90-100)")
    admission_type_id: int = Field(default=1, ge=1, le=8, description="Admission type code (1=Emergency, 2=Urgent, 3=Elective, etc.)")
    admission_source_id: int = Field(default=7, ge=1, le=26, description="Admission source code (7=Emergency Room, 1=Physician Referral, etc.)")
    time_in_hospital: int = Field(..., ge=1, le=14, description="Length of hospital stay in days (1-14)")
    num_lab_procedures: int = Field(..., ge=1, le=150, description="Total number of lab tests performed")
    num_procedures: int = Field(default=0, ge=0, le=20, description="Total number of non-lab clinical procedures")
    num_medications: int = Field(..., ge=1, le=100, description="Total number of distinct medications administered")
    number_outpatient: int = Field(default=0, ge=0, description="Number of outpatient visits in the preceding year")
    number_emergency: int = Field(default=0, ge=0, description="Number of emergency visits in the preceding year")
    number_inpatient: int = Field(default=0, ge=0, description="Number of inpatient admissions in the preceding year")
    number_diagnoses: int = Field(..., ge=1, le=16, description="Number of diagnoses entered on encounter claim")
    
    # Lab & Medication features
    max_glu_serum: Optional[str] = Field(default="None", description="Glucose serum test result (None, Norm, >200, >300)")
    A1Cresult: Optional[str] = Field(default="None", description="HbA1c test result (None, Norm, >7, >8)")
    metformin: Optional[str] = Field(default="No", description="Metformin dosage change (No, Steady, Up, Down)")
    repaglinide: Optional[str] = Field(default="No", description="Repaglinide dosage change")
    glimepiride: Optional[str] = Field(default="No", description="Glimepiride dosage change")
    glipizide: Optional[str] = Field(default="No", description="Glipizide dosage change")
    glyburide: Optional[str] = Field(default="No", description="Glyburide dosage change")
    pioglitazone: Optional[str] = Field(default="No", description="Pioglitazone dosage change")
    rosiglitazone: Optional[str] = Field(default="No", description="Rosiglitazone dosage change")
    insulin: Optional[str] = Field(default="No", description="Insulin dosage change (No, Steady, Up, Down)")
    change: Optional[str] = Field(default="No", description="Any diabetic medication change (Ch or No)")
    diabetesMed: Optional[str] = Field(default="Yes", description="Any diabetic medication prescribed (Yes or No)")

    # Diagnostic inputs (raw ICD-9 code or category)
    diag_1: Optional[str] = Field(default="250.01", description="Primary diagnosis (ICD-9 code)")
    diag_2: Optional[str] = Field(default=None, description="Secondary diagnosis (ICD-9 code)")
    diag_3: Optional[str] = Field(default=None, description="Additional secondary diagnosis (ICD-9 code)")

    diag_1_category: Optional[str] = Field(default=None, description="ICD-9 category (auto-derived if omitted)")
    diag_2_category: Optional[str] = Field(default=None, description="ICD-9 category (auto-derived if omitted)")
    diag_3_category: Optional[str] = Field(default=None, description="ICD-9 category (auto-derived if omitted)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    def resolve_categories(self) -> Dict[str, Any]:
        """Resolves diagnosis categories from raw ICD-9 codes if omitted."""
        data = self.model_dump()
        for idx in [1, 2, 3]:
            cat_key = f"diag_{idx}_category"
            diag_key = f"diag_{idx}"
            if not data.get(cat_key):
                raw_code = data.get(diag_key)
                data[cat_key] = categorize_icd9(raw_code)
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[60-70)",
                "admission_type_id": 1,
                "admission_source_id": 7,
                "time_in_hospital": 3,
                "num_lab_procedures": 48,
                "num_procedures": 1,
                "num_medications": 14,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 1,
                "number_diagnoses": 7,
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "metformin": "Steady",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes",
                "diag_1": "410.1",
                "diag_2": "250.02",
                "diag_3": "428"
            }
        }
    }

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    direction: str
    feature_value_scaled: float

class PredictionResponse(BaseModel):
    readmission_probability: float
    predicted_class: int
    predicted_label: str
    decision_threshold: float
    model_version: str
    model_alias: str
    top_contributing_features: List[FeatureExplanation]
    latency_ms: float
    request_id: str
    disclaimer: str = ACADEMIC_DISCLAIMER

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    model_alias: str
    uptime_seconds: float
    disclaimer: str = ACADEMIC_DISCLAIMER

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    decision_threshold: float
    tags: Dict[str, str]
    metrics: Dict[str, Any]
    disclaimer: str = ACADEMIC_DISCLAIMER

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    p95_latency_ms: float
    high_risk_predictions: int
    low_risk_predictions: int
    avg_predicted_risk: float
