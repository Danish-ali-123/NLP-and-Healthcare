# src/data/anonymize.py
from __future__ import annotations
from typing import Dict

# List of columns that contain pseudonymized data
PSEUDO_COLS = {
    "Pseudonymized_patient_id",
    "Pseudonymized_age",
    "Pseudonymized_gender",
    "Pseudonymized_Diagnosis",
    "Pseudonymized_Remarks",
    "Pseudonymized_doctor_id",
    "Pseudonymized_Patient History",
    "Pseudonymized_age_group",
    "Pseudonymized_gender_numeric",
    "Pseudonymized_symptoms",
    "Pseudonymized_treatment",
    "Pseudonymized_timespan",
    "Pseudonymized_Diagnosis Category",
}

def pick_pseudonymized(row: Dict[str, str]) -> Dict[str, str]:
    """Return only pseudonymized fields from the row."""
    return {k: row.get(k) for k in PSEUDO_COLS if k in row}
