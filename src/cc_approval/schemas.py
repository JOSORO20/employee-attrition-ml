from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class Application(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    years_employed: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    existing_debt: float = Field(..., ge=0)
    requested_limit: float = Field(..., ge=0)
    debt_to_income: float = Field(..., ge=0)
    years_at_address: float = Field(..., ge=0)
    dependents: int = Field(..., ge=0, le=10)

    gender: str
    education_level: str
    marital_status: str
    housing_status: str
    has_default_history: str  # "yes"/"no"

class Prediction(BaseModel):
    approval_probability: float
    approve: bool
