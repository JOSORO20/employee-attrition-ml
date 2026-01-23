from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Feature configuration
NUMERIC_FEATURES: List[str] = [
    "age", "annual_income", "years_employed", "credit_score",
    "existing_debt", "requested_limit", "debt_to_income", "years_at_address", "dependents"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender", "education_level", "marital_status", "housing_status", "has_default_history"
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def build_model_pipeline() -> Pipeline:
    pre = build_preprocessor()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return pipe
