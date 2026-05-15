"""
Data and model loading utilities.

Streamlit reruns the script on every interaction, so caching is essential
to avoid re-loading the Excel file and pickle models on every click.

- @st.cache_data: for DataFrames (returns a copy on each call)
- @st.cache_resource: for ML models (returns the same object, shared)
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Project root (one level up from utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Names of all model artifacts produced by build_models.py and train_response_model.py
MODEL_ARTIFACTS = [
    "scaler",
    "pca",
    "pca_2d",
    "kmeans_model",
    "cluster_label_map",
    "feature_columns",
    "median_values",
    "label_encoders",
    "response_model",
    "response_model_meta",
]


@st.cache_data
def load_customer_data() -> pd.DataFrame:
    """Load the clustered customer dataset from Excel."""
    path = DATA_DIR / "Clustered_data.xlsx"
    if not path.exists():
        st.error(f"Data file not found: {path}")
        st.stop()
    return pd.read_excel(path)


@st.cache_resource
def load_models() -> dict:
    """Load all model artifacts (scaler, PCA, KMeans, etc.) from disk."""
    artifacts = {}
    for name in MODEL_ARTIFACTS:
        path = MODELS_DIR / f"{name}.pkl"
        if not path.exists():
            st.error(f"Model artifact not found: {path}")
            st.stop()
        with open(path, "rb") as f:
            artifacts[name] = pickle.load(f)
    return artifacts
