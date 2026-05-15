"""
Classification pipeline for new customers.

Wraps the trained Scaler -> PCA -> KMeans pipeline and provides clean
interfaces for both single-customer prediction (from a form) and batch
prediction (from a CSV/Excel upload).

Key functions:
- classify_single_customer: predict for one customer dict
- classify_batch: predict for a DataFrame, auto-detects raw vs preprocessed format
- build_features_from_form: translate UI form inputs to model features
"""

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_single_customer(customer: Dict, models: Dict) -> Dict:
    """
    Classify a single customer described by a dictionary of model features.

    Returns a dict with:
      - cluster_id        : int (0 or 1)
      - cluster_label     : str ("Prosperous" or "Families")
      - probabilities     : dict {label: probability}
    """
    feature_columns = models["feature_columns"]
    medians = models["median_values"]

    row_data = {col: customer.get(col, medians[col]) for col in feature_columns}
    row = pd.DataFrame([row_data], columns=feature_columns)

    return _predict(row, models)


def classify_batch(df: pd.DataFrame, models: Dict, add_pca_2d: bool = False) -> pd.DataFrame:
    """
    Classify a DataFrame of customers. Auto-detects whether the input is in
    raw Kaggle format (MntWines, Year_Birth, ...) or preprocessed format
    (Wines, Age_on_2014, ...) and applies feature engineering accordingly.

    Returns a copy of the original DataFrame with the engineered features
    merged in, plus the columns: Cluster, Cluster_Label, and optionally
    PCA_x / PCA_y when add_pca_2d=True.
    """
    feature_columns = models["feature_columns"]
    medians = models["median_values"]

    # Apply feature engineering if needed
    df_prepared = _preprocess_if_needed(df, models)

    # Fill missing columns with medians
    for col in feature_columns:
        if col not in df_prepared.columns:
            df_prepared[col] = medians[col]
        else:
            df_prepared[col] = df_prepared[col].fillna(medians[col])

    # Ensure correct column order
    features = df_prepared[feature_columns]

    # Apply the pipeline
    scaled = models["scaler"].transform(features)
    reduced = models["pca"].transform(scaled)
    clusters = models["kmeans_model"].predict(reduced)

    label_map = models["cluster_label_map"]

    # Start from preprocessed (so engineered features are available downstream)
    result = df_prepared.copy()
    result["Cluster"] = clusters
    result["Cluster_Label"] = [label_map[int(c)] for c in clusters]

    # Optional: 2D PCA coordinates for visualization
    if add_pca_2d and "pca_2d" in models:
        reduced_2d = models["pca_2d"].transform(scaled)
        result["PCA_x"] = reduced_2d[:, 0]
        result["PCA_y"] = reduced_2d[:, 1]

    # Re-attach ID column from the original input if present
    if "ID" in df.columns:
        result.insert(0, "ID", df["ID"].values)

    return result


def build_features_from_form(form_data: Dict, models: Dict) -> Dict:
    """
    Translate simplified UI form inputs into a complete feature dictionary
    for the classifier.

    Expected keys in form_data:
      - income, age, education, family_size
      - wines, meat, spent
      - recency, web_visits, n_accepted_campaigns
      - preferred_channel ("Web" | "Store" | "Catalog" | "No preference")
    """
    medians = models["median_values"]
    encoders = models["label_encoders"]

    # Start with medians for all features, then override with form data
    features = dict(medians)

    # --- Direct mappings ---
    features["Income"] = form_data["income"]
    features["Age_on_2014"] = form_data["age"]
    features["Family_size"] = form_data["family_size"]
    features["Wines"] = form_data["wines"]
    features["Meat"] = form_data["meat"]
    features["Spent"] = form_data["spent"]
    features["Recency"] = form_data["recency"]
    features["NumWebVisitsMonth"] = form_data["web_visits"]

    # --- Education: apply label encoding ---
    features["Education"] = encoders["Education"][form_data["education"]]

    # --- Campaigns: distribute "n accepted" across AcceptedCmp1..5 ---
    n = form_data["n_accepted_campaigns"]
    for i in range(1, 6):
        features[f"AcceptedCmp{i}"] = 1 if i <= n else 0

    # --- Spending: distribute remainder across small categories ---
    # Spent - Wines - Meat = remaining for Fish + Sweets + Fruits + Gold
    remaining = max(0, form_data["spent"] - form_data["wines"] - form_data["meat"])
    small_cats = ["Fish", "Sweets", "Fruits", "Gold"]
    median_sum = sum(medians[c] for c in small_cats)
    if median_sum > 0:
        for c in small_cats:
            features[c] = remaining * medians[c] / median_sum
    # If remaining is 0 (rare), categories stay at 0

    # --- Channel preference: skew purchase counts ---
    channel = form_data["preferred_channel"]
    total = (
        medians["NumWebPurchases"]
        + medians["NumStorePurchases"]
        + medians["NumCatalogPurchases"]
    )
    if channel == "Web":
        features["NumWebPurchases"] = total * 0.6
        features["NumStorePurchases"] = total * 0.2
        features["NumCatalogPurchases"] = total * 0.2
    elif channel == "Store":
        features["NumWebPurchases"] = total * 0.2
        features["NumStorePurchases"] = total * 0.6
        features["NumCatalogPurchases"] = total * 0.2
    elif channel == "Catalog":
        features["NumWebPurchases"] = total * 0.2
        features["NumStorePurchases"] = total * 0.2
        features["NumCatalogPurchases"] = total * 0.6
    # else "No preference" -> keep medians

    # --- Derived family fields ---
    family_size = form_data["family_size"]
    features["Is_parent"] = 1 if family_size >= 3 else 0
    features["Children"] = max(0, family_size - 2)
    # Living_with: Partner if family_size >= 2, else Alone
    living_label = "Partner" if family_size >= 2 else "Alone"
    if living_label in encoders["Living_with"]:
        features["Living_with"] = encoders["Living_with"][living_label]

    return features


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict(row: pd.DataFrame, models: Dict) -> Dict:
    """Run the Scaler -> PCA -> KMeans pipeline on a single-row DataFrame."""
    scaled = models["scaler"].transform(row)
    reduced = models["pca"].transform(scaled)
    cluster_id = int(models["kmeans_model"].predict(reduced)[0])

    # Pseudo-probabilities from inverse distances to centroids
    distances = models["kmeans_model"].transform(reduced)[0]
    inv = 1 / (distances + 1e-6)
    probs = inv / inv.sum()

    label_map = models["cluster_label_map"]
    return {
        "cluster_id": cluster_id,
        "cluster_label": label_map[cluster_id],
        "probabilities": {
            label_map[i]: float(probs[i]) for i in range(len(probs))
        },
    }


def _preprocess_if_needed(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """
    Detects whether the input DataFrame is in raw Kaggle format and applies
    the same feature engineering as build_models.py if so. Also applies
    label encoders to any remaining string columns.
    """
    df = df.copy()

    # Detect raw format by looking for original Kaggle columns
    is_raw = "MntWines" in df.columns or "Year_Birth" in df.columns

    if is_raw:
        # Income: fill missing with median
        if "Income" in df.columns:
            df["Income"] = df["Income"].fillna(df["Income"].median())

        # Date column not used by the model
        if "Dt_Customer" in df.columns:
            df = df.drop(columns=["Dt_Customer"])

        # Year_Birth -> Age_on_2014
        if "Year_Birth" in df.columns:
            df["Age_on_2014"] = 2014 - df["Year_Birth"]
            df = df.drop(columns=["Year_Birth"])

        # Mnt* -> short names
        rename_map = {
            "MntWines": "Wines", "MntFruits": "Fruits",
            "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Spent = sum of all 6 categories
        spent_cols = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold"]
        if all(c in df.columns for c in spent_cols):
            df["Spent"] = df[spent_cols].sum(axis=1)

        # Marital_Status -> Living_with
        if "Marital_Status" in df.columns:
            df["Living_with"] = df["Marital_Status"].replace({
                "Married": "Partner", "Together": "Partner",
                "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
                "Divorced": "Alone", "Single": "Alone",
            })
            df = df.drop(columns=["Marital_Status"])

        # Children, Family_size, Is_parent
        if "Kidhome" in df.columns and "Teenhome" in df.columns:
            df["Children"] = df["Kidhome"] + df["Teenhome"]
            df["Is_parent"] = (df["Children"] > 0).astype(int)
            if "Living_with" in df.columns:
                df["Family_size"] = (
                    df["Living_with"].replace({"Alone": 1, "Partner": 2})
                    + df["Children"]
                ).astype(int)

        # Education simplification
        if "Education" in df.columns:
            df["Education"] = df["Education"].replace({
                "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
                "Graduation": "Graduate",
                "Master": "Postgraduate", "PhD": "Postgraduate",
            })

        # Drop unused columns
        for col in ["Z_CostContact", "Z_Revenue", "ID"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Apply label encoders to any remaining string columns
    for col, encoder in models["label_encoders"].items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(encoder)

    return df
