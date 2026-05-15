"""
Trains a Random Forest classifier to predict customer response
(AcceptedCmp -> Premium-Club readiness proxy).

Outputs:
  - response_model.pkl       trained classifier
  - response_model_meta.pkl  metadata: feature importance, metrics, etc.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

SRC = "/mnt/user-data/uploads/customer_segmentation.csv"
OUT = Path("/home/claude/models")
OUT.mkdir(exist_ok=True)

# =========================================================================
# 1) Load + same feature engineering as build_models.py
# =========================================================================
df = pd.read_csv(SRC)
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d.%m.%y")

df["Age_on_2014"] = 2014 - df["Year_Birth"]
df["Spent"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)
df["Living_with"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
    "Divorced": "Alone", "Single": "Alone",
})
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Family_size"] = (
    df["Living_with"].replace({"Alone": 1, "Partner": 2}) + df["Children"]
).astype(int)
df["Is_parent"] = np.where(df["Children"] > 0, 1, 0)
df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate",
})
df = df.rename(columns={
    "MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
    "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
})

# Outliers (same filter as in build_models.py)
df = df[(df["Age_on_2014"] < 90) & (df["Income"] < 600000)].reset_index(drop=True)

# Label encode categoricals
encoders = {}
for col in ["Education", "Living_with"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = {label: int(code) for code, label in enumerate(le.classes_)}

# =========================================================================
# 2) Define target = Response, drop leakage columns
# =========================================================================
y = df["Response"].copy()
print(f"Target distribution: {y.value_counts().to_dict()} "
      f"({y.mean():.1%} positives)")

# Drop the target itself + the date + ID + admin columns.
# We KEEP AcceptedCmp1..5 because they are legitimate predictors:
# they describe past behavior available at scoring time, not the target.
X = df.drop(columns=[
    "Response", "Dt_Customer", "Marital_Status", "Year_Birth",
    "ID", "Z_CostContact", "Z_Revenue",
])

feature_names = X.columns.tolist()
print(f"Features used: {len(feature_names)}")

# =========================================================================
# 3) Train/test split + Random Forest
# =========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",   # handles the imbalance
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# =========================================================================
# 4) Evaluate
# =========================================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
    "roc_auc":   roc_auc_score(y_test, y_prob),
}
print("\nTest metrics:")
for k, v in metrics.items():
    print(f"  {k:9s} {v:.3f}")

print("\nConfusion matrix [TN FP / FN TP]:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 5-fold CV ROC-AUC for robustness
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"\n5-fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# =========================================================================
# 5) Feature importances
# =========================================================================
importances = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 10 features:")
print(importances.head(10).to_string(index=False))

# =========================================================================
# 6) Save artifacts
# =========================================================================
with open(OUT / "response_model.pkl", "wb") as f:
    pickle.dump(model, f)

meta = {
    "feature_names": feature_names,
    "metrics": metrics,
    "cv_roc_auc_mean": float(cv_scores.mean()),
    "cv_roc_auc_std": float(cv_scores.std()),
    "confusion_matrix": cm.tolist(),
    "feature_importance": importances.to_dict("records"),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "target_positive_rate": float(y.mean()),
}
with open(OUT / "response_model_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved response_model.pkl and response_model_meta.pkl")
