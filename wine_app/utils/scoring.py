"""
Premium-club candidate scoring (ML-based).

Replaces the previous hand-tuned composite formula with a trained
Random Forest classifier that predicts customer responsiveness to
marketing campaigns. The score is the predicted probability of
"response = yes" scaled to 0-100.

Target proxy: the `Response` column from the original dataset captures
whether the customer accepted the last marketing campaign. Customers
with a high predicted response probability are interpreted as good
premium-club candidates.

Notes
-----
* The model expects the same engineered features used during training
  (see train_response_model.py). The feature list is stored alongside
  the model in response_model_meta.pkl.
* For dataframes coming from the app (already classified, with feature
  engineering applied) the score can be computed directly.
"""

from typing import Dict

import pandas as pd


def compute_premium_score(df: pd.DataFrame, models: Dict) -> pd.Series:
    """
    Compute the premium-club candidate score for every row in the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain (at least) the engineered features the model was
        trained on. Missing features are filled with the training-set
        median to keep the function robust to partial datasets.
    models : dict
        The loaded model artifacts dict (must contain 'response_model',
        'response_model_meta', and 'median_values').

    Returns
    -------
    Series of floats in [0, 100], rounded to one decimal place.
    """
    model = models["response_model"]
    meta = models["response_model_meta"]
    medians = models["median_values"]

    feature_names = meta["feature_names"]

    # Build a feature matrix in the correct order, filling missing columns
    # with median values from the training data.
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col].fillna(medians.get(col, 0))
        else:
            X[col] = medians.get(col, 0)

    # Predict probability of class 1 (= responsive) and scale to 0-100
    probs = model.predict_proba(X)[:, 1]
    return pd.Series((probs * 100).round(1), index=df.index)
