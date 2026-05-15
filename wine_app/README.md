# Wine Reseller Analytics

Streamlit web app for the Capstone Project on customer segmentation for
Weingut Robert Weil (Frankfurt). Built on top of a k-means clustering
analysis (Prosperous / Families) and extended with a Random Forest
classifier that predicts customer responsiveness for premium-club scoring.

## Project structure

```
wine_app/
├── Dashboard.py                    # Main entry: Dashboard with file upload
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml.example        # Template for the Anthropic API key
├── data/
│   └── Clustered_data.xlsx         # Default dataset with IDs + PCA coords
├── models/                         # 10 pickle files (clustering + scoring)
├── pages/
│   ├── 1_Premium_Candidates.py     # ML-ranked candidate list
│   ├── 2_Quick_Lookup.py           # Single-customer classification
│   └── 3_AI_Assistant.py           # AI tools: emails, strategy, chat
└── utils/
    ├── data_loader.py              # Cached loaders for data and models
    ├── scoring.py                  # ML-based premium-club score
    ├── classifier.py               # Classification pipeline
    └── ai_client.py                # Anthropic API client + data summary
```

## Setup (once)

```
pip install -r requirements.txt
```

### Enable AI features (optional)

The AI Assistant page needs an Anthropic API key.

1. Get a key at https://console.anthropic.com (free $5 credit on signup).
2. In `.streamlit/`, copy `secrets.toml.example` to `secrets.toml`.
3. Open `secrets.toml` and replace `sk-ant-...` with your actual key.
4. Restart the app.

The other pages work fine without a key.

## Run the app

```
streamlit run Dashboard.py
```

Opens at http://localhost:8501. Stop with Ctrl+C.

## Modules

- **Dashboard** - KPIs, cluster profiles, PCA scatter, spending by category,
  interactive variable explorer. Upload your own CSV/Excel in the sidebar.
- **Premium Candidates** - ML-ranked candidate list with score distribution,
  recency-vs-spending scatter, feature importance chart, and CSV export.
- **Quick Lookup** - Single-customer form classification.
- **AI Assistant** - Email generator, strategy advisor, free-form chat
  (requires API key).

## Score model

The premium-club readiness score is the output of a Random Forest
classifier trained to predict customer response (the `Response` column)
based on demographics, spending, and channel behavior.

**Performance (test set):**
- ROC-AUC: 0.89
- Accuracy: 89%
- 5-fold CV ROC-AUC: 0.88 (+/- 0.02)

**Top features:** Recency, Spent, Income, Gold, Wines, Meat,
NumStorePurchases, AcceptedCmp3.

The score for each customer is `model.predict_proba(X)[:, 1] * 100`.
Source code for training: `train_response_model.py` (in the project
deliverables zip).

## Method notes

- Clustering uses **PCA with 19 components** (preserves 93% of variance)
  and **KMeans with k=2**.
- The Dashboard scatter uses a **second 2D PCA** for visualization only;
  cluster assignment always comes from the 19-component model.
- The AI Assistant sends only an **aggregated data summary** to the API,
  never raw customer rows.
