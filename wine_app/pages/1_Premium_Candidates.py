"""
Premium Candidates page.

Ranks customers by an ML-based "premium-club readiness" score, lets the
merchant filter, inspect, and export the top candidates.

Score = Random Forest predicted probability of campaign response,
scaled to 0-100. See utils/scoring.py and train_response_model.py.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_customer_data, load_models
from utils.scoring import compute_premium_score


COLOR_PROSPEROUS = "#1D9E75"
COLOR_FAMILIES = "#BA7517"
CLUSTER_COLORS = {"Prosperous": COLOR_PROSPEROUS, "Families": COLOR_FAMILIES}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Premium Candidates", layout="wide")

st.title("Premium Candidates")
st.markdown(
    "Top customer candidates for the premium-club outreach, ranked by a "
    "**predicted response probability** from a Random Forest classifier. "
    "Filter the list and export it for your CRM or mailing tool."
)


# ---------------------------------------------------------------------------
# Load data and models
# ---------------------------------------------------------------------------
models = load_models()

if "uploaded_data" in st.session_state:
    df = st.session_state["uploaded_data"].copy()
    source = st.session_state["uploaded_name"]
else:
    df = load_customer_data().copy()
    source = "Default dataset"

st.caption(f"Source: **{source}** ({len(df):,} customers)")


# Compute the ML-based score
df["Score"] = compute_premium_score(df, models)


# ---------------------------------------------------------------------------
# Model info expander
# ---------------------------------------------------------------------------
meta = models["response_model_meta"]
metrics = meta["metrics"]

with st.expander("How is the score computed? (click to learn more)"):
    st.markdown(
        "The score is the **predicted probability** (0-100) that a customer "
        "will respond positively to a marketing campaign, output by a "
        "Random Forest classifier trained on historical campaign-response "
        "data.\n\n"
        "**Why this is meaningful:** customers with a high predicted "
        "response probability are receptive to outreach and are the most "
        "promising targets for a premium-club invitation."
    )

    st.markdown("##### Model performance (test set)")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    pc2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    pc3.metric("Precision", f"{metrics['precision']:.1%}")
    pc4.metric("Recall", f"{metrics['recall']:.1%}")

    st.caption(
        f"Trained on {meta['n_train']:,} customers, evaluated on "
        f"{meta['n_test']:,}. 5-fold CV ROC-AUC: "
        f"{meta['cv_roc_auc_mean']:.3f} (+/- {meta['cv_roc_auc_std']:.3f})."
    )

    # Feature importance bar chart
    st.markdown("##### Top features driving the score")
    importance_df = pd.DataFrame(meta["feature_importance"]).head(10)
    importance_df = importance_df.sort_values("importance")  # ascending for horizontal bar
    fig_imp = px.bar(
        importance_df, x="importance", y="feature", orientation="h",
        labels={"importance": "Feature importance", "feature": ""},
    )
    fig_imp.update_traces(marker_color=COLOR_PROSPEROUS)
    fig_imp.update_layout(
        height=320, margin=dict(t=10, b=20, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)

with f1:
    cluster_choice = st.selectbox(
        "Cluster",
        options=["Prosperous", "Families", "All"],
        index=0,
        help="Default: only Prosperous - the target group for the premium club.",
    )

with f2:
    min_score = st.slider("Minimum score", 0, 100, 50, 5)

with f3:
    max_recency = st.slider(
        "Max. days since last purchase",
        0, int(df["Recency"].max()), 60, 5,
    )

with f4:
    min_spent = st.number_input(
        "Min. total spending (EUR)",
        min_value=0, max_value=int(df["Spent"].max()),
        value=0, step=100,
    )

# Apply filters
filtered = df.copy()
if cluster_choice != "All":
    filtered = filtered[filtered["Cluster_Label"] == cluster_choice]
filtered = filtered[
    (filtered["Score"] >= min_score)
    & (filtered["Recency"] <= max_recency)
    & (filtered["Spent"] >= min_spent)
]
filtered = filtered.sort_values("Score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Matching candidates", f"{len(filtered):,}")
if len(filtered) > 0:
    m2.metric("Average score", f"{filtered['Score'].mean():.1f}")
    m3.metric("Total spending (EUR)", f"{filtered['Spent'].sum():,.0f}")
else:
    m2.metric("Average score", "-")
    m3.metric("Total spending (EUR)", "-")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Score and behavior")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("**Score distribution**")
    st.caption("The vertical line shows the current minimum-score filter.")
    fig_hist = px.histogram(
        df, x="Score", color="Cluster_Label",
        color_discrete_map=CLUSTER_COLORS,
        nbins=30, opacity=0.7,
        labels={"Score": "Premium-club score", "Cluster_Label": "Cluster"},
    )
    fig_hist.add_vline(
        x=min_score, line_dash="dash", line_color="#999",
        annotation_text=f"Cutoff: {min_score}", annotation_position="top",
    )
    fig_hist.update_layout(
        height=400, margin=dict(t=20, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.2,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with viz_col2:
    st.markdown("**Recency vs. spending**")
    st.caption("Each point is a customer. Larger markers indicate a higher score.")
    plot_df = df.copy()
    plot_df["MarkerSize"] = plot_df["Score"].clip(lower=2)
    fig_scatter = px.scatter(
        plot_df,
        x="Recency", y="Spent",
        color="Cluster_Label",
        color_discrete_map=CLUSTER_COLORS,
        size="MarkerSize", size_max=18,
        opacity=0.55,
        labels={
            "Recency": "Days since last purchase",
            "Spent": "Total annual spending (EUR)",
            "Cluster_Label": "Cluster",
        },
        hover_data={
            "Score": ":.1f", "Income": ":,.0f", "MarkerSize": False,
        },
    )
    fig_scatter.update_layout(
        height=400, margin=dict(t=20, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.2,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Candidate list")

if len(filtered) == 0:
    st.info("No customers match the current filters. Try relaxing them.")
else:
    display_columns = [
        "ID", "Score", "Cluster_Label", "Income", "Spent", "Wines",
        "Recency", "Family_size", "Is_parent",
        "NumStorePurchases", "NumCatalogPurchases", "NumWebPurchases",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5",
    ]
    display_columns = [c for c in display_columns if c in filtered.columns]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        height=500,
        column_config={
            "ID": st.column_config.NumberColumn("Customer ID", format="%d"),
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Predicted response probability (0-100), from Random Forest",
                min_value=0, max_value=100, format="%.1f",
            ),
            "Income": st.column_config.NumberColumn("Income", format="EUR %d"),
            "Spent": st.column_config.NumberColumn("Spent", format="EUR %d"),
            "Wines": st.column_config.NumberColumn("Wines", format="EUR %d"),
            "Recency": st.column_config.NumberColumn("Recency", format="%d days"),
            "Is_parent": st.column_config.NumberColumn("Parent", format="%d"),
        },
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download candidate list as CSV",
        data=csv,
        file_name="premium_candidates.csv",
        mime="text/csv",
    )
