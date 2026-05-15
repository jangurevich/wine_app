"""
Wine Reseller Analytics - Dashboard

Main entry point. Shows the current customer base (default: the dataset
shipped with the app, or any uploaded file) with KPIs, cluster profiles,
spending charts, a PCA scatter plot, and an interactive variable explorer.

Run with:  streamlit run Dashboard.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_customer_data, load_models
from utils.classifier import classify_batch


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Reseller Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_PROSPEROUS = "#1D9E75"
COLOR_FAMILIES = "#BA7517"
CLUSTER_COLORS = {"Prosperous": COLOR_PROSPEROUS, "Families": COLOR_FAMILIES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_active_data() -> tuple[pd.DataFrame, str]:
    """
    Return the active dataset.
    Uses uploaded data from session_state if present, else the default file.
    """
    if "uploaded_data" in st.session_state:
        return st.session_state["uploaded_data"], st.session_state["uploaded_name"]
    return load_customer_data(), "Default dataset"


def reset_to_default():
    """Clear any uploaded data and revert to the default dataset."""
    for key in ("uploaded_data", "uploaded_name"):
        if key in st.session_state:
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Load models (always cached)
# ---------------------------------------------------------------------------
models = load_models()


# ---------------------------------------------------------------------------
# Sidebar: upload + status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Wine Reseller Analytics")
    st.caption("Customer Segmentation App")
    st.divider()

    st.markdown("**Load customer data**")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help=(
            "Raw Kaggle format (with MntWines, Year_Birth, ...) and "
            "preprocessed format (with Wines, Age_on_2014, ...) are both "
            "supported. The file is auto-detected and classified."
        ),
    )

    if uploaded_file is not None:
        # Only re-process if the file changed (avoids re-running on every rerun)
        if st.session_state.get("uploaded_name") != uploaded_file.name:
            try:
                if uploaded_file.name.endswith(".csv"):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)
                with st.spinner("Classifying customers..."):
                    classified = classify_batch(raw_df, models, add_pca_2d=True)
                st.session_state["uploaded_data"] = classified
                st.session_state["uploaded_name"] = uploaded_file.name
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process file: {e}")

    if "uploaded_name" in st.session_state:
        st.success(f"Using uploaded file: **{st.session_state['uploaded_name']}**")
        if st.button("Reset to default dataset"):
            reset_to_default()
            st.rerun()

    st.divider()
    df, source_name = get_active_data()
    st.info(f"**{len(df):,}** customers loaded\n\nSource: {source_name}")

    st.divider()
    st.caption(
        "Based on the Capstone Project for Weingut Robert Weil - Frankfurt."
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Dashboard")
st.markdown(
    "Overview of the customer base, segmented into two clusters via "
    "k-means clustering. Upload a new file in the sidebar to analyze your "
    "own customer data."
)

prosperous = df[df["Cluster_Label"] == "Prosperous"]
families = df[df["Cluster_Label"] == "Families"]


# --- KPI cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total customers", f"{len(df):,}")
col2.metric(
    "Prosperous",
    f"{len(prosperous):,}",
    f"{len(prosperous) / len(df) * 100:.0f}% of total",
)
col3.metric(
    "Families",
    f"{len(families):,}",
    f"{len(families) / len(df) * 100:.0f}% of total",
)
col4.metric("Avg. total spending", f"EUR {df['Spent'].mean():,.0f}")


st.divider()

# --- Cluster profile cards ---
st.subheader("Cluster profiles")

col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("### Prosperous")
        st.caption("High income, premium-club candidates")
        st.markdown(
            f"- **Avg. income:** EUR {prosperous['Income'].mean():,.0f}\n"
            f"- **Avg. spending:** EUR {prosperous['Spent'].mean():,.0f}\n"
            f"- **Avg. family size:** {prosperous['Family_size'].mean():.2f}\n"
            f"- **Parents:** {prosperous['Is_parent'].mean() * 100:.0f}%\n"
            f"- **Web visits per month:** {prosperous['NumWebVisitsMonth'].mean():.1f}\n"
            f"- **Days since last purchase:** {prosperous['Recency'].mean():.1f}"
        )

with col_b:
    with st.container(border=True):
        st.markdown("### Families")
        st.caption("Mid income, family-oriented households")
        st.markdown(
            f"- **Avg. income:** EUR {families['Income'].mean():,.0f}\n"
            f"- **Avg. spending:** EUR {families['Spent'].mean():,.0f}\n"
            f"- **Avg. family size:** {families['Family_size'].mean():.2f}\n"
            f"- **Parents:** {families['Is_parent'].mean() * 100:.0f}%\n"
            f"- **Web visits per month:** {families['NumWebVisitsMonth'].mean():.1f}\n"
            f"- **Days since last purchase:** {families['Recency'].mean():.1f}"
        )


st.divider()

# --- PCA Scatter Plot ---
st.subheader("Cluster map (2D PCA projection)")
st.caption(
    "Each point is one customer, projected onto the two principal components "
    "that explain the most variance. Colors show the cluster assignment from "
    "the full 19-dimensional model."
)

if "PCA_x" in df.columns and "PCA_y" in df.columns:
    fig_pca = px.scatter(
        df,
        x="PCA_x",
        y="PCA_y",
        color="Cluster_Label",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.6,
        labels={
            "PCA_x": "Principal Component 1",
            "PCA_y": "Principal Component 2",
            "Cluster_Label": "Cluster",
        },
        hover_data={
            "Income": ":,.0f",
            "Spent": ":,.0f",
            "PCA_x": False,
            "PCA_y": False,
        },
    )
    fig_pca.update_traces(marker=dict(size=6))
    fig_pca.update_layout(
        height=480,
        margin=dict(t=20, b=80),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="center", x=0.5,
        ),
    )
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.info(
        "PCA coordinates not available for this dataset. "
        "Upload a fresh file to generate them."
    )


st.divider()

# --- Spending by product category ---
st.subheader("Average spending per product category")
st.caption("Annual spending in euros, grouped by cluster.")

categories = ["Wines", "Meat", "Fish", "Sweets", "Fruits", "Gold"]
available_cats = [c for c in categories if c in df.columns]

if available_cats:
    prosperous_means = [prosperous[c].mean() for c in available_cats]
    families_means = [families[c].mean() for c in available_cats]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Prosperous", x=available_cats, y=prosperous_means,
                marker_color=COLOR_PROSPEROUS,
                text=[f"{v:.0f}" for v in prosperous_means],
                textposition="outside",
            ),
            go.Bar(
                name="Families", x=available_cats, y=families_means,
                marker_color=COLOR_FAMILIES,
                text=[f"{v:.0f}" for v in families_means],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Avg. annual spending (EUR)",
        xaxis_title="Product category",
        height=420,
        margin=dict(t=30, b=80),
        legend=dict(
            orientation="h", yanchor="top", y=-0.2,
            xanchor="center", x=0.5,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


st.divider()

# --- Variable explorer ---
st.subheader("Variable explorer")
st.caption(
    "Pick a variable to see how the two clusters differ on it. "
    "Useful for the merchant to inspect any feature interactively."
)

# Numeric variables suitable for distribution comparison
numeric_candidates = [
    "Income", "Age_on_2014", "Spent", "Wines", "Meat", "Fish",
    "Sweets", "Fruits", "Gold", "Recency", "Family_size", "Children",
    "NumWebVisitsMonth", "NumWebPurchases", "NumStorePurchases",
    "NumCatalogPurchases", "NumDealsPurchases",
]
available_vars = [v for v in numeric_candidates if v in df.columns]

c1, c2 = st.columns([1, 3])
with c1:
    selected_var = st.selectbox("Select a variable", options=available_vars, index=0)
    chart_type = st.radio(
        "Chart type",
        options=["Histogram", "Box plot"],
        index=0,
    )

with c2:
    if chart_type == "Histogram":
        fig_var = px.histogram(
            df, x=selected_var, color="Cluster_Label",
            color_discrete_map=CLUSTER_COLORS,
            barmode="overlay", opacity=0.6, nbins=40,
            labels={"Cluster_Label": "Cluster"},
        )
    else:
        fig_var = px.box(
            df, x="Cluster_Label", y=selected_var,
            color="Cluster_Label",
            color_discrete_map=CLUSTER_COLORS,
            labels={"Cluster_Label": "Cluster"},
        )
    fig_var.update_layout(
        height=400, margin=dict(t=20, b=80),
        legend=dict(
            orientation="h", yanchor="top", y=-0.2,
            xanchor="center", x=0.5,
        ),
    )
    st.plotly_chart(fig_var, use_container_width=True)
