"""
Quick Lookup page.

Form-based classification of a single customer. Useful for ad-hoc lookups
(e.g. a walk-in customer, a phone inquiry) without modifying the active
dataset. For bulk classification, use the upload feature on the Dashboard.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_models
from utils.classifier import build_features_from_form, classify_single_customer


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Quick Lookup", layout="wide")

st.title("Quick Lookup")
st.markdown(
    "Quickly classify a single customer to get a segment assignment and "
    "tailored recommendations. This does not modify the active dataset. "
    "For bulk classification, use the upload feature on the Dashboard."
)

models = load_models()

COLOR_PROSPEROUS = "#1D9E75"
COLOR_FAMILIES = "#BA7517"

RECOMMENDATIONS = {
    "Prosperous": (
        "**Recommended actions:**\n"
        "- Send a personalized Premium Club invitation with early access to limited-edition wines\n"
        "- Invite to exclusive in-store events: wine tastings, food pairings, vineyard tours\n"
        "- Promote bundled gourmet packages and higher-margin add-ons\n"
        "- Add to the loyalty reward program with redeemable exclusive experiences"
    ),
    "Families": (
        "**Recommended actions:**\n"
        "- Add to the family-oriented newsletter with budget-friendly shopping tips\n"
        "- Highlight bundle deals and larger discounts on bulk purchases\n"
        "- Award loyalty points for frequent shopping\n"
        "- Share recipe ideas and pairings suitable for family meals"
    ),
}


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
with st.form(key="single_customer_form"):
    st.markdown(
        "Enter the customer's profile. Fields not asked here are filled "
        "with the median value from the training data."
    )

    # --- Demographics ---
    st.markdown("##### Demographics")
    c1, c2, c3, c4 = st.columns(4)
    income = c1.number_input(
        "Annual income (EUR)",
        min_value=0, max_value=200000, value=50000, step=1000,
    )
    age = c2.number_input(
        "Age", min_value=18, max_value=100, value=45, step=1,
    )
    education = c3.selectbox(
        "Education",
        options=["Undergraduate", "Graduate", "Postgraduate"],
        index=1,
    )
    family_size = c4.number_input(
        "Family size", min_value=1, max_value=5, value=2, step=1,
        help="Total household size (1 = single, 2 = couple, 3+ with children)",
    )

    # --- Spending ---
    st.markdown("##### Annual spending")
    c1, c2, c3 = st.columns(3)
    wines = c1.number_input(
        "Spending on wines (EUR)",
        min_value=0, max_value=2000, value=300, step=10,
    )
    meat = c2.number_input(
        "Spending on meat (EUR)",
        min_value=0, max_value=1000, value=150, step=10,
    )
    spent = c3.number_input(
        "Total spending - all categories (EUR)",
        min_value=0, max_value=3000, value=550, step=10,
        help="Sum across wines, meat, fish, sweets, fruits, and gold products",
    )

    # --- Behavior ---
    st.markdown("##### Customer behavior")
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input(
        "Days since last purchase",
        min_value=0, max_value=200, value=30, step=1,
    )
    web_visits = c2.number_input(
        "Web visits per month",
        min_value=0, max_value=30, value=5, step=1,
    )
    n_campaigns = c3.slider(
        "Accepted marketing campaigns (out of 5)",
        min_value=0, max_value=5, value=0, step=1,
    )

    # --- Channel ---
    st.markdown("##### Channel preference")
    preferred_channel = st.radio(
        "Preferred purchase channel",
        options=["No preference", "Store", "Web", "Catalog"],
        index=0,
        horizontal=True,
    )

    submitted = st.form_submit_button("Classify customer", type="primary")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
if submitted:
    if wines + meat > spent:
        st.error(
            "Spending on wines and meat exceeds total spending. "
            "Please correct the values."
        )
    else:
        form_data = {
            "income": income, "age": age, "education": education,
            "family_size": family_size,
            "wines": wines, "meat": meat, "spent": spent,
            "recency": recency, "web_visits": web_visits,
            "n_accepted_campaigns": n_campaigns,
            "preferred_channel": preferred_channel,
        }
        features = build_features_from_form(form_data, models)
        result = classify_single_customer(features, models)

        st.divider()
        st.subheader("Classification result")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            label = result["cluster_label"]
            color = COLOR_PROSPEROUS if label == "Prosperous" else COLOR_FAMILIES
            st.markdown(
                f"<div style='background-color:{color}; "
                f"padding: 20px; border-radius: 8px; text-align: center;'>"
                f"<div style='font-size: 14px; color: white; opacity: 0.9;'>"
                f"Assigned cluster</div>"
                f"<div style='font-size: 32px; color: white; font-weight: 600;'>"
                f"{label}</div></div>",
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown("**Confidence**")
            probs = result["probabilities"]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[probs["Prosperous"] * 100, probs["Families"] * 100],
                        y=["Prosperous", "Families"],
                        orientation="h",
                        marker_color=[COLOR_PROSPEROUS, COLOR_FAMILIES],
                        text=[
                            f"{probs['Prosperous']:.1%}",
                            f"{probs['Families']:.1%}",
                        ],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                height=160, margin=dict(t=10, b=20, l=10, r=10),
                xaxis=dict(range=[0, 110], showticklabels=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(RECOMMENDATIONS[label])
