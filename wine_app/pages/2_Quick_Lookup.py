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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_models
from utils.classifier import build_features_from_form, classify_single_customer
from utils.scoring import compute_premium_score


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

        # Compute the premium score for this single customer
        features_df = pd.DataFrame([features])
        premium_score = float(compute_premium_score(features_df, models).iloc[0])

        st.divider()
        st.subheader("Classification result")

        # --- Top row: Cluster on the left, Score on the right ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Customer segment**")
            label = result["cluster_label"]
            cluster_color = COLOR_PROSPEROUS if label == "Prosperous" else COLOR_FAMILIES
            st.markdown(
                f"<div style='background-color:{cluster_color}; "
                f"padding: 20px; border-radius: 8px; text-align: center;'>"
                f"<div style='font-size: 14px; color: white; opacity: 0.9;'>"
                f"Assigned cluster</div>"
                f"<div style='font-size: 32px; color: white; font-weight: 600;'>"
                f"{label}</div></div>",
                unsafe_allow_html=True,
            )
            # Cluster confidence as small bar
            probs = result["probabilities"]
            st.caption(
                f"Confidence: {label} {probs[label]:.0%}"
            )

        with col_right:
            st.markdown("**Premium-club readiness**")
            # Color-code the score: red < 30, amber 30-60, green > 60
            if premium_score >= 60:
                score_color = "#1D9E75"   # green
                score_band = "High"
            elif premium_score >= 30:
                score_color = "#E0A800"   # amber
                score_band = "Medium"
            else:
                score_color = "#B84A4A"   # red
                score_band = "Low"

            st.markdown(
                f"<div style='background-color:{score_color}; "
                f"padding: 20px; border-radius: 8px; text-align: center;'>"
                f"<div style='font-size: 14px; color: white; opacity: 0.9;'>"
                f"Predicted response probability ({score_band})</div>"
                f"<div style='font-size: 32px; color: white; font-weight: 600;'>"
                f"{premium_score:.1f} / 100</div></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Score from the Random Forest model trained on historical "
                "campaign response data."
            )

        # --- Combined recommendation ---
        st.divider()
        st.markdown("##### Recommended actions")

        # Pick recommendation based on combination of cluster and score
        if label == "Prosperous" and premium_score >= 60:
            st.success(
                "**High-priority premium-club target.** This customer profile "
                "matches the Prosperous cluster *and* has a high predicted "
                "response probability."
            )
            st.markdown(
                "- Send a personalized Premium Club invitation **this week**\n"
                "- Offer early access to a limited-edition wine batch\n"
                "- Invite to the next VIP tasting or vineyard tour\n"
                "- Assign to the loyalty reward program tier"
            )
        elif label == "Prosperous" and premium_score >= 30:
            st.info(
                "**Warm premium-club candidate.** Prosperous profile, "
                "but moderate response probability - approach more carefully."
            )
            st.markdown(
                "- Include in the next premium newsletter, not a hard sell yet\n"
                "- Highlight bundled gourmet packages\n"
                "- Track engagement before sending a Premium Club invite"
            )
        elif label == "Prosperous":
            st.warning(
                "**Prosperous profile but low response signal.** Worth "
                "nurturing, but no aggressive outreach."
            )
            st.markdown(
                "- Standard premium newsletter only\n"
                "- Avoid pushy promotions - this customer is not engaging\n"
                "- Re-evaluate score in 6 months"
            )
        elif label == "Families" and premium_score >= 60:
            st.info(
                "**Highly engaged Families customer.** Not a Premium Club "
                "target, but a strong responder - great for family-oriented "
                "campaigns."
            )
            st.markdown(
                "- Feature in bundle-deal campaigns\n"
                "- Send the family newsletter with recipe ideas\n"
                "- Reward with loyalty points"
            )
        else:
            st.markdown(
                "**Families profile, low engagement.** Standard treatment."
            )
            st.markdown(
                "- Add to general family newsletter\n"
                "- Highlight discounts and bulk deals\n"
                "- No targeted Premium outreach"
            )
