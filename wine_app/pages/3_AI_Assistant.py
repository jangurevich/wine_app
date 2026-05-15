"""
AI Assistant page.

Three tabs:
  1. Email Generator    - drafts marketing emails for a chosen cluster + occasion
  2. Strategy Advisor   - generates 3-5 prioritized action recommendations
  3. Free-Form Chat     - lets the merchant ask any data-related question

All three share the same data summary as context.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.data_loader import load_customer_data
from utils.ai_client import (
    check_api_key,
    call_claude,
    summarize_data_context,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("AI Assistant")
st.markdown(
    "AI-powered tools to help with marketing communications and strategic "
    "decisions. All features run on your current dataset."
)


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
is_ready, message = check_api_key()
if not is_ready:
    st.warning("**AI features are not active**")
    st.markdown(message)
    st.stop()


# ---------------------------------------------------------------------------
# Load active dataset and build the shared context
# ---------------------------------------------------------------------------
if "uploaded_data" in st.session_state:
    df = st.session_state["uploaded_data"].copy()
    source = st.session_state["uploaded_name"]
else:
    df = load_customer_data().copy()
    source = "Default dataset"

st.caption(f"Source: **{source}** ({len(df):,} customers)")

data_context = summarize_data_context(df)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_email, tab_strategy, tab_chat = st.tabs(
    ["Email Generator", "Strategy Advisor", "Free-Form Chat"]
)


# ===========================================================================
# TAB 1: Email Generator
# ===========================================================================
with tab_email:
    st.markdown(
        "Draft a marketing email tailored to a specific customer segment. "
        "Tone, framing, and product recommendations are adjusted automatically "
        "based on the cluster profile."
    )

    c1, c2 = st.columns(2)
    with c1:
        target_cluster = st.selectbox(
            "Target cluster",
            options=["Prosperous", "Families"],
            index=0,
            key="email_cluster",
        )
        tone = st.selectbox(
            "Tone",
            options=[
                "Match the cluster (auto)",
                "Formal and exclusive",
                "Warm and friendly",
                "Urgent and promotional",
            ],
            index=0,
            key="email_tone",
        )
    with c2:
        occasion = st.text_input(
            "Occasion or topic",
            placeholder="e.g., Riesling tasting event on June 15",
            key="email_occasion",
        )
        length = st.selectbox(
            "Length",
            options=["Short (under 100 words)", "Medium (around 200 words)"],
            index=1,
            key="email_length",
        )

    extra = st.text_area(
        "Optional additional instructions",
        placeholder="e.g., Mention the cooperation with vineyard X. Use the recipient's first name.",
        key="email_extra",
        height=80,
    )

    if st.button("Generate email", type="primary", key="generate_email"):
        if not occasion.strip():
            st.warning("Please enter an occasion or topic.")
        else:
            cluster_profile = (
                "PROSPEROUS cluster: high income (avg EUR 72k), low family size "
                "(avg 2.0), 39% are parents, premium-club target audience, "
                "values exclusivity and quality, low web visits (3.6/month) - "
                "they purchase deliberately."
                if target_cluster == "Prosperous"
                else
                "FAMILIES cluster: mid income (avg EUR 40k), larger family size "
                "(avg 2.9), 91% are parents, value bundle deals and discounts, "
                "high web visits (6.4/month), price-sensitive."
            )

            system_prompt = (
                "You are a marketing copywriter for a German wine reseller "
                "(Weingut Robert Weil, Frankfurt). The reseller sells "
                "high-quality wines along with gourmet products (meat, fish, "
                "sweets, fruits). Write in English. Output only the email - "
                "a subject line on the first line prefixed with 'Subject: ', "
                "then a blank line, then the body. No preamble, no commentary."
            )

            user_prompt = (
                f"Write a marketing email.\n\n"
                f"Target audience: {cluster_profile}\n\n"
                f"Occasion / topic: {occasion}\n"
                f"Tone: {tone}\n"
                f"Length: {length}\n"
            )
            if extra.strip():
                user_prompt += f"Additional instructions: {extra}\n"

            with st.spinner("Drafting email..."):
                try:
                    email_text = call_claude(
                        messages=[{"role": "user", "content": user_prompt}],
                        system=system_prompt,
                        max_tokens=800,
                    )
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    st.stop()

            st.divider()
            st.markdown("##### Draft")
            st.code(email_text, language="markdown")
            st.caption(
                "Tip: click the copy icon in the top-right of the box "
                "to copy the email."
            )


# ===========================================================================
# TAB 2: Strategy Advisor
# ===========================================================================
with tab_strategy:
    st.markdown(
        "Get prioritized, data-driven action recommendations for the current "
        "customer base. Recommendations are tailored to your dataset."
    )

    focus = st.selectbox(
        "Focus area",
        options=[
            "General recommendations",
            "Increase Premium Club enrollment",
            "Reactivate inactive customers",
            "Improve marketing campaign acceptance",
            "Grow Families segment revenue",
        ],
        index=0,
        key="strategy_focus",
    )

    if st.button("Generate recommendations", type="primary", key="generate_strategy"):
        system_prompt = (
            "You are a senior business strategist advising a mid-sized German "
            "wine reseller (Weingut Robert Weil, Frankfurt). Their strategic "
            "goal is to grow revenue, particularly through a premium-club "
            "membership program targeting the Prosperous cluster. You receive "
            "a summary of the merchant's customer dataset. Provide 3 to 5 "
            "specific, actionable recommendations. Each recommendation must:\n"
            "- Reference concrete numbers from the data summary\n"
            "- State the expected business outcome\n"
            "- Suggest a concrete first step\n"
            "Format as a numbered list. Be concise: 2-4 sentences per "
            "recommendation. No introduction, no conclusion."
        )

        user_prompt = (
            f"{data_context}\n\n"
            f"Focus area for the recommendations: {focus}\n\n"
            f"Give 3-5 prioritized, specific recommendations."
        )

        with st.spinner("Analyzing data and generating recommendations..."):
            try:
                advice = call_claude(
                    messages=[{"role": "user", "content": user_prompt}],
                    system=system_prompt,
                    max_tokens=1200,
                )
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        st.divider()
        st.markdown("##### Recommendations")
        st.markdown(advice)


# ===========================================================================
# TAB 3: Free-Form Chat
# ===========================================================================
with tab_chat:
    st.markdown(
        "Ask any question about your customer base. The assistant has "
        "access to a summary of the current dataset."
    )

    # Initialize chat history per session
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested starter questions as quick buttons
    if not st.session_state["chat_history"]:
        st.markdown("**Suggested questions to start with:**")
        cols = st.columns(2)
        starters = [
            "Which cluster has higher customer lifetime value potential?",
            "What is the biggest revenue opportunity I'm missing?",
            "How should I prioritize my marketing budget across clusters?",
            "What types of products should I add to the catalog?",
        ]
        for i, q in enumerate(starters):
            if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending question (from starter click) or new chat input
    user_input = st.chat_input("Ask anything about your customers...")
    pending = st.session_state.pop("pending_question", None)
    question = user_input or pending

    if question:
        # Show user message immediately
        st.session_state["chat_history"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        system_prompt = (
            "You are a helpful data analyst assisting a German wine reseller "
            "(Weingut Robert Weil, Frankfurt). You are given a summary of "
            "their customer dataset. Answer the merchant's questions based "
            "on this summary. Be specific and reference concrete numbers when "
            "possible. If the summary does not contain the information needed "
            "to answer, say so clearly and suggest what additional data would "
            "help. Keep answers concise: 3-6 sentences unless more detail is "
            "explicitly requested."
        )

        # Build messages: just the current question with context.
        # We deliberately don't send the full chat history to keep token
        # usage low for a student project; each Q is treated independently.
        api_messages = [
            {
                "role": "user",
                "content": f"{data_context}\n\nQuestion: {question}",
            }
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = call_claude(
                        messages=api_messages,
                        system=system_prompt,
                        max_tokens=800,
                    )
                except Exception as e:
                    answer = f"API call failed: {e}"
            st.markdown(answer)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )

    if st.session_state["chat_history"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()
