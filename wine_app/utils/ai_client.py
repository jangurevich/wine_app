"""
Anthropic API client and helpers for the AI Assistant.

Provides:
- get_client: returns a cached Anthropic client (or None if no key configured)
- check_api_key: status check + user-friendly message
- summarize_data_context: compact text summary of the current dataset that
  is cheap to send as context to the model
- call_claude: thin wrapper around the Messages API with sensible defaults
"""

from typing import List, Dict, Optional

import pandas as pd
import streamlit as st


MODEL = "claude-sonnet-4-5"
MAX_TOKENS_DEFAULT = 1024


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    """
    Return a cached Anthropic client. Returns None if the SDK is missing
    or the API key is not configured.
    """
    try:
        import anthropic
    except ImportError:
        return None

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key or api_key.startswith("sk-ant-...") or api_key == "":
        return None

    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def check_api_key() -> tuple[bool, str]:
    """
    Check whether the AI features are usable.
    Returns (is_ready, user_message).
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, (
            "The `anthropic` Python package is not installed. "
            "Run `pip install anthropic` and restart the app."
        )

    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        return False, (
            "No Anthropic API key configured. To enable AI features:\n\n"
            "1. Get a key at https://console.anthropic.com\n"
            "2. Open `.streamlit/secrets.toml.example`\n"
            "3. Rename it to `secrets.toml` and paste your key\n"
            "4. Restart the app"
        )

    if api_key.startswith("sk-ant-..."):
        return False, (
            "The API key in `.streamlit/secrets.toml` is still the placeholder. "
            "Replace `sk-ant-...` with your actual key from "
            "https://console.anthropic.com"
        )

    return True, "AI features ready."


# ---------------------------------------------------------------------------
# Data context
# ---------------------------------------------------------------------------

def summarize_data_context(df: pd.DataFrame) -> str:
    """
    Build a compact text summary of the dataset that can be included in
    prompts without sending raw rows. Aimed at <500 tokens.
    """
    n = len(df)
    n_prosperous = int((df["Cluster_Label"] == "Prosperous").sum())
    n_families = int((df["Cluster_Label"] == "Families").sum())

    def cluster_stats(sub: pd.DataFrame) -> str:
        if len(sub) == 0:
            return "no customers"
        return (
            f"avg income EUR {sub['Income'].mean():,.0f}, "
            f"avg spending EUR {sub['Spent'].mean():,.0f}, "
            f"avg wines spending EUR {sub['Wines'].mean():,.0f}, "
            f"avg family size {sub['Family_size'].mean():.2f}, "
            f"avg recency {sub['Recency'].mean():.1f} days, "
            f"avg web visits/month {sub['NumWebVisitsMonth'].mean():.1f}"
        )

    prosperous = df[df["Cluster_Label"] == "Prosperous"]
    families = df[df["Cluster_Label"] == "Families"]

    # Recency-based segments within Prosperous
    inactive_prosp = int(((prosperous["Recency"] > 60)).sum())

    summary = (
        f"DATASET SUMMARY\n"
        f"Total customers: {n:,}\n"
        f"Prosperous cluster: {n_prosperous} ({n_prosperous/n:.0%}) - {cluster_stats(prosperous)}\n"
        f"Families cluster: {n_families} ({n_families/n:.0%}) - {cluster_stats(families)}\n"
        f"\n"
        f"NOTABLE SEGMENTS\n"
        f"- Prosperous customers inactive >60 days: {inactive_prosp}\n"
    )

    # Top spending categories per cluster
    cats = ["Wines", "Meat", "Fish", "Sweets", "Fruits", "Gold"]
    available = [c for c in cats if c in df.columns]
    if available:
        prosp_top = prosperous[available].mean().sort_values(ascending=False).head(3)
        fam_top = families[available].mean().sort_values(ascending=False).head(3)
        summary += (
            f"- Top categories Prosperous: " +
            ", ".join(f"{c} (EUR {v:.0f})" for c, v in prosp_top.items()) + "\n"
            f"- Top categories Families: " +
            ", ".join(f"{c} (EUR {v:.0f})" for c, v in fam_top.items()) + "\n"
        )

    return summary


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_claude(
    messages: List[Dict],
    system: Optional[str] = None,
    max_tokens: int = MAX_TOKENS_DEFAULT,
) -> str:
    """
    Call the Anthropic Messages API. Returns the text content of the response.
    Raises if the client is not available - callers should check first.
    """
    client = get_client()
    if client is None:
        raise RuntimeError("Anthropic client is not configured.")

    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    # Concatenate text blocks (there can be more than one)
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
