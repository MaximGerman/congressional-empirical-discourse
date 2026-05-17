import math
import re

import pandas as pd


def apply_dark_theme(fig, is_categorical=False):
    """Utility helper to apply a gorgeous unified dark theme to Plotly charts."""
    layout_update = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        font_family="'Outfit', sans-serif",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    if not is_categorical:
        layout_update["xaxis"] = dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showgrid=True,
        )
        layout_update["yaxis"] = dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showgrid=True,
        )
    fig.update_layout(**layout_update)
    return fig


def z_test_p_value(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Computes the Z-test statistic and two-tailed p-value for two independent proportions.

    Uses the standard normal distribution CDF calculated via math.erf.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0

    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0, 1.0

    # Standard error of the difference
    se_diff = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se_diff == 0:
        return 0.0, 1.0

    z = (p1 - p2) / se_diff
    # Two-tailed p-value: 1 - erf(|z| / sqrt(2))
    p_val = 1.0 - math.erf(abs(z) / math.sqrt(2))
    return z, p_val


def compute_proportion_stats(
    df: pd.DataFrame, group_cols: list, target_col: str = "is_empirical_proxy"
) -> pd.DataFrame:
    """Computes mean, count, standard error, and 95% Confidence Intervals for a binary column.

    Enforces observed=False to match pandas standard.
    """
    if df.empty:
        # Return empty DataFrame with expected columns
        cols = [*group_cols, "mean", "count", "se", "ci_95", "mean_pct", "ci_95_pct"]
        return pd.DataFrame(columns=cols)

    # Group and aggregate
    stats = (
        df.groupby(group_cols, observed=False).agg(mean=(target_col, "mean"), count=(target_col, "count")).reset_index()
    )

    # Calculate standard error: sqrt(p * (1 - p) / n)
    stats["se"] = (stats["mean"] * (1 - stats["mean"]) / stats["count"]).apply(lambda x: math.sqrt(x) if x > 0 else 0.0)
    # 95% Confidence Interval limit (Z = 1.96)
    stats["ci_95"] = stats["se"] * 1.96

    # Convert to percentages for plotting
    stats["mean_pct"] = stats["mean"] * 100.0
    stats["ci_95_pct"] = stats["ci_95"] * 100.0

    return stats


def highlight_search_terms(
    text: str,
    query: str,
    case_sensitive: bool = False,
    whole_words: bool = False,
    is_regex: bool = False,
) -> str:
    """Highlights search terms inside text by wrapping them in beautiful glassmorphic HTML spans.

    Args:
        text: The source text block.
        query: The search query (can be multi-word or regex).
        case_sensitive: Whether matching is case-sensitive.
        whole_words: Whether to only match whole words.
        is_regex: Whether the query should be treated as a raw regular expression.
    """
    if not text or not query:
        return text

    flags = 0 if case_sensitive else re.IGNORECASE

    if is_regex:
        # Direct regex search highlighting path
        try:
            pattern_str = f"({query})"
            if whole_words:
                pattern_str = rf"\b{pattern_str}\b"
            pattern = re.compile(pattern_str, flags)
        except Exception:
            return text
    else:
        # Extract terms to search and escape them to be safe in regex
        # Support both multi-word space-separated and exact matches
        raw_terms = [q.strip() for q in query.split() if q.strip()]
        if not raw_terms:
            return text

        # Sort terms by length descending to avoid partial highlights of sub-words first
        raw_terms.sort(key=len, reverse=True)

        escaped_terms = []
        for t in raw_terms:
            try:
                escaped = re.escape(t)
                if whole_words:
                    escaped = rf"\b{escaped}\b"
                escaped_terms.append(escaped)
            except Exception:
                continue

        if not escaped_terms:
            return text

        # Combine into a single OR regex pattern
        pattern_str = f"({'|'.join(escaped_terms)})"

        try:
            pattern = re.compile(pattern_str, flags)
        except Exception:
            return text

    # Premium style: Glassmorphic highlight card with fine border and golden glow
    highlight_style = (
        "background: rgba(255, 213, 79, 0.22); "
        "color: #ffe082; "
        "border: 1px solid rgba(255, 213, 79, 0.3); "
        "padding: 2px 4px; "
        "border-radius: 4px; "
        "font-weight: 600; "
        "box-shadow: 0 0 8px rgba(255, 213, 79, 0.15);"
    )

    return pattern.sub(f'<span style="{highlight_style}">\\1</span>', text)
