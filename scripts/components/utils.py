from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def apply_dark_theme(fig, is_categorical=False):
    """Utility helper to apply a gorgeous unified dark theme to Plotly charts."""
    layout_update: dict[str, Any] = dict(
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
    stats["se"] = ((stats["mean"] * (1 - stats["mean"]) / stats["count"]).clip(lower=0)) ** 0.5
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


def classify_empirical_discourse(df: pd.DataFrame, keywords: list[str], whole_words: bool = True) -> pd.DataFrame:
    """Classifies text inside a DataFrame into empirical discourse based on keywords list.

    Creates or updates the 'is_empirical_proxy' column in-place.
    """
    if df.empty or "text" not in df.columns:
        df["is_empirical_proxy"] = False
        return df

    if not keywords:
        df["is_empirical_proxy"] = False
        return df

    escaped_kws = [re.escape(kw) for kw in keywords]
    if whole_words:
        pattern = "|".join(rf"\b{kw}\b" for kw in escaped_kws)
    else:
        pattern = "|".join(escaped_kws)

    df["is_empirical_proxy"] = df["text"].str.contains(pattern, case=False, na=False)
    return df


def metric_card(title: str, value: str, subtext: str = "", accent_color: str = "#2b5cff") -> None:
    """Utility to render a beautiful unified glassmorphic metric card inside Streamlit."""
    st.markdown(
        f"""<div style="
background: rgba(22, 28, 45, 0.45);
border: 1px solid rgba(255, 255, 255, 0.05);
border-left: 4px solid {accent_color};
border-radius: 12px;
padding: 22px 18px;
box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
margin-bottom: 15px;
backdrop-filter: blur(8px);
-webkit-backdrop-filter: blur(8px);
">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
<div style="font-size: 0.82rem; font-weight: 600; color: rgba(255, 255, 255, 0.55); text-transform: uppercase; letter-spacing: 0.8px;">{title}</div>
<div style="width: 8px; height: 8px; border-radius: 50%; background-color: {accent_color}; box-shadow: 0 0 8px {accent_color};"></div>
</div>
<div style="font-size: 1.9rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{value}</div>
<div style="font-size: 0.76rem; color: rgba(255, 255, 255, 0.38); margin-top: 3px; font-weight: 400;">{subtext}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_glass_card(title: str, content: str, border_color: str = "#ab63fa") -> None:
    """Renders a beautiful generic glassmorphic information/insight card."""
    st.markdown(
        f"""<div class="glass-card" style="border-left: 4px solid {border_color}; margin-top: 15px;">
<h5 style="color: #ffffff; margin-top: 0; font-size: 1rem; font-weight: 600;">{title}</h5>
<p style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-bottom: 0; line-height: 1.45;">
{content}
</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_statistical_report(
    title: str, stats: dict[str, Any], interpretation: str, border_color: str = "#ab63fa"
) -> None:
    """Renders a standard, professional, statistical comparison report block in glassmorphic card format.

    Args:
        title: Title of the statistical report.
        stats: Dictionary of name-value pairs to display in a grid.
               Example: {"Minority (N=...)": "12.5%", "Majority (N=...)": "10.1%"}
        interpretation: Text summarizing the statistical interpretation.
        border_color: Border accent color.
    """
    stats_cols_html = ""
    for label, val in stats.items():
        stats_cols_html += (
            f"<div>"
            f'<div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase;">{label}</div>'
            f'<div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-top: 3px; line-height: 1.2;">{val}</div>'
            f"</div>"
        )

    st.markdown(
        f"""<div class="glass-card" style="margin-top: 15px; border-left: 4px solid {border_color};">
<h5 style="color: #ffffff; margin-top: 0; font-size: 1rem; font-weight: 600;">{title}</h5>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 12px; margin-top: 10px;">
{stats_cols_html}
</div>
<p style="font-size: 0.82rem; color: rgba(255,255,255,0.75); margin-bottom: 0; line-height: 1.45; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px; margin-top: 12px;">
{interpretation}
</p>
</div>""",
        unsafe_allow_html=True,
    )


def render_proportion_chart(
    df: pd.DataFrame,
    group_cols: list[str],
    x: str,
    y: str = "mean_pct",
    target_col: str = "is_empirical_proxy",
    chart_type: str = "bar",
    error_y: str = "ci_95_pct",
    title: str = "",
    color: str | None = None,
    barmode: str = "group",
    orientation: str = "v",
    color_discrete_map: dict[str, str] | None = None,
    color_discrete_sequence: list[str] | None = None,
    color_continuous_scale: str | None = None,
    labels: dict[str, str] | None = None,
    category_orders: dict[str, list] | None = None,
    height: int | None = None,
    hovermode: str | None = None,
    yaxis_range_mult: float | None = None,
    mapping_col: str | None = None,
    mapping_dict: dict | None = None,
    new_col_name: str | None = None,
    use_container_width: bool = True,
) -> pd.DataFrame:
    """Aggregates binary target_col over group_cols, applies optional mapping,

    and renders a beautiful standard Plotly chart with error bars and dark theme.

    Returns the computed statistics DataFrame.
    """
    if df.empty:
        st.warning(f"No data available to plot: {title}")
        return pd.DataFrame()

    # 1. Compute stats
    stats = compute_proportion_stats(df, group_cols, target_col)

    if stats.empty:
        st.warning(f"No aggregated stats computed for plot: {title}")
        return stats

    # 2. Apply optional column mapping (e.g. minority maps to Status)
    if mapping_col and mapping_dict and new_col_name and mapping_col in stats.columns:
        stats[new_col_name] = stats[mapping_col].map(mapping_dict)

    # 3. Handle orientation logic
    kwargs: dict[str, Any] = {
        "title": title,
        "labels": labels or {},
    }

    if category_orders:
        kwargs["category_orders"] = category_orders

    if color:
        kwargs["color"] = color

    if color_discrete_map:
        kwargs["color_discrete_map"] = color_discrete_map

    if color_discrete_sequence:
        kwargs["color_discrete_sequence"] = color_discrete_sequence

    if height:
        kwargs["height"] = height

    # 4. Generate figures based on type
    if chart_type == "bar":
        if color_continuous_scale and "color" in kwargs:
            kwargs.pop("color_discrete_map", None)
            kwargs.pop("color_discrete_sequence", None)
            kwargs["color_continuous_scale"] = color_continuous_scale
            kwargs["color"] = y

        if orientation == "h":
            kwargs["error_x"] = error_y
            kwargs["orientation"] = "h"
            fig = px.bar(stats, x=y, y=x, barmode=barmode, **kwargs)
        else:
            kwargs["error_y"] = error_y
            kwargs["orientation"] = "v"
            fig = px.bar(stats, x=x, y=y, barmode=barmode, **kwargs)

        apply_dark_theme(fig, is_categorical=(color_continuous_scale is None))
    elif chart_type == "line":
        kwargs["error_y"] = error_y
        kwargs["markers"] = True
        fig = px.line(stats, x=x, y=y, **kwargs)
        apply_dark_theme(fig, is_categorical=False)
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    # 5. Fine tune layout
    layout_update: dict[str, Any] = {}
    if hovermode:
        layout_update["hovermode"] = hovermode

    if yaxis_range_mult and y in stats.columns:
        max_val = stats[y].max()
        layout_update["yaxis"] = dict(
            range=[0, max_val * yaxis_range_mult],
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showgrid=True,
        )

    if layout_update:
        fig.update_layout(**layout_update)

    # 6. Render
    st.plotly_chart(fig, use_container_width=use_container_width)
    return stats


def render_speaker_leaderboard(
    df: pd.DataFrame,
    target_col: str = "is_empirical_proxy",
    min_sentences: int = 50,
    speaker_col: str = "speaker",
    party_col: str = "party",
) -> None:
    """Computes, filters, and renders side-by-side tables for the most and least empirical speakers."""
    if df.empty or speaker_col not in df.columns or target_col not in df.columns:
        st.info("Speaker details or empirical proxy columns are not available.")
        return

    # Aggregate speaker statistics
    group_fields = [speaker_col]
    if party_col in df.columns:
        group_fields.append(party_col)

    speaker_stats = (
        df.groupby(group_fields, observed=False)
        .agg(empirical_pct=(target_col, "mean"), sentence_count=(target_col, "count"))
        .reset_index()
    )

    speaker_stats["Empirical %"] = (speaker_stats["empirical_pct"] * 100).round(1)
    speaker_filtered = speaker_stats[speaker_stats["sentence_count"] >= min_sentences]

    if not speaker_filtered.empty:
        col_lead1, col_lead2 = st.columns(2)

        # Standard column names for display
        rename_dict = {
            speaker_col: "Lawmaker",
            "Empirical %": "Empirical %",
            "sentence_count": "Total Sentences",
        }
        if party_col in df.columns:
            rename_dict[party_col] = "Party"

        show_cols = [speaker_col]
        if party_col in df.columns:
            show_cols.append(party_col)
        show_cols.extend(["Empirical %", "sentence_count"])

        with col_lead1:
            st.markdown("##### Top 10 Most Empirical Speakers")
            top_sp = speaker_filtered.sort_values("Empirical %", ascending=False).head(10).reset_index(drop=True)
            top_sp.index += 1
            st.table(top_sp[show_cols].rename(columns=rename_dict))

        with col_lead2:
            st.markdown("##### Top 10 Least Empirical Speakers")
            bot_sp = speaker_filtered.sort_values("Empirical %", ascending=True).head(10).reset_index(drop=True)
            bot_sp.index += 1
            st.table(bot_sp[show_cols].rename(columns=rename_dict))
    else:
        st.info("No speakers met the minimum sentence count threshold. Try lowering the slider.")
