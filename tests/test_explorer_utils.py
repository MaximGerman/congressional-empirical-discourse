import math

import pandas as pd
import plotly.express as px

from scripts.components.utils import (
    apply_dark_theme,
    compute_proportion_stats,
    highlight_search_terms,
    z_test_p_value,
)


def test_apply_dark_theme_smoke():
    fig = px.bar(x=[1, 2], y=[3, 4])
    updated_fig = apply_dark_theme(fig, is_categorical=False)
    assert updated_fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert updated_fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert updated_fig.layout.font.color == "#ffffff"


def test_z_test_p_value_identical():
    # Identical proportions should result in Z=0, p=1
    z, p_val = z_test_p_value(0.5, 100, 0.5, 100)
    assert z == 0.0
    assert p_val == 1.0


def test_z_test_p_value_known_values():
    # Comparing 0.6 (60/100) vs 0.4 (40/100)
    # n1=100, n2=100
    # pooled p = (60+40)/200 = 0.5
    # SE = sqrt(0.5 * 0.5 * (1/100 + 1/100)) = sqrt(0.25 * 0.02) = sqrt(0.005) ≈ 0.07071
    # Z = (0.6 - 0.4) / 0.07071 ≈ 2.8284
    # p-value ≈ 1 - erf(2.8284 / sqrt(2)) = 1 - erf(2) ≈ 0.0047
    z, p_val = z_test_p_value(0.6, 100, 0.4, 100)
    assert math.isclose(z, 2.828427, rel_tol=1e-4)
    assert math.isclose(p_val, 0.0046777, rel_tol=1e-4)


def test_z_test_p_value_zero_or_negative_inputs():
    z, p_val = z_test_p_value(0.5, 0, 0.5, 100)
    assert z == 0.0
    assert p_val == 1.0

    z, p_val = z_test_p_value(0.5, 100, 0.5, -5)
    assert z == 0.0
    assert p_val == 1.0

    z, p_val = z_test_p_value(0.0, 100, 0.0, 100)
    assert z == 0.0
    assert p_val == 1.0


def test_compute_proportion_stats_empty():
    df = pd.DataFrame()
    stats = compute_proportion_stats(df, ["group"])
    assert stats.empty
    assert "mean" in stats.columns
    assert "ci_95_pct" in stats.columns


def test_compute_proportion_stats_basic():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "is_empirical_proxy": [1, 1, 1, 0, 0, 0, 0, 1],  # A: 3/4=75%, B: 1/4=25%
        }
    )
    stats = compute_proportion_stats(df, ["group"])
    assert len(stats) == 2

    # Group A
    stats_a = stats[stats["group"] == "A"].iloc[0]
    assert stats_a["mean"] == 0.75
    assert stats_a["count"] == 4
    # SE = sqrt(0.75 * 0.25 / 4) = sqrt(0.1875 / 4) = sqrt(0.046875) ≈ 0.2165
    assert math.isclose(stats_a["se"], 0.216506, rel_tol=1e-4)
    assert math.isclose(stats_a["ci_95"], 0.216506 * 1.96, rel_tol=1e-4)
    assert stats_a["mean_pct"] == 75.0

    # Group B
    stats_b = stats[stats["group"] == "B"].iloc[0]
    assert stats_b["mean"] == 0.25
    assert stats_b["count"] == 4
    assert stats_b["mean_pct"] == 25.0


def test_highlight_search_terms_basic():
    text = "The quick brown fox jumps over the lazy dog."
    res = highlight_search_terms(text, "quick dog")
    # Should highlight "quick" and "dog"
    assert "quick" in res
    assert '<span style="' in res
    assert '<span style="' in res


def test_highlight_search_terms_case_insensitive():
    text = "The QUICK brown fox jumps."
    res = highlight_search_terms(text, "quick", case_sensitive=False)
    assert "QUICK" in res
    assert '<span style="' in res


def test_highlight_search_terms_case_sensitive():
    text = "The QUICK brown fox jumps."
    res = highlight_search_terms(text, "quick", case_sensitive=True)
    # Should NOT highlight "QUICK" because of case sensitivity
    assert '<span style="' not in res


def test_highlight_search_terms_whole_words():
    text = "This is a factual claim in a factory."
    # Whole words only should only match "fact" as a whole word
    res_whole = highlight_search_terms(text, "fact", whole_words=True)
    assert '<span style="' not in res_whole  # Neither "factual" nor "factory" are exactly "fact"

    res_part = highlight_search_terms(text, "fact", whole_words=False)
    assert '<span style="' in res_part  # Matches partial substring


def test_highlight_search_terms_regex():
    text = "The price is $123 or €456."
    # Regex matching digits
    res = highlight_search_terms(text, r"\d+", is_regex=True)
    assert "123" in res
    assert "456" in res
    assert '<span style="' in res
