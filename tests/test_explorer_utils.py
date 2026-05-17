import math
from unittest.mock import patch

import pandas as pd
import plotly.express as px

from scripts.components.utils import (
    apply_dark_theme,
    classify_empirical_discourse,
    compute_proportion_stats,
    highlight_search_terms,
    metric_card,
    render_glass_card,
    render_proportion_chart,
    render_speaker_leaderboard,
    render_statistical_report,
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


def test_classify_empirical_discourse_basic():
    df = pd.DataFrame(
        {
            "text": [
                "This has data and evidence.",
                "This is simple speech.",
                "Another fact based claim.",
            ]
        }
    )
    keywords = ["data", "fact"]

    # Match whole words only
    res = classify_empirical_discourse(df.copy(), keywords, whole_words=True)
    assert bool(res.loc[0, "is_empirical_proxy"]) is True
    assert bool(res.loc[1, "is_empirical_proxy"]) is False
    assert bool(res.loc[2, "is_empirical_proxy"]) is True


def test_classify_empirical_discourse_word_boundary():
    df = pd.DataFrame({"text": ["This is factual.", "This is a fact."]})
    keywords = ["fact"]

    # Match whole words only (factual should NOT match fact)
    res_whole = classify_empirical_discourse(df.copy(), keywords, whole_words=True)
    assert bool(res_whole.loc[0, "is_empirical_proxy"]) is False
    assert bool(res_whole.loc[1, "is_empirical_proxy"]) is True

    # Partial match
    res_partial = classify_empirical_discourse(df.copy(), keywords, whole_words=False)
    assert bool(res_partial.loc[0, "is_empirical_proxy"]) is True
    assert bool(res_partial.loc[1, "is_empirical_proxy"]) is True


def test_classify_empirical_discourse_empty():
    df = pd.DataFrame()
    res = classify_empirical_discourse(df, ["data"])
    assert res.empty

    df_no_text = pd.DataFrame({"other": ["hello"]})
    res_no_text = classify_empirical_discourse(df_no_text, ["hello"])
    assert bool(res_no_text.loc[0, "is_empirical_proxy"]) is False


@patch("scripts.components.utils.st")
def test_metric_card_smoke(mock_st):
    metric_card(title="Test Title", value="123", subtext="Some Subtext", accent_color="#ff0000")
    mock_st.markdown.assert_called_once()
    args = mock_st.markdown.call_args[0][0]
    assert "Test Title" in args
    assert "123" in args
    assert "Some Subtext" in args
    assert "#ff0000" in args


@patch("scripts.components.utils.st")
def test_render_glass_card_smoke(mock_st):
    render_glass_card(title="My Insight", content="Some rich findings.", border_color="#ffffff")
    mock_st.markdown.assert_called_once()
    args = mock_st.markdown.call_args[0][0]
    assert "My Insight" in args
    assert "Some rich findings." in args
    assert "#ffffff" in args


@patch("scripts.components.utils.st")
def test_render_statistical_report_smoke(mock_st):
    stats = {"Group A": "12.0%", "Group B": "15.5%"}
    render_statistical_report(
        title="Robustness Check",
        stats=stats,
        interpretation="Matches theoretical priors.",
        border_color="#00ff00",
    )
    mock_st.markdown.assert_called_once()
    args = mock_st.markdown.call_args[0][0]
    assert "Robustness Check" in args
    assert "Group A" in args
    assert "12.0%" in args
    assert "Group B" in args
    assert "15.5%" in args
    assert "Matches theoretical priors." in args
    assert "#00ff00" in args


@patch("scripts.components.utils.st")
def test_render_proportion_chart_bar(mock_st):
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "is_empirical_proxy": [1, 1, 0, 1],
        }
    )
    # Test vertical bar
    stats = render_proportion_chart(
        df=df,
        group_cols=["group"],
        x="group",
        chart_type="bar",
        title="Proportions of A vs B",
        color="group",
    )
    assert not stats.empty
    assert "mean" in stats.columns
    mock_st.plotly_chart.assert_called_once()


@patch("scripts.components.utils.st")
def test_render_proportion_chart_line(mock_st):
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "is_empirical_proxy": [1, 0, 0, 1],
        }
    )
    # Test line chart
    stats = render_proportion_chart(
        df=df,
        group_cols=["year"],
        x="year",
        chart_type="line",
        title="Trend over time",
    )
    assert not stats.empty
    mock_st.plotly_chart.assert_called_once()


@patch("scripts.components.utils.st")
def test_render_speaker_leaderboard(mock_st):
    from unittest.mock import MagicMock

    mock_st.columns.return_value = (MagicMock(), MagicMock())
    df = pd.DataFrame(
        {
            "speaker": ["Alice", "Alice", "Bob", "Bob", "Bob", "Bob"],
            "party": ["D", "D", "R", "R", "R", "R"],
            "is_empirical_proxy": [1, 1, 0, 0, 0, 1],  # Alice: 2/2=100%, Bob: 1/4=25%
        }
    )
    # Alice has 2 sentences, Bob has 4 sentences. Let's filter min_sentences=3 (only Bob should show)
    render_speaker_leaderboard(df, target_col="is_empirical_proxy", min_sentences=3)
    mock_st.columns.assert_called_once()
    mock_st.table.assert_called()

    # Bob's stats
    tables = [call[0][0] for call in mock_st.table.call_args_list]
    assert len(tables) == 2
    assert "Bob" in tables[0].to_string()
    assert "Alice" not in tables[0].to_string()  # Alice filtered out
