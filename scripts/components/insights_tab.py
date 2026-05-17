import re

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.components.utils import (
    apply_dark_theme,
    compute_proportion_stats,
    z_test_p_value,
)


def render_insights_tab(global_stats, empirical_keywords=None, whole_words=True):
    if global_stats is None:
        st.warning("Global statistics are unavailable. Please ensure the optimized dataset is generated.")
        return

    if empirical_keywords is None:
        empirical_keywords = [
            "data",
            "evidence",
            "statistics",
            "statistical",
            "study",
            "report",
            "research",
            "analysis",
            "percent",
            "%",
            "number",
            "increase",
            "decrease",
            "caused",
            "correlation",
            "fact",
            "empirical",
        ]

    st.header("Global Dataset Insights")
    st.markdown(
        "This tab provides a scientific statistical summary of the **entire** dataset (3.5M+ rows) "
        "leveraging pre-computed metadata analysis and stratified row-group sampling."
    )

    # Section 1: Data Completeness
    st.subheader("1. Data Completeness & Null Analysis")
    null_df = (global_stats["null_counts"] / global_stats["total_rows"] * 100).reset_index()
    null_df.columns = ["Column", "Missing %"]
    null_df = null_df.sort_values("Missing %", ascending=False)

    fig_null = px.bar(
        null_df,
        x="Missing %",
        y="Column",
        orientation="h",
        title="Percentage of Missing Data by Column",
        color="Missing %",
        color_continuous_scale="Reds",
    )
    apply_dark_theme(fig_null, is_categorical=True)
    st.plotly_chart(fig_null, use_container_width=True)

    # Section 2: Temporal & Demographic Trends
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Temporal Volume")
        if "congress_counts" in global_stats:
            cong_df = global_stats["congress_counts"].reset_index()
            cong_df.columns = ["Congress", "Sentences"]
            fig_temp = px.line(
                cong_df, x="Congress", y="Sentences", markers=True, title="Speech Volume Across Congresses"
            )
            apply_dark_theme(fig_temp)
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Congress data not available.")

    with col2:
        st.subheader("3. Chamber Split")
        if "chamber_counts" in global_stats:
            cham_df = global_stats["chamber_counts"].reset_index()
            cham_df.columns = ["Chamber", "Count"]
            fig_cham = px.pie(
                cham_df,
                values="Count",
                names="Chamber",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            apply_dark_theme(fig_cham, is_categorical=True)
            st.plotly_chart(fig_cham, use_container_width=True)
        else:
            st.info("Chamber data not available.")

    # Section 3: Distribution Analysis (Sampled)
    st.markdown("---")
    st.subheader("4. Scientific Distributions (Sampled Dataset)")
    s_col1, s_col2 = st.columns(2)
    sample_df = global_stats["sample_df"].copy()

    with s_col1:
        # Ideology Distribution
        if "nominate_dim1" in sample_df.columns:
            fig_ideo = px.histogram(
                sample_df,
                x="nominate_dim1",
                color="party" if "party" in sample_df.columns else None,
                marginal="box",
                title="Ideology Distribution (DW-NOMINATE)",
                color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
                labels={"nominate_dim1": "Lib-Con Ideology"},
            )
            apply_dark_theme(fig_ideo)
            st.plotly_chart(fig_ideo, use_container_width=True)
        else:
            st.info("Ideology data (DW-NOMINATE) not available.")

    with s_col2:
        # Match Score Distribution
        if "match_score" in sample_df.columns:
            fig_match = px.histogram(
                sample_df,
                x="match_score",
                title="Speaker Match Confidence Distribution",
                color_discrete_sequence=["#00CC96"],
            )
            apply_dark_theme(fig_match)
            st.plotly_chart(fig_match, use_container_width=True)
        else:
            st.info("Match score data not available.")

    # Apply Heuristic calculations for all subsequent analyses
    if "text" in sample_df.columns and "minority" in sample_df.columns:
        if not empirical_keywords:
            sample_df["is_empirical_proxy"] = False
        else:
            escaped_kws = [re.escape(kw) for kw in empirical_keywords]
            if whole_words:
                pattern = "|".join(rf"\b{kw}\b" for kw in escaped_kws)
            else:
                pattern = "|".join(escaped_kws)
            sample_df["is_empirical_proxy"] = sample_df["text"].str.contains(pattern, case=False, na=False)

        # SPATIAL US CHOROPLETH MAP
        st.markdown("---")
        st.subheader("Spatial Analysis: Empirical Discourse by State Delegation")
        st.markdown(
            "Explore spatial patterns in evidence-based legislative speech. "
            "Hover over states to view delegation size and empirical speech densities."
        )
        if "state_abbrev" in sample_df.columns:
            # Aggregate by state
            state_df = (
                sample_df.groupby("state_abbrev", observed=False)
                .agg(empirical_pct=("is_empirical_proxy", "mean"), sentence_count=("text", "count"))
                .reset_index()
            )
            state_df["Empirical %"] = (state_df["empirical_pct"] * 100).round(2)
            # Remove states with negligible data to clean map noise
            state_df = state_df[state_df["sentence_count"] >= 15]

            if not state_df.empty:
                fig_map = px.choropleth(
                    state_df,
                    locations="state_abbrev",
                    locationmode="USA-states",
                    color="Empirical %",
                    scope="usa",
                    color_continuous_scale="Purples",
                    hover_data=["sentence_count"],
                    labels={"state_abbrev": "State Code", "sentence_count": "Sample Sentences"},
                    title="Average Empirical Discourse (%) by US State Delegation",
                )
                fig_map.update_layout(
                    geo=dict(
                        bgcolor="rgba(0,0,0,0)",
                        lakecolor="rgba(12, 15, 23, 0.8)",
                        landcolor="rgba(25, 30, 48, 0.4)",
                        subunitcolor="rgba(255,255,255,0.12)",
                        showlakes=True,
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff",
                    font_family="'Outfit', sans-serif",
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("Insufficient data to display state-level spatial analysis.")
        else:
            st.info("State abbreviation columns (state_abbrev) not present in dataset.")

        # Section 4: The Minority Gap (Heuristic Proxy with 95% Confidence Intervals)
        st.markdown("---")
        st.subheader("5. The 'Minority Gap' Trend with 95% CIs")
        st.info(
            "Note: Since the full RoBERTa/DeBERTa classifier is still being fine-tuned for this period, "
            "we use a keyword-based heuristic to approximate empirical discourse frequency. "
            "You can customize these keywords in the sidebar!"
        )

        # Group by congress and minority status with confidence intervals
        gap_df = compute_proportion_stats(sample_df, ["congress", "minority"], "is_empirical_proxy")
        gap_df["Status"] = gap_df["minority"].map({1: "Minority", 0: "Majority"})

        fig_gap = px.line(
            gap_df,
            x="congress",
            y="mean_pct",
            color="Status",
            error_y="ci_95_pct",
            markers=True,
            title="Empirical Discourse: Majority vs. Minority with 95% Confidence Intervals",
            color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
            labels={"congress": "Congress", "mean_pct": "% Empirical Discourse"},
        )
        apply_dark_theme(fig_gap)
        fig_gap.update_layout(
            hovermode="x unified",
            yaxis_range=[0, gap_df["mean_pct"].max() * 1.3],
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        # Overall Z-test for minority vs majority
        min_mask = sample_df["minority"] == 1
        maj_mask = sample_df["minority"] == 0
        n_min = int(min_mask.sum())
        n_maj = int(maj_mask.sum())

        p_min = sample_df.loc[min_mask, "is_empirical_proxy"].mean() if n_min > 0 else 0.0
        p_maj = sample_df.loc[maj_mask, "is_empirical_proxy"].mean() if n_maj > 0 else 0.0

        z_stat, p_val = z_test_p_value(p_min, n_min, p_maj, n_maj)

        # Format p-value nicely
        if p_val < 0.001:
            p_str = "p < 0.001 (Highly Significant ***)"
            significance_desc = (
                "Democratic/Republican evidence differences are extremely robust and highly statistically significant."
            )
        elif p_val < 0.01:
            p_str = f"p = {p_val:.4f} (Very Significant **)"
            significance_desc = "The difference in evidence usage is very robust and statistically significant."
        elif p_val < 0.05:
            p_str = f"p = {p_val:.4f} (Statistically Significant *)"
            significance_desc = "The difference in evidence usage is statistically significant."
        else:
            p_str = f"p = {p_val:.4f} (Not Significant)"
            significance_desc = "The difference in evidence usage between the parties does not meet standard statistical significance thresholds."

        diff_pct = (p_min - p_maj) * 100

        # Display the stats card
        st.markdown(
            f"""
            <div class="glass-card" style="margin-top: 15px; border-left: 4px solid #ab63fa;">
                <h5 style="color: #ffffff; margin-top: 0; font-size: 1rem; font-weight: 600;">🔬 Statistical Robustness Report: Minority vs. Majority</h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 12px; margin-top: 10px;">
                    <div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase;">Minority (N={n_min:,})</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFD700;">{p_min * 100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase;">Majority (N={n_maj:,})</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #808080;">{p_maj * 100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase;">Difference (Gap)</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #00cc96;">{diff_pct:+.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase;">Z-Statistic / P-Value</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #ffe082; margin-top: 2px;">Z = {z_stat:.2f}<br><span style="font-size: 0.85rem; color: #ab63fa;">{p_str}</span></div>
                    </div>
                </div>
                <p style="font-size: 0.82rem; color: rgba(255,255,255,0.7); margin-bottom: 0; line-height: 1.4;">
                    <strong>Interpretation:</strong> {significance_desc} The data shows that the Minority party uses <strong>{abs(diff_pct):.2f}%</strong> {"more" if diff_pct > 0 else "less"} empirical arguments than the Majority party.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            > **Research Insight: The 'Power or Knowledge' Framework**
            >
            > The original Haim & Barak-Corren (2026) paper posits that empirical discourse is the 'weapon of the powerless.'
            > When a party loses formal agenda control (Majority -> Minority), they increase their reliance on data-driven
            > arguments to challenge dominant forces. This chart allows us to monitor if this gap persists into the 2017-2025 era.
            """
        )

        # Section: Leadership status and custom role analysis (Power or Knowledge)
        st.markdown("---")
        st.subheader("Power vs. Knowledge: Leadership Status Analysis")
        st.markdown(
            "Compare the evidence usage of **Committee Chairs** (who control the agenda), "
            "**Ranking Members** (the minority's lead voice), and **Regular Members**."
        )
        if "chairspeech" in sample_df.columns and "rankmemspeech" in sample_df.columns:

            def determine_role(row):
                if row.get("chairspeech") == 1:
                    return "Committee Chair (Majority Lead)"
                elif row.get("rankmemspeech") == 1:
                    return "Ranking Member (Minority Lead)"
                else:
                    return "Regular Member"

            sample_df["legislative_role"] = sample_df.apply(determine_role, axis=1)
            role_gap = compute_proportion_stats(sample_df, ["legislative_role"], "is_empirical_proxy")

            fig_role = px.bar(
                role_gap,
                x="legislative_role",
                y="mean_pct",
                error_y="ci_95_pct",
                color="legislative_role",
                title="Evidence Usage by Committee Leadership Role (with 95% CIs)",
                color_discrete_map={
                    "Committee Chair (Majority Lead)": "#FF4B4B",
                    "Ranking Member (Minority Lead)": "#FFD700",
                    "Regular Member": "#808080",
                },
                labels={"legislative_role": "Legislative Role", "mean_pct": "% Empirical Discourse"},
            )
            apply_dark_theme(fig_role, is_categorical=True)
            st.plotly_chart(fig_role, use_container_width=True)

            st.markdown(
                """
                > **Research Insight: The Leadership Gap**
                >
                > Leaders holding formal agenda-setting power (Chairs) face less institutional pressure to defend their positions
                > with data, using their institutional advantage instead. In contrast, Ranking Members must rely heavily on empirical assertions
                > to challenge and question bills.
                """
            )
        else:
            st.info("Leadership details (chairspeech/rankmemspeech) are not available in sample.")

        # Section: Career stages (Freshmen vs. Veteran lawmakers)
        st.markdown("---")
        st.subheader("Career Stages & Seniority Analysis")
        st.markdown(
            "Do newer members use data-heavy arguments to establish legislative credibility? "
            "We compare **Freshmen** (first term) against **Veteran Members**, and plot empirical rhetoric trends across seniority levels."
        )

        c_fresh1, c_fresh2 = st.columns(2)

        with c_fresh1:
            if "freshman" in sample_df.columns:
                fresh_gap = compute_proportion_stats(sample_df, ["freshman"], "is_empirical_proxy")
                fresh_gap["Career Stage"] = fresh_gap["freshman"].map({1: "Freshman (1st Term)", 0: "Veteran Member"})

                fig_fresh = px.bar(
                    fresh_gap,
                    x="Career Stage",
                    y="mean_pct",
                    error_y="ci_95_pct",
                    color="Career Stage",
                    title="Evidence Usage: Freshmen vs. Veteran Lawmakers (with 95% CIs)",
                    color_discrete_map={"Freshman (1st Term)": "#00cc96", "Veteran Member": "#ab63fa"},
                    labels={"mean_pct": "% Empirical Discourse"},
                )
                apply_dark_theme(fig_fresh, is_categorical=True)
                st.plotly_chart(fig_fresh, use_container_width=True)
            else:
                st.info("Freshman indicator not present in sample.")

        with c_fresh2:
            if "seniority" in sample_df.columns:
                sen_gap = compute_proportion_stats(sample_df, ["seniority"], "is_empirical_proxy")

                fig_sen = px.line(
                    sen_gap,
                    x="seniority",
                    y="mean_pct",
                    error_y="ci_95_pct",
                    markers=True,
                    title="Evidence Usage Trend by Seniority (Terms Served, with 95% CIs)",
                    labels={"seniority": "Seniority (Terms Served)", "mean_pct": "% Empirical Discourse"},
                )
                apply_dark_theme(fig_sen)
                st.plotly_chart(fig_sen, use_container_width=True)
            else:
                st.info("Seniority tenure details not available in sample.")

        # Section: Speaker Empirical Leaderboard
        st.markdown("---")
        st.subheader("Legislator Leaderboard: Who Uses the Most (and Least) Evidence?")
        st.markdown(
            "Rank individual lawmakers based on their empirical discourse rate. "
            "Use the slider to filter out speakers with low sentence volume and focus on active debaters."
        )
        if "speaker" in sample_df.columns:
            min_sentences = st.slider(
                "Minimum Sentences Spoken in Sample", min_value=10, max_value=200, value=50, step=10
            )

            # Aggregate speaker statistics
            speaker_stats = (
                sample_df.groupby(["speaker", "party"], observed=False)
                .agg(empirical_pct=("is_empirical_proxy", "mean"), sentence_count=("text", "count"))
                .reset_index()
            )

            speaker_stats["Empirical %"] = (speaker_stats["empirical_pct"] * 100).round(1)
            speaker_filtered = speaker_stats[speaker_stats["sentence_count"] >= min_sentences]

            if not speaker_filtered.empty:
                col_lead1, col_lead2 = st.columns(2)

                with col_lead1:
                    st.markdown("##### Top 10 Most Empirical Speakers")
                    top_sp = (
                        speaker_filtered.sort_values("Empirical %", ascending=False).head(10).reset_index(drop=True)
                    )
                    top_sp.index += 1
                    st.table(
                        top_sp[["speaker", "party", "Empirical %", "sentence_count"]].rename(
                            columns={
                                "speaker": "Lawmaker",
                                "party": "Party",
                                "Empirical %": "Empirical %",
                                "sentence_count": "Total Sentences",
                            }
                        )
                    )

                with col_lead2:
                    st.markdown("##### Top 10 Least Empirical Speakers")
                    bot_sp = speaker_filtered.sort_values("Empirical %", ascending=True).head(10).reset_index(drop=True)
                    bot_sp.index += 1
                    st.table(
                        bot_sp[["speaker", "party", "Empirical %", "sentence_count"]].rename(
                            columns={
                                "speaker": "Lawmaker",
                                "party": "Party",
                                "Empirical %": "Empirical %",
                                "sentence_count": "Total Sentences",
                            }
                        )
                    )
            else:
                st.info("No speakers met the minimum sentence count threshold. Try lowering the slider.")
        else:
            st.info("Speaker name details are not present in sample.")

        # Section 6: Polarization vs Empirical
        st.markdown("---")
        st.subheader("6. Ideological Extremity & Evidence")
        if "abs_dwnom1" in sample_df.columns:
            # Create bins for ideological extremity with confidence intervals
            sample_df["extremity_bin"] = pd.cut(sample_df["abs_dwnom1"], bins=5)
            pol_df = compute_proportion_stats(sample_df, ["extremity_bin"], "is_empirical_proxy")
            pol_df["Extremity"] = pol_df["extremity_bin"].astype(str)

            fig_pol = px.bar(
                pol_df,
                x="Extremity",
                y="mean_pct",
                error_y="ci_95_pct",
                title="Does Ideological Extremity Correlate with Data Usage? (95% CIs)",
                color="mean_pct",
                color_continuous_scale="Viridis",
                labels={
                    "Extremity": "Ideological Extremity (Absolute DW-NOMINATE)",
                    "mean_pct": "% Empirical Discourse",
                },
            )
            apply_dark_theme(fig_pol, is_categorical=True)
            st.plotly_chart(fig_pol, use_container_width=True)
        else:
            st.info("Ideological extremity data (abs_dwnom1) not available in sample.")

        # Section 7: Chamber Comparison (House vs. Senate)
        st.markdown("---")
        st.subheader("7. Chamber Comparison: House vs. Senate")
        if "chamber" in sample_df.columns:
            sample_df["Chamber"] = sample_df["chamber"].str.capitalize()
            chamber_gap = compute_proportion_stats(sample_df, ["Chamber", "minority"], "is_empirical_proxy")
            chamber_gap["Status"] = chamber_gap["minority"].map({1: "Minority", 0: "Majority"})

            fig_chamber = px.bar(
                chamber_gap,
                x="Chamber",
                y="mean_pct",
                color="Status",
                error_y="ci_95_pct",
                barmode="group",
                title="Evidence Usage by Chamber and Party Power Status (with 95% CIs)",
                color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                labels={"Chamber": "Chamber", "mean_pct": "% Empirical Discourse"},
            )
            apply_dark_theme(fig_chamber, is_categorical=True)
            st.plotly_chart(fig_chamber, use_container_width=True)

            st.markdown(
                """
                > **Research Insight: Chamber Dynamics**
                >
                > In the Senate, with longer term lengths and smaller body size (100 members), the majority-minority gap
                > is expected to show distinct behavior compared to the fast-paced, highly structured House of Representatives (435 members).
                """
            )
        else:
            st.info("Chamber data not available in sample.")

        # Section 8: Dynamic Natural Experiment transitions
        st.markdown("---")
        st.subheader("8. Natural Experiment: Congressional Power Shifts")
        st.markdown(
            "Select a chamber and a transition period to analyze how evidence-based discourse changed "
            "when agenda control shifted between parties."
        )

        col_shift1, col_shift2 = st.columns(2)
        with col_shift1:
            chamber_sel = st.selectbox("Select Target Chamber", ["House", "Senate"], index=0)
        with col_shift2:
            transition_sel = st.selectbox(
                "Select Congressional Transition",
                [
                    "115th to 116th Congress (2019)",
                    "116th to 117th Congress (2021)",
                    "117th to 118th Congress (2023)",
                ],
                index=0,
            )

        # Parse transition congress numbers
        if "115th to 116th" in transition_sel:
            before_cong, after_cong = 115, 116
            label_before, label_after = "115th Congress", "116th Congress"
        elif "116th to 117th" in transition_sel:
            before_cong, after_cong = 116, 117
            label_before, label_after = "116th Congress", "117th Congress"
        else:
            before_cong, after_cong = 117, 118
            label_before, label_after = "117th Congress", "118th Congress"

        # Determine agenda control states historically
        # House Majority: 115: Republican, 116: Democratic, 117: Democratic, 118: Republican
        # Senate Majority: 115: Republican, 116: Republican, 117: Democratic, 118: Democratic
        if chamber_sel == "House":
            maj_by_cong = {115: "Republican", 116: "Democratic", 117: "Democratic", 118: "Republican"}
        else:
            maj_by_cong = {115: "Republican", 116: "Republican", 117: "Democratic", 118: "Democratic"}

        maj_before = maj_by_cong[before_cong]
        maj_after = maj_by_cong[after_cong]
        has_power_shift = maj_before != maj_after

        st.markdown(
            f"Analyzing **{chamber_sel}** transition from **{label_before}** ({maj_before} Majority) "
            f"to **{label_after}** ({maj_after} Majority). "
            f"{'🔄 **Power shift occurred!**' if has_power_shift else '➖ **No power shift occurred (majority party retained control).**'}"  # noqa: RUF001
        )

        # Filter dataset to transition
        shift_df = sample_df[
            (sample_df["chamber"].str.lower() == chamber_sel.lower())
            & (sample_df["party"].isin(["Democratic", "Republican"]))
            & (sample_df["congress"].isin([before_cong, after_cong]))
        ].copy()

        if not shift_df.empty:
            # Group and calculate stats
            shift_grouped = compute_proportion_stats(shift_df, ["congress", "party"], "is_empirical_proxy")
            shift_grouped["Congress Label"] = shift_grouped["congress"].map(
                {
                    before_cong: f"{before_cong}th ({maj_before} Maj)",
                    after_cong: f"{after_cong}th ({maj_after} Maj)",
                }
            )
            shift_grouped = shift_grouped.sort_values("congress")

            fig_shift = px.line(
                shift_grouped,
                x="Congress Label",
                y="mean_pct",
                color="party",
                error_y="ci_95_pct",
                markers=True,
                title=f"Evidence Usage Shift: {chamber_sel} ({before_cong}th ➔ {after_cong}th Congress)",
                color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
                labels={
                    "party": "Party",
                    "mean_pct": "% Empirical Discourse",
                    "Congress Label": "Congressional Session",
                },
            )
            apply_dark_theme(fig_shift)
            fig_shift.update_layout(yaxis_range=[0, shift_grouped["mean_pct"].max() * 1.3])
            st.plotly_chart(fig_shift, use_container_width=True)

            # Perform Z-tests for BOTH parties across the transition
            # Democrats before vs after
            dem_before_mask = (shift_df["party"] == "Democratic") & (shift_df["congress"] == before_cong)
            dem_after_mask = (shift_df["party"] == "Democratic") & (shift_df["congress"] == after_cong)
            n_dem_b = int(dem_before_mask.sum())
            n_dem_a = int(dem_after_mask.sum())
            p_dem_b = shift_df.loc[dem_before_mask, "is_empirical_proxy"].mean() if n_dem_b > 0 else 0.0
            p_dem_a = shift_df.loc[dem_after_mask, "is_empirical_proxy"].mean() if n_dem_a > 0 else 0.0
            _dem_z, dem_p = z_test_p_value(p_dem_a, n_dem_a, p_dem_b, n_dem_b)
            dem_diff = (p_dem_a - p_dem_b) * 100

            # Republicans before vs after
            gop_before_mask = (shift_df["party"] == "Republican") & (shift_df["congress"] == before_cong)
            gop_after_mask = (shift_df["party"] == "Republican") & (shift_df["congress"] == after_cong)
            n_gop_b = int(gop_before_mask.sum())
            n_gop_a = int(gop_after_mask.sum())
            p_gop_b = shift_df.loc[gop_before_mask, "is_empirical_proxy"].mean() if n_gop_b > 0 else 0.0
            p_gop_a = shift_df.loc[gop_after_mask, "is_empirical_proxy"].mean() if n_gop_a > 0 else 0.0
            _gop_z, gop_p = z_test_p_value(p_gop_a, n_gop_a, p_gop_b, n_gop_b)
            gop_diff = (p_gop_a - p_gop_b) * 100

            def evaluate_alignment(party_name, diff, p_val, before_maj, after_maj):
                gained_power = (before_maj != party_name) and (after_maj == party_name)
                lost_power = (before_maj == party_name) and (after_maj != party_name)
                retained_power = (before_maj == party_name) and (after_maj == party_name)
                remained_minority = (before_maj != party_name) and (after_maj != party_name)

                if retained_power or remained_minority:
                    prediction = "No major changes predicted (retained power status)"
                    is_significant = p_val < 0.05
                    actual_dir = "increased" if diff > 0 else "decreased"
                    status = f"➖ **Neutral**: {party_name} evidence usage {actual_dir} by {abs(diff):.2f}% ({'statistically significant' if is_significant else 'not statistically significant'}, p={p_val:.3f})."  # noqa: RUF001
                    return prediction, status

                if gained_power:
                    prediction = "Evidence usage should **DECREASE** (gained majority/agenda control)"
                    if p_val < 0.05:
                        if diff < 0:
                            status = f"🟢 **Supports Theory**: Evidence usage significantly decreased by **{abs(diff):.2f}%** (p={p_val:.4f}). Gaining power reduced strategic reliance on data."
                        else:
                            status = f"🔴 **Contradicts Theory**: Evidence usage significantly *increased* by **{abs(diff):.2f}%** (p={p_val:.4f}) despite gaining agenda control."
                    else:
                        status = f"🟡 **Inconclusive**: Actual change was {diff:+.2f}% but not statistically significant (p={p_val:.3f})."
                    return prediction, status

                if lost_power:
                    prediction = "Evidence usage should **INCREASE** (lost majority/agenda control)"
                    if p_val < 0.05:
                        if diff > 0:
                            status = f"🟢 **Supports Theory**: Evidence usage significantly increased by **{abs(diff):.2f}%** (p={p_val:.4f}). Losing power forced a strategic shift to data-heavy discourse."
                        else:
                            status = f"🔴 **Contradicts Theory**: Evidence usage significantly *decreased* by **{abs(diff):.2f}%** (p={p_val:.4f}) despite losing power."
                    else:
                        status = f"🟡 **Inconclusive**: Actual change was {diff:+.2f}% but not statistically significant (p={p_val:.3f})."
                    return prediction, status

            dem_pred, dem_status = evaluate_alignment("Democratic", dem_diff, dem_p, maj_before, maj_after)
            gop_pred, gop_status = evaluate_alignment("Republican", gop_diff, gop_p, maj_before, maj_after)

            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #00cc96;">
                    <h5 style="color: #ffffff; margin-top: 0; font-size: 1rem; font-weight: 600;">📈 Live Transition Experiment: Strategic Hypothesis Evaluation</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 10px; margin-bottom: 15px;">
                        <div style="background: rgba(46, 91, 255, 0.08); border: 1px solid rgba(46, 91, 255, 0.15); padding: 15px; border-radius: 8px;">
                            <strong style="color: #2E5BFF; font-size: 0.9rem; text-transform: uppercase;">Democrats (Blue)</strong>
                            <div style="font-size: 1.5rem; font-weight: 700; margin-top: 5px;">{p_dem_b * 100:.2f}% ➔ {p_dem_a * 100:.2f}%</div>
                            <div style="font-size: 0.76rem; color: rgba(255,255,255,0.6); margin-top: 3px;">Change: <strong>{dem_diff:+.2f}%</strong> (N={n_dem_b:,} ➔ {n_dem_a:,})</div>
                            <div style="font-size: 0.78rem; color: #ffe082; margin-top: 8px; font-style: italic;">Theory: {dem_pred}</div>
                            <div style="font-size: 0.8rem; margin-top: 8px; font-weight: 500; line-height: 1.3;">{dem_status}</div>
                        </div>
                        <div style="background: rgba(255, 75, 75, 0.08); border: 1px solid rgba(255, 75, 75, 0.15); padding: 15px; border-radius: 8px;">
                            <strong style="color: #FF4B4B; font-size: 0.9rem; text-transform: uppercase;">Republicans (Red)</strong>
                            <div style="font-size: 1.5rem; font-weight: 700; margin-top: 5px;">{p_gop_b * 100:.2f}% ➔ {p_gop_a * 100:.2f}%</div>
                            <div style="font-size: 0.76rem; color: rgba(255,255,255,0.6); margin-top: 3px;">Change: <strong>{gop_diff:+.2f}%</strong> (N={n_gop_b:,} ➔ {n_gop_a:,})</div>
                            <div style="font-size: 0.78rem; color: #ffe082; margin-top: 8px; font-style: italic;">Theory: {gop_pred}</div>
                            <div style="font-size: 0.8rem; margin-top: 8px; font-weight: 500; line-height: 1.3;">{gop_status}</div>
                        </div>
                    </div>
                    <p style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.55); margin-bottom: 0; line-height: 1.4;">
                        <strong>Research Insight:</strong> The 'Strategic Discourse Shift' framework predicts that when formal legislative power transitions, the newly disempowered party experiences a substantial surge in evidence-based speech to compensate for their lack of agenda control. Meanwhile, the newly empowered party relaxes their empirical arguments.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Insufficient data for the {chamber_sel} transition in this sample.")

        st.markdown(
            """
            > **Research Insight: The 2019 Power Shift**
            >
            > In 2019 (116th Congress), Democrats gained the House majority. According to the strategic framework,
            > newly empowered Democrats should show a *decrease* in empirical discourse frequency, while newly disempowered
            > Republicans should display a corresponding *increase*.
            """
        )

        # Section 9: Committee-Level Heterogeneity
        st.markdown("---")
        st.subheader("9. Committee-Level Heterogeneity")
        if "committee_name" in sample_df.columns:
            comm_df = sample_df[sample_df["committee_name"].notna() & (sample_df["committee_name"] != "")].copy()

            if not comm_df.empty:
                top_committees = comm_df["committee_name"].value_counts().head(15).index
                comm_filtered = comm_df[comm_df["committee_name"].isin(top_committees)].copy()

                comm_gap = compute_proportion_stats(comm_filtered, ["committee_name", "minority"], "is_empirical_proxy")
                comm_gap["Status"] = comm_gap["minority"].map({1: "Minority", 0: "Majority"})

                # Sort committees by overall empirical percentage
                overall_pct = (
                    comm_filtered.groupby("committee_name")["is_empirical_proxy"]
                    .mean()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                sorted_committees = overall_pct["committee_name"].tolist()

                fig_comm = px.bar(
                    comm_gap,
                    y="committee_name",
                    x="mean_pct",
                    color="Status",
                    error_x="ci_95_pct",
                    barmode="group",
                    orientation="h",
                    title="Evidence Usage by Committee (Top 15 Committees by Volume, with 95% CIs)",
                    category_orders={"committee_name": sorted_committees},
                    color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                    labels={"committee_name": "Committee Name", "mean_pct": "% Empirical Discourse"},
                )
                apply_dark_theme(fig_comm, is_categorical=True)
                fig_comm.update_layout(height=500, margin=dict(l=20))
                st.plotly_chart(fig_comm, use_container_width=True)

                st.markdown(
                    """
                    > **Research Insight: Technical vs. Partisan Committees**
                    >
                    > Fact-heavy committees (e.g., related to Science or Appropriations) typically feature a higher baseline of
                    > empirical discourse, while the minority-majority gap varies significantly based on committee focus and polarization.
                    """
                )
            else:
                st.info("No valid committee names found in sample.")
        else:
            st.info("Committee name column not available in sample.")

        # Section 10: Electoral Safety vs. Empirical Discourse
        st.markdown("---")
        st.subheader("10. Electoral Safety vs. Empirical Discourse")
        if "vote_pct" in sample_df.columns:
            safety_df = sample_df[sample_df["vote_pct"].notna()].copy()
            if not safety_df.empty:

                def categorize_safety(vote):
                    if vote < 55:
                        return "Marginal (<55%)"
                    elif vote < 65:
                        return "Competitive (55-65%)"
                    else:
                        return "Safe (>65%)"

                safety_df["Safety Margin"] = safety_df["vote_pct"].apply(categorize_safety)
                safety_gap = compute_proportion_stats(safety_df, ["Safety Margin", "minority"], "is_empirical_proxy")
                safety_gap["Status"] = safety_gap["minority"].map({1: "Minority", 0: "Majority"})

                fig_safety = px.bar(
                    safety_gap,
                    x="Safety Margin",
                    y="mean_pct",
                    color="Status",
                    error_y="ci_95_pct",
                    barmode="group",
                    category_orders={"Safety Margin": ["Marginal (<55%)", "Competitive (55-65%)", "Safe (>65%)"]},
                    title="Evidence Usage by Electoral Safety and Party Power Status (with 95% CIs)",
                    color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                    labels={"Safety Margin": "Electoral Safety", "mean_pct": "% Empirical Discourse"},
                )
                apply_dark_theme(fig_safety, is_categorical=True)
                st.plotly_chart(fig_safety, use_container_width=True)

                st.markdown(
                    """
                    > **Research Insight: Electoral Margins & Justification**
                    >
                    > Lawmakers representing highly marginal or competitive seats are often under higher pressure to defend their votes
                    > and discourse using objective data, compared to their peers in extremely safe districts.
                    """
                )
            else:
                st.info("Electoral safety data (vote_pct) is empty in sample.")
        else:
            st.info("Electoral safety data column (vote_pct) not available in sample.")

    else:
        st.warning("Speaker text or minority status columns are missing in the sample data.")

    # Section 5: Technical Metadata
    st.markdown("---")
    with st.expander("Advanced Data Science Metadata"):
        st.subheader("Column Specifications & Memory")
        meta_df = pd.DataFrame(
            {
                "Dtype": global_stats["dtypes"].astype(str),
                "Null Count": global_stats["null_counts"],
                "Null %": (global_stats["null_counts"] / global_stats["total_rows"] * 100).round(2),
            }
        )
        st.table(meta_df)

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Global Row Count", f"{global_stats['total_rows']:,}")
        m_col2.metric("Metadata RAM Footprint", f"{global_stats['memory_usage']:.1f} MB")
