import pandas as pd
import plotly.express as px
import streamlit as st


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


def render_insights_tab(global_stats, empirical_keywords=None):
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
        # Vectorized check for keywords
        pattern = "|".join(empirical_keywords)
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

        # Section 4: The Minority Gap (Heuristic Proxy)
        st.markdown("---")
        st.subheader("5. The 'Minority Gap' Trend (Custom Keyword Heuristic)")
        st.info(
            "Note: Since the full RoBERTa/DeBERTa classifier is still being fine-tuned for this period, "
            "we use a keyword-based heuristic to approximate empirical discourse frequency. "
            "You can customize these keywords in the sidebar!"
        )

        # Group by congress and minority status
        gap_df = sample_df.groupby(["congress", "minority"], observed=False)["is_empirical_proxy"].mean().reset_index()
        gap_df["Empirical %"] = gap_df["is_empirical_proxy"] * 100
        gap_df["Status"] = gap_df["minority"].map({1: "Minority", 0: "Majority"})

        fig_gap = px.line(
            gap_df,
            x="congress",
            y="Empirical %",
            color="Status",
            markers=True,
            title="Empirical Discourse: Majority vs. Minority (115th - 118th)",
            color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
            labels={"congress": "Congress", "Empirical %": "% Empirical Discourse"},
        )
        apply_dark_theme(fig_gap)
        fig_gap.update_layout(
            hovermode="x unified",
            yaxis_range=[0, gap_df["Empirical %"].max() * 1.2],
        )
        st.plotly_chart(fig_gap, use_container_width=True)

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
            role_gap = sample_df.groupby("legislative_role", observed=False)["is_empirical_proxy"].mean().reset_index()
            role_gap["Empirical %"] = role_gap["is_empirical_proxy"] * 100

            fig_role = px.bar(
                role_gap,
                x="legislative_role",
                y="Empirical %",
                color="legislative_role",
                title="Evidence Usage by Committee Leadership Role",
                color_discrete_map={
                    "Committee Chair (Majority Lead)": "#FF4B4B",
                    "Ranking Member (Minority Lead)": "#FFD700",
                    "Regular Member": "#808080",
                },
                labels={"legislative_role": "Legislative Role", "Empirical %": "% Empirical Discourse"},
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
                fresh_gap = sample_df.groupby("freshman", observed=False)["is_empirical_proxy"].mean().reset_index()
                fresh_gap["Empirical %"] = fresh_gap["is_empirical_proxy"] * 100
                fresh_gap["Career Stage"] = fresh_gap["freshman"].map({1: "Freshman (1st Term)", 0: "Veteran Member"})

                fig_fresh = px.bar(
                    fresh_gap,
                    x="Career Stage",
                    y="Empirical %",
                    color="Career Stage",
                    title="Evidence Usage: Freshmen vs. Veteran Lawmakers",
                    color_discrete_map={"Freshman (1st Term)": "#00cc96", "Veteran Member": "#ab63fa"},
                    labels={"Empirical %": "% Empirical Discourse"},
                )
                apply_dark_theme(fig_fresh, is_categorical=True)
                st.plotly_chart(fig_fresh, use_container_width=True)
            else:
                st.info("Freshman indicator not present in sample.")

        with c_fresh2:
            if "seniority" in sample_df.columns:
                sen_gap = sample_df.groupby("seniority", observed=False)["is_empirical_proxy"].mean().reset_index()
                sen_gap["Empirical %"] = sen_gap["is_empirical_proxy"] * 100

                fig_sen = px.line(
                    sen_gap,
                    x="seniority",
                    y="Empirical %",
                    markers=True,
                    title="Evidence Usage Trend by Seniority (Terms Served)",
                    labels={"seniority": "Seniority (Terms Served)", "Empirical %": "% Empirical Discourse"},
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
            # Create bins for ideological extremity
            sample_df["extremity_bin"] = pd.cut(sample_df["abs_dwnom1"], bins=5)
            pol_df = sample_df.groupby("extremity_bin", observed=False)["is_empirical_proxy"].mean().reset_index()
            pol_df["Empirical %"] = pol_df["is_empirical_proxy"] * 100
            pol_df["Extremity"] = pol_df["extremity_bin"].astype(str)

            fig_pol = px.bar(
                pol_df,
                x="Extremity",
                y="Empirical %",
                title="Does Ideological Extremity Correlate with Data Usage?",
                color="Empirical %",
                color_continuous_scale="Viridis",
                labels={
                    "Extremity": "Ideological Extremity (Absolute DW-NOMINATE)",
                    "Empirical %": "% Empirical Discourse",
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
            chamber_gap = (
                sample_df.groupby(["Chamber", "minority"], observed=False)["is_empirical_proxy"].mean().reset_index()
            )
            chamber_gap["Empirical %"] = chamber_gap["is_empirical_proxy"] * 100
            chamber_gap["Status"] = chamber_gap["minority"].map({1: "Minority", 0: "Majority"})

            fig_chamber = px.bar(
                chamber_gap,
                x="Chamber",
                y="Empirical %",
                color="Status",
                barmode="group",
                title="Evidence Usage by Chamber and Party Power Status",
                color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                labels={"Chamber": "Chamber", "Empirical %": "% Empirical Discourse"},
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

        # Section 8: The 2019 House Power Shift (Natural Experiment)
        st.markdown("---")
        st.subheader("8. Natural Experiment: The 2019 House Power Shift")

        # Filter to House, major parties, and 115th vs 116th Congress
        house_shift_df = sample_df[
            (sample_df["chamber"].str.lower() == "house")
            & (sample_df["party"].isin(["Democratic", "Republican"]))
            & (sample_df["congress"].isin([115, 116]))
        ].copy()

        if not house_shift_df.empty:
            shift_grouped = (
                house_shift_df.groupby(["congress", "party"], observed=False)["is_empirical_proxy"].mean().reset_index()
            )
            shift_grouped["Empirical %"] = shift_grouped["is_empirical_proxy"] * 100
            shift_grouped["Congress Label"] = shift_grouped["congress"].map(
                {115: "115th (GOP Majority)", 116: "116th (Dem Majority)"}
            )

            fig_shift = px.line(
                shift_grouped,
                x="Congress Label",
                y="Empirical %",
                color="party",
                markers=True,
                title="Evidence Usage Across the 2019 House Power Shift",
                color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
                labels={
                    "party": "Party",
                    "Empirical %": "% Empirical Discourse",
                    "Congress Label": "Congressional Session",
                },
            )
            apply_dark_theme(fig_shift)
            st.plotly_chart(fig_shift, use_container_width=True)

            st.markdown(
                """
                > **Research Insight: The 2019 Power Shift**
                >
                > In 2019 (116th Congress), Democrats gained the House majority. According to the strategic framework,
                > newly empowered Democrats should show a *decrease* in empirical discourse frequency, while newly disempowered
                > Republicans should display a corresponding *increase*.
                """
            )
        else:
            st.info("Insufficient data for the 115th/116th House sessions in this sample.")

        # Section 9: Committee-Level Heterogeneity
        st.markdown("---")
        st.subheader("9. Committee-Level Heterogeneity")
        if "committee_name" in sample_df.columns:
            comm_df = sample_df[sample_df["committee_name"].notna() & (sample_df["committee_name"] != "")].copy()

            if not comm_df.empty:
                top_committees = comm_df["committee_name"].value_counts().head(15).index
                comm_filtered = comm_df[comm_df["committee_name"].isin(top_committees)].copy()

                comm_gap = (
                    comm_filtered.groupby(["committee_name", "minority"], observed=False)["is_empirical_proxy"]
                    .mean()
                    .reset_index()
                )
                comm_gap["Empirical %"] = comm_gap["is_empirical_proxy"] * 100
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
                    x="Empirical %",
                    color="Status",
                    barmode="group",
                    orientation="h",
                    title="Evidence Usage by Committee (Top 15 Committees by Volume)",
                    category_orders={"committee_name": sorted_committees},
                    color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                    labels={"committee_name": "Committee Name", "Empirical %": "% Empirical Discourse"},
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

                safety_gap = (
                    safety_df.groupby(["Safety Margin", "minority"], observed=False)["is_empirical_proxy"]
                    .mean()
                    .reset_index()
                )
                safety_gap["Empirical %"] = safety_gap["is_empirical_proxy"] * 100
                safety_gap["Status"] = safety_gap["minority"].map({1: "Minority", 0: "Majority"})

                fig_safety = px.bar(
                    safety_gap,
                    x="Safety Margin",
                    y="Empirical %",
                    color="Status",
                    barmode="group",
                    category_orders={"Safety Margin": ["Marginal (<55%)", "Competitive (55-65%)", "Safe (>65%)"]},
                    title="Evidence Usage by Electoral Safety and Party Power Status",
                    color_discrete_map={"Minority": "#FFD700", "Majority": "#808080"},
                    labels={"Safety Margin": "Electoral Safety", "Empirical %": "% Empirical Discourse"},
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
