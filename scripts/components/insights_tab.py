import pandas as pd
import plotly.express as px
import streamlit as st


def render_insights_tab(global_stats):
    if global_stats is None:
        st.warning("Global statistics are unavailable. Please ensure the optimized dataset is generated.")
        return

    st.header("Global Dataset Insights")
    st.markdown(
        "This tab provides a statistical summary of the **entire** dataset (3.5M+ rows) using efficient metadata analysis and sampling."
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
            st.plotly_chart(fig_cham, use_container_width=True)
        else:
            st.info("Chamber data not available.")

    # Section 3: Distribution Analysis (Sampled)
    st.markdown("---")
    st.subheader("4. Scientific Distributions (250k Sample)")
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
            st.plotly_chart(fig_match, use_container_width=True)
        else:
            st.info("Match score data not available.")

    # Section 4: The Minority Gap (Heuristic Proxy)
    st.markdown("---")
    st.subheader("5. The 'Minority Gap' Trend (Keyword Proxy)")
    st.info(
        "💡 **Note:** Since the full RoBERTa classifier is still being fine-tuned for this period, "
        "we use a keyword-based heuristic (*'data', 'evidence', 'statistics', 'percent', etc.*) "
        "to approximate empirical discourse frequency."
    )

    # Keywords for empirical discourse proxy
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

    # Calculate empirical proxy on the sample
    if "text" in sample_df.columns and "minority" in sample_df.columns:
        # Vectorized check for keywords
        pattern = "|".join(empirical_keywords)
        sample_df["is_empirical_proxy"] = sample_df["text"].str.contains(pattern, case=False, na=False)

        # Group by congress and minority status
        gap_df = sample_df.groupby(["congress", "minority"])["is_empirical_proxy"].mean().reset_index()
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

        fig_gap.update_layout(
            hovermode="x unified",
            yaxis_range=[0, gap_df["Empirical %"].max() * 1.2],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_gap, use_container_width=True)

        st.markdown(
            """
            > **Research Insight: The 'Power or Knowledge' Framework**
            >
            > The original Haim & Barak-Corren (2026) paper posits that empirical discourse is the **'weapon of the powerless.'**
            > When a party loses formal agenda control (Majority → Minority), they increase their reliance on data-driven
            > arguments to challenge dominant forces. This chart allows us to monitor if this gap persists into the 2017-2025 era.
            """
        )

        # New: Polarization vs Empirical
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
            st.plotly_chart(fig_pol, use_container_width=True)
        else:
            st.info("Ideological extremity data (abs_dwnom1) not available in sample.")
    else:
        st.warning("Speaker text or minority status columns are missing in the sample data.")

    # Section 5: Technical Metadata
    st.markdown("---")
    with st.expander("🛠️ Advanced Data Science Metadata"):
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
