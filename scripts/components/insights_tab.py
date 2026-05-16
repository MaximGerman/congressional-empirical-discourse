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
    sample_df = global_stats["sample_df"]

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

    # Section 4: Technical Metadata
    st.markdown("---")
    with st.expander("🛠️ Advanced Data Science Metadata"):
        st.subheader("Column Specifications & Memory")
        meta_df = pd.DataFrame(
            {
                "Dtype": global_stats["dtypes"],
                "Null Count": global_stats["null_counts"],
                "Null %": (global_stats["null_counts"] / global_stats["total_rows"] * 100).round(2),
            }
        )
        st.table(meta_df)

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Global Row Count", f"{global_stats['total_rows']:,}")
        m_col2.metric("Metadata RAM Footprint", f"{global_stats['memory_usage']:.1f} MB")
