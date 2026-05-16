import plotly.express as px
import streamlit as st


def render_diagnostics_tab(filtered_df):
    st.subheader("Matching Quality & Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Unmatched Speakers")
        unmatched = filtered_df[filtered_df["bioguide_id"].isna()]
        if not unmatched.empty:
            unmatched_counts = unmatched["speaker"].value_counts().head(20).reset_index()
            unmatched_counts.columns = ["Speaker Name", "Sentence Count"]
            st.write("Top speakers unable to be matched to a Bioguide ID:")
            st.table(unmatched_counts)
        else:
            st.success("All speakers in this view are matched!")

    with col2:
        st.markdown("### Match Type Distribution")
        if "match_type" in filtered_df.columns:
            match_counts = filtered_df["match_type"].value_counts().reset_index()
            match_counts.columns = ["Strategy", "Count"]
            fig = px.bar(match_counts, x="Strategy", y="Count", color="Strategy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Match type information not available in dataset.")

    st.markdown("---")
    st.markdown("### Low Confidence Matches (< 80%)")
    if "match_score" in filtered_df.columns:
        low_conf = filtered_df[(filtered_df["match_score"] < 80) & (filtered_df["match_score"] > 0)]
        if not low_conf.empty:
            st.dataframe(
                low_conf[["speaker", "member_first_name", "speaker_last_name", "match_score", "match_type"]]
                .drop_duplicates()
                .head(50),
                use_container_width=True,
            )
        else:
            st.success("No low-confidence matches found in this sample.")
