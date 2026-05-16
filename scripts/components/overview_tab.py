import plotly.express as px
import streamlit as st


def render_overview_tab(filtered_df):
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Unique Members",
            filtered_df["bioguide_id"].nunique() if "bioguide_id" in filtered_df.columns else 0,
        )
    with m2:
        st.metric("Unique Hearings", filtered_df["hearing_id"].nunique())
    with m3:
        st.metric(
            "Avg. Match Score",
            f"{filtered_df['match_score'].mean():.1f}%" if "match_score" in filtered_df.columns else "N/A",
        )
    with m4:
        st.metric(
            "Female Representation",
            f"{(filtered_df['female'].mean() * 100):.1f}%" if "female" in filtered_df.columns else "N/A",
        )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Party Distribution")
        party_counts = filtered_df["party"].value_counts().reset_index()
        party_counts.columns = ["Party", "Sentences"]
        fig = px.pie(
            party_counts,
            values="Sentences",
            names="Party",
            color="Party",
            color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Ideology vs. Seniority")
        if "nominate_dim1" in filtered_df.columns and "seniority" in filtered_df.columns:
            # Sample for plot to keep it fast
            plot_df = filtered_df.dropna(subset=["nominate_dim1", "seniority"]).sample(min(2000, len(filtered_df)))
            fig = px.scatter(
                plot_df,
                x="nominate_dim1",
                y="seniority",
                color="party",
                hover_data=["speaker"],
                labels={"nominate_dim1": "DW-NOMINATE (Lib-Con)", "seniority": "Terms Served"},
                color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
            )
            st.plotly_chart(fig, use_container_width=True)
