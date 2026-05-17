import plotly.express as px
import streamlit as st

from scripts.components.utils import metric_card


def render_overview_tab(filtered_df):
    if filtered_df.empty:
        st.warning("No data available matching current sidebar filters.")
        return

    # Metrics Row using elegant HTML Cards
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        members_count = filtered_df["bioguide_id"].nunique() if "bioguide_id" in filtered_df.columns else 0
        metric_card(
            title="Unique Members",
            value=f"{members_count:,}",
            subtext="Speakers mapped to Bio ID",
            accent_color="#2b5cff",
        )

    with m2:
        hearings_count = filtered_df["hearing_id"].nunique() if "hearing_id" in filtered_df.columns else 0
        metric_card(
            title="Unique Hearings",
            value=f"{hearings_count:,}",
            subtext="Congressional sessions",
            accent_color="#00cc96",
        )

    with m3:
        avg_match = filtered_df["match_score"].mean() if "match_score" in filtered_df.columns else 0
        match_str = f"{avg_match:.1f}%" if avg_match > 0 else "N/A"
        metric_card(
            title="Avg. Match Score",
            value=match_str,
            subtext="Speaker validation accuracy",
            accent_color="#ab63fa",
        )

    with m4:
        female_pct = filtered_df["female"].mean() * 100 if "female" in filtered_df.columns else 0
        female_str = f"{female_pct:.1f}%" if female_pct > 0 else "N/A"
        metric_card(
            title="Female Representation",
            value=female_str,
            subtext="Women in loaded sample",
            accent_color="#ff6692",
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
            hole=0.4,
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            font_family="'Outfit', sans-serif",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Ideology vs. Seniority")
        if "nominate_dim1" in filtered_df.columns and "seniority" in filtered_df.columns:
            # Sample for plot to keep it fast
            plot_df = filtered_df.dropna(subset=["nominate_dim1", "seniority"])
            if not plot_df.empty:
                plot_df = plot_df.sample(min(2000, len(plot_df)), random_state=42)
                fig = px.scatter(
                    plot_df,
                    x="nominate_dim1",
                    y="seniority",
                    color="party",
                    hover_data=["speaker"],
                    labels={"nominate_dim1": "DW-NOMINATE (Lib-Con)", "seniority": "Terms Served"},
                    color_discrete_map={"Democratic": "#2E5BFF", "Republican": "#FF4B4B"},
                )

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff",
                    font_family="'Outfit', sans-serif",
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.06)",
                        zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.15)",
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.06)",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ideology and seniority data available for selected filter range.")
        else:
            st.info("Ideology and seniority data (DW-NOMINATE) not available.")
