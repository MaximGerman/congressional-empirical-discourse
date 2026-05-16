import streamlit as st


def render_search_tab(filtered_df):
    st.subheader("Search Transcripts")
    search_query = st.text_input("Enter keywords (e.g., 'climate change', 'inflation')", "")

    if search_query:
        search_results = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]
        st.write(f"Found **{len(search_results):,}** sentences matching your query.")

        st.dataframe(
            search_results[["speaker", "party", "committee_name", "text", "hearing_date"]],
            use_container_width=True,
        )
    else:
        st.info("Enter a search term above to browse specific utterances.")
