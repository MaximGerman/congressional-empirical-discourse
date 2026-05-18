import re

import streamlit as st

from scripts.components.utils import highlight_search_terms


def render_search_tab(filtered_df):
    if filtered_df.empty:
        st.warning("No data available to search.")
        return

    st.subheader("Search Transcripts & Dialogue Context")
    search_query = st.text_input(
        "Enter keywords (e.g., 'climate change', 'inflation', 'clean energy')",
        value="",
        key="search_query_input",
    )

    # Sub-columns for advanced search filters
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        case_sensitive = st.checkbox(
            "Case Sensitive Search",
            value=False,
            help="If checked, matches case exactly (e.g. 'FDA' won't match 'fda').",
            key="search_case_sensitive",
        )
    with col_opt2:
        whole_words = st.checkbox(
            "Match Whole Words Only",
            value=False,
            help="If checked, matches query on word boundaries (e.g. 'tax' won't match 'taxes').",
            key="search_whole_words",
        )
    with col_opt3:
        use_regex = st.checkbox(
            "Use Regular Expressions",
            value=False,
            help="If checked, treats the query as a python regular expression (e.g. 'climate.*change').",
            key="search_use_regex",
        )

    if search_query:
        valid_query = True

        # Validate regular expression syntax if enabled
        if use_regex:
            try:
                re.compile(search_query)
            except re.error as e:
                st.error(f"Invalid Regular Expression syntax: {e}")
                valid_query = False

        if valid_query:
            # Apply matching filters
            if use_regex:
                search_results = filtered_df[
                    filtered_df["text"].str.contains(search_query, case=case_sensitive, regex=True, na=False)
                ]
            else:
                if whole_words:
                    # Smart boundary: Only append \b if the boundary edge is a word character
                    start_b = r"\b" if re.match(r"^\w", search_query) else ""
                    end_b = r"\b" if re.search(r"\w$", search_query) else ""
                    pattern = f"{start_b}{re.escape(search_query)}{end_b}"
                    search_results = filtered_df[
                        filtered_df["text"].str.contains(pattern, case=case_sensitive, regex=True, na=False)
                    ]
                else:
                    search_results = filtered_df[
                        filtered_df["text"].str.contains(search_query, case=case_sensitive, regex=False, na=False)
                    ]

            st.write(f"Found **{len(search_results):,}** sentences matching your query.")

            if not search_results.empty:
                # Columns to display
                display_cols = ["speaker", "party", "committee_name", "text", "hearing_date"]
                # Filter columns that are present in the dataset
                cols_to_show = [c for c in display_cols if c in search_results.columns]

                st.markdown("### Sentence Explorer")

                # Limit the displayed rows in the interactive table to avoid browser lag
                max_display = 1000
                display_df = search_results[cols_to_show]
                if len(display_df) > max_display:
                    st.warning(
                        f"Showing the first {max_display:,} matching sentences to maintain dashboard performance."
                    )
                    display_df = display_df.head(max_display)
                else:
                    st.caption("Tip: Select a row in the table below to inspect its surrounding dialogue context.")

                # Interactive selection dataframe
                event = st.dataframe(
                    display_df,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="search_results_select_df",
                )

                # Handle row selection
                selected_rows = event.selection.rows if hasattr(event, "selection") and event.selection else []

                if selected_rows:
                    row_idx = selected_rows[0]
                    selected_row = search_results.iloc[row_idx]

                    st.markdown("---")
                    st.markdown("### Surrounding Speech Flow Context")

                    # Fetch dialogue variables
                    speaker = selected_row.get("speaker", "Unknown Speaker")
                    party = selected_row.get("party", "Unknown Party")
                    date_val = selected_row.get("hearing_date")
                    date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
                    committee = selected_row.get("committee_name", "Unknown Committee")

                    context_before = selected_row.get("context_before", None)
                    context_after = selected_row.get("context_after", None)

                    before_str = (
                        context_before
                        if isinstance(context_before, str) and context_before.strip()
                        else "*(No preceding context)*"
                    )
                    after_str = (
                        context_after
                        if isinstance(context_after, str) and context_after.strip()
                        else "*(No succeeding context)*"
                    )

                    # Highlight search terms in the displayed context blocks
                    before_highlighted = highlight_search_terms(
                        before_str,
                        search_query,
                        case_sensitive=case_sensitive,
                        whole_words=whole_words,
                        is_regex=use_regex,
                    )
                    target_highlighted = highlight_search_terms(
                        selected_row.get("text", ""),
                        search_query,
                        case_sensitive=case_sensitive,
                        whole_words=whole_words,
                        is_regex=use_regex,
                    )
                    after_highlighted = highlight_search_terms(
                        after_str,
                        search_query,
                        case_sensitive=case_sensitive,
                        whole_words=whole_words,
                        is_regex=use_regex,
                    )

                    # Color code based on party
                    party_color = (
                        "#2E5BFF" if party == "Democratic" else ("#FF4B4B" if party == "Republican" else "#ffc107")
                    )

                    html_content = (
                        f'<div style="background: rgba(22, 28, 45, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 25px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 18px;">'
                        f'<span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">Speaker: {speaker} <span style="font-size: 0.9rem; font-weight: 600; color: {party_color}; margin-left: 6px;">({party})</span></span>'
                        f'<span style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.5); font-weight: 500;">Hearing Date: {date_str}</span>'
                        f"</div>"
                        f'<div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.4); margin-bottom: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Committee: {committee}</div>'
                        f'<div style="color: rgba(255, 255, 255, 0.5); font-style: italic; margin-bottom: 15px; font-size: 0.95rem; line-height: 1.5; padding: 0 5px;"><strong style="color: rgba(255,255,255,0.7); font-style: normal; font-size: 0.8rem; text-transform: uppercase;">[Preceding Context]:</strong><br>{before_highlighted}</div>'
                        f'<div style="background: rgba(43, 92, 255, 0.12); border-left: 4px solid #2b5cff; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-size: 1.05rem; font-weight: 500; margin-bottom: 15px; line-height: 1.5; box-shadow: 0 4px 15px rgba(43, 92, 255, 0.08);"><strong style="color: #2b5cff; font-size: 0.8rem; text-transform: uppercase; font-weight: 700;">[Target Sentence]:</strong><br>{target_highlighted}</div>'
                        f'<div style="color: rgba(255, 255, 255, 0.5); font-style: italic; font-size: 0.95rem; line-height: 1.5; padding: 0 5px;"><strong style="color: rgba(255,255,255,0.7); font-style: normal; font-size: 0.8rem; text-transform: uppercase;">[Succeeding Context]:</strong><br>{after_highlighted}</div>'
                        f"</div>"
                    )
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.info(
                        "Click any row in the search results table above to open the dialogue flow inspector and read the surrounding context."
                    )
            else:
                st.info("No records found matching your keyword. Try another term.")
    else:
        st.info("Enter a search term above to browse specific transcripts and explore target speech contexts.")
