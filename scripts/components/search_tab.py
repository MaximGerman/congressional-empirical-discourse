import streamlit as st


def render_search_tab(filtered_df):
    if filtered_df.empty:
        st.warning("No data available to search.")
        return

    st.subheader("Search Transcripts & Dialogue Context")
    search_query = st.text_input("Enter keywords (e.g., 'climate change', 'inflation', 'clean energy')", "")

    if search_query:
        # Filter matching rows
        search_results = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]
        st.write(f"Found **{len(search_results):,}** sentences matching your query.")

        if not search_results.empty:
            # Columns to display
            display_cols = ["speaker", "party", "committee_name", "text", "hearing_date"]
            # Filter columns that are present in the dataset
            cols_to_show = [c for c in display_cols if c in search_results.columns]

            st.markdown("### Sentence Explorer")
            st.caption("Tip: Select a row in the table below to inspect its surrounding dialogue context.")

            # Interactive selection dataframe
            event = st.dataframe(
                search_results[cols_to_show],
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

                # Color code based on party
                party_color = (
                    "#2E5BFF" if party == "Democratic" else ("#FF4B4B" if party == "Republican" else "#ffc107")
                )

                st.markdown(
                    f"""
                    <div style="
                        background: rgba(22, 28, 45, 0.5);
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 12px;
                        padding: 25px;
                        backdrop-filter: blur(12px);
                        -webkit-backdrop-filter: blur(12px);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 18px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                                Speaker: {speaker}
                                <span style="font-size: 0.9rem; font-weight: 600; color: {party_color}; margin-left: 6px;">({party})</span>
                            </span>
                            <span style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.5); font-weight: 500;">
                                Hearing Date: {date_str}
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.4); margin-bottom: 15px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">
                            Committee: {committee}
                        </div>

                        <div style="color: rgba(255, 255, 255, 0.5); font-style: italic; margin-bottom: 15px; font-size: 0.95rem; line-height: 1.5; padding: 0 5px;">
                            <strong style="color: rgba(255,255,255,0.7); font-style: normal; font-size: 0.8rem; text-transform: uppercase;">[Preceding Context]:</strong><br>
                            {before_str}
                        </div>

                        <div style="
                            background: rgba(43, 92, 255, 0.12);
                            border-left: 4px solid #2b5cff;
                            border-radius: 6px;
                            padding: 15px 20px;
                            color: #ffffff;
                            font-size: 1.05rem;
                            font-weight: 500;
                            margin-bottom: 15px;
                            line-height: 1.5;
                            box-shadow: 0 4px 15px rgba(43, 92, 255, 0.08);
                        ">
                            <strong style="color: #2b5cff; font-size: 0.8rem; text-transform: uppercase; font-weight: 700;">[Target Sentence]:</strong><br>
                            {selected_row.get("text", "")}
                        </div>

                        <div style="color: rgba(255, 255, 255, 0.5); font-style: italic; font-size: 0.95rem; line-height: 1.5; padding: 0 5px;">
                            <strong style="color: rgba(255,255,255,0.7); font-style: normal; font-size: 0.8rem; text-transform: uppercase;">[Succeeding Context]:</strong><br>
                            {after_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "Click any row in the search results table above to open the dialogue flow inspector and read the surrounding context."
                )
        else:
            st.info("No records found matching your keyword. Try another term.")
    else:
        st.info("Enter a search term above to browse specific transcripts and explore target speech contexts.")
