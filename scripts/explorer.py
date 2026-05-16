import os

import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st
from optimize_data import CSV_PATH, PARQUET_PATH, convert_csv_to_parquet

# Page configuration
st.set_page_config(
    page_title="BICAM Dataset Explorer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths are imported from optimize_data for consistency
# DATA_PATH = PARQUET_PATH


def check_and_optimize():
    """Checks if the optimized parquet file needs to be generated or updated."""
    needs_optimization = False

    if not os.path.exists(PARQUET_PATH):
        st.info("🚀 Optimized dataset not found. Preparing for first-time use...")
        needs_optimization = True
    elif os.path.exists(CSV_PATH):
        csv_mtime = os.path.getmtime(CSV_PATH)
        pq_mtime = os.path.getmtime(PARQUET_PATH)
        if csv_mtime > pq_mtime:
            st.warning("⚠️ Source CSV is newer than the optimized dataset. Updating...")
            needs_optimization = True

    if needs_optimization:
        if not os.path.exists(CSV_PATH):
            st.error(f"Error: Source file `{CSV_PATH}` not found. Cannot optimize.")
            st.stop()

        progress_bar = st.progress(0, text="Optimizing data for performance...")

        def update_progress(percent):
            progress_bar.progress(percent, text=f"Optimizing data... {int(percent*100)}%")

        success = convert_csv_to_parquet(progress_callback=update_progress)

        if success:
            progress_bar.empty()
            st.success("✅ Optimization complete! Loading data...")
            st.cache_data.clear()  # Clear cache to force reload of new data
        else:
            st.error("❌ Optimization failed. Please check the logs.")
            st.stop()


# Run the check before any data loading or indexing
check_and_optimize()


@st.cache_data
def get_dataset_info():
    """Reads only the congress column to provide indexing and stats."""
    base_dir = os.getcwd()
    full_path = os.path.join(base_dir, PARQUET_PATH)

    if not os.path.exists(full_path):
        return None, 0

    try:
        # With Parquet, reading a single column is extremely fast and memory-efficient
        df_c = pd.read_parquet(full_path, columns=["congress"])
        return df_c["congress"], len(df_c)
    except Exception as e:
        st.error(f"Error indexing dataset: {e}")
        return None, 0


@st.cache_data
def load_data(nrows=100000, sampling="Top N", selected_congresses=None):
    base_dir = os.getcwd()
    full_path = os.path.join(base_dir, PARQUET_PATH)

    if not os.path.exists(full_path):
        return None

    try:
        # Parquet filters are much more efficient than CSV skiprows
        filters = []
        if selected_congresses:
            filters.append(("congress", "in", selected_congresses))

        # Load data with filters (pyarrow engine handles this efficiently)
        df = pd.read_parquet(full_path, filters=filters if filters else None)

        # Apply sampling in memory (fast since the dataset is now compressed and filtered)
        if sampling == "Random" and len(df) > nrows:
            df = df.sample(nrows, random_state=42)
        elif len(df) > nrows:
            df = df.head(nrows)

        # Post-processing
        if "hearing_date" in df.columns:
            df["hearing_date"] = pd.to_datetime(df["hearing_date"])

        if "text" not in df.columns and "target_sentence" in df.columns:
            df = df.rename(columns={"target_sentence": "text"})

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def get_global_overview():
    """Computes high-level stats on a large sample to keep it fast and low-memory."""
    if not os.path.exists(PARQUET_PATH):
        return None

    try:
        # Get available columns from the Parquet file
        available_cols = pq.read_table(PARQUET_PATH).column_names

        # Desired metadata columns
        desired_cols = [
            "congress",
            "chamber",
            "party",
            "gender",
            "female",
            "seniority",
            "nominate_dim1",
            "match_score",
            "match_type",
            "minority",
            "unified",
        ]

        # Intersection of what we want and what we have
        cols_to_read = [c for c in desired_cols if c in available_cols]

        # Read only what's available
        df_sample = pd.read_parquet(PARQUET_PATH, columns=cols_to_read)

        stats = {
            "total_rows": len(df_sample),
            "available_cols": available_cols,
            "null_counts": df_sample.isnull().sum(),
            "dtypes": df_sample.dtypes,
            "avg_match_score": df_sample["match_score"].mean() if "match_score" in df_sample.columns else 0,
            "memory_usage": df_sample.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "sample_df": df_sample.sample(min(250000, len(df_sample))),  # Sub-sample for histograms
        }

        # Optional categorical counts
        if "party" in df_sample.columns:
            stats["party_counts"] = df_sample["party"].value_counts()
        if "chamber" in df_sample.columns:
            stats["chamber_counts"] = df_sample["chamber"].value_counts()
        if "congress" in df_sample.columns:
            stats["congress_counts"] = df_sample["congress"].value_counts().sort_index()

        return stats
    except Exception as e:
        st.error(f"Error computing global stats: {e}")
        return None


# Sidebar - Configuration
with st.sidebar:
    st.subheader("Data Loading Options")

    sampling_mode = st.radio(
        "Sampling Method", ["Top N", "Random"], index=1, help="Random sample avoids bias if the file is sorted."
    )

    row_limit = st.select_slider("Number of Rows", options=[10000, 50000, 100000, 250000, 500000], value=100000)

    # Manual Regenerate Button
    if st.button("🛠️ Force Regenerate Optimized Data", help="Manually re-run the CSV -> Parquet conversion"):
        st.info("Re-optimizing data...")
        progress_bar = st.progress(0, text="Optimizing data for performance...")

        def update_progress(percent):
            progress_bar.progress(percent, text=f"Optimizing data... {int(percent*100)}%")

        if convert_csv_to_parquet(progress_callback=update_progress):
            st.success("Optimization finished!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Failed to re-optimize.")

    st.markdown("---")

    congress_idx, total_rows = get_dataset_info()
    all_available_congress = (
        sorted(congress_idx.unique().tolist()) if congress_idx is not None else [115, 116, 117, 118]
    )

    selected_congress_load = st.multiselect(
        "Congresses to Load",
        all_available_congress,
        default=all_available_congress,
        help="Filtering here speeds up loading by skipping unused rows.",
    )

    if st.button("🔄 Refresh Data Cache"):
        st.cache_data.clear()
        st.rerun()

# Pre-compute global stats for the insights tab
global_stats = get_global_overview()

# Load the data with filters

# Load the data with filters
with st.spinner("Loading dataset..."):
    df = load_data(nrows=row_limit, sampling=sampling_mode, selected_congresses=selected_congress_load)

if df is None:
    if not os.path.exists(PARQUET_PATH):
        st.error(f"Optimized dataset not found at `{PARQUET_PATH}`.")
        st.info("Please ensure your source CSV exists and run the optimization.")
    else:
        st.error("Error loading dataset. Please check the logs.")
    st.stop()

# Sidebar - Global Filters (On the loaded sample)
st.sidebar.markdown("---")
st.sidebar.header("View Filters")

# Party filter
all_parties = sorted(df["party"].dropna().unique())
selected_parties = st.sidebar.multiselect("Party", all_parties, default=all_parties)

# Committee filter
all_committees = sorted(df["committee_name"].dropna().unique())
selected_committees = st.sidebar.multiselect("Committee", all_committees, default=[])

# Apply view filters
filtered_df = df[df["party"].isin(selected_parties)]
if selected_committees:
    filtered_df = filtered_df[filtered_df["committee_name"].isin(selected_committees)]

# Header
st.title("Congressional Discourse Dataset Explorer")
st.markdown(
    f"Currently viewing **{len(filtered_df):,}** sentences from a **{len(df):,}** row sample (Total dataset: **{total_rows:,}** rows)."
)

# Main Tabs
tabs = st.tabs(["📊 Overview", "🔍 Transcript Search", "🧪 Matching Diagnostics", "🧬 Data Science & Insights"])

with tabs[0]:
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Unique Members", filtered_df["bioguide_id"].nunique() if "bioguide_id" in filtered_df.columns else 0)
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

with tabs[1]:
    st.subheader("Search Transcripts")
    search_query = st.text_input("Enter keywords (e.g., 'climate change', 'inflation')", "")

    if search_query:
        search_results = filtered_df[filtered_df["text"].str.contains(search_query, case=False, na=False)]
        st.write(f"Found **{len(search_results):,}** sentences matching your query.")

        st.dataframe(
            search_results[["speaker", "party", "committee_name", "text", "hearing_date"]], use_container_width=True
        )
    else:
        st.info("Enter a search term above to browse specific utterances.")

with tabs[2]:
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

with tabs[3]:
    if global_stats is None:
        st.warning("Global statistics are unavailable. Please ensure the optimized dataset is generated.")
    else:
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

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data Dictionary available in `docs/project/data_dictionary.md`.")
