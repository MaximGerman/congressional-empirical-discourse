import os

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from scripts.components.diagnostics_tab import render_diagnostics_tab
from scripts.components.insights_tab import render_insights_tab
from scripts.components.overview_tab import render_overview_tab
from scripts.components.search_tab import render_search_tab
from scripts.optimize_data import CSV_PATH, PARQUET_PATH, convert_csv_to_parquet

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
    render_overview_tab(filtered_df)

with tabs[1]:
    render_search_tab(filtered_df)

with tabs[2]:
    render_diagnostics_tab(filtered_df)

with tabs[3]:
    render_insights_tab(global_stats)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data Dictionary available in `docs/project/data_dictionary.md`.")
