import os
import sys

# Ensure the project root is in sys.path to resolve cross-module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium styling and font loading
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0c0f17;
    }

    .gradient-text {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glassmorphic custom card widgets */
    .glass-card {
        background: rgba(22, 28, 45, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }

    /* Custom style for standard metric containers */
    .stMetric {
        background: rgba(30, 36, 56, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
    }

    .stMetric:hover {
        border-color: rgba(255, 255, 255, 0.15) !important;
        background: rgba(35, 42, 65, 0.55) !important;
    }

    /* Elegant Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #090b11 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Styled Subsections */
    .section-header {
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #ffffff;
        border-bottom: 2px solid #2b5cff;
        padding-bottom: 8px;
        margin-bottom: 15px;
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
        st.info("Optimized dataset not found. Preparing for first-time use...")
        needs_optimization = True
    elif os.path.exists(CSV_PATH):
        csv_mtime = os.path.getmtime(CSV_PATH)
        pq_mtime = os.path.getmtime(PARQUET_PATH)
        if csv_mtime > pq_mtime:
            st.warning("Source CSV is newer than the optimized dataset. Updating...")
            needs_optimization = True

    if needs_optimization:
        if not os.path.exists(CSV_PATH):
            st.error(f"Error: Source file `{CSV_PATH}` not found. Cannot optimize.")
            st.stop()

        progress_bar = st.progress(0, text="Optimizing data for performance...")

        def update_progress(percent):
            progress_bar.progress(percent, text=f"Optimizing data... {int(percent * 100)}%")

        success = convert_csv_to_parquet(progress_callback=update_progress)

        if success:
            progress_bar.empty()
            st.success("Optimization complete! Loading data...")
            st.cache_data.clear()  # Clear cache to force reload of new data
        else:
            st.error("Optimization failed. Please check the logs.")
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
        # Get available columns from the Parquet file metadata
        parquet_file = pq.ParquetFile(PARQUET_PATH)
        available_cols = parquet_file.schema.names
        total_rows = parquet_file.metadata.num_rows

        if total_rows == 0:
            return {
                "total_rows": 0,
                "available_cols": available_cols,
                "null_counts": pd.Series(0, index=available_cols),
                "dtypes": pd.Series(object, index=available_cols),
                "avg_match_score": 0,
                "memory_usage": 0,
                "sample_df": pd.DataFrame(columns=available_cols),
            }

        # 1. Read metadata columns for ALL rows (efficient)
        metadata_cols = [
            "congress",
            "chamber",
            "party",
            "gender",
            "female",
            "seniority",
            "nominate_dim1",
            "abs_dwnom1",
            "match_score",
            "match_type",
            "minority",
            "unified",
        ]
        cols_to_read = [c for c in metadata_cols if c in available_cols]
        df_meta = pd.read_parquet(PARQUET_PATH, columns=cols_to_read) if cols_to_read else pd.DataFrame()

        # 2. Sample rows for the heavy 'text' analysis using stratified row-group sampling (very low memory)
        sample_size = min(250000, total_rows)

        # Determine the text column name in the Parquet file
        text_col = (
            "target_sentence" if "target_sentence" in available_cols else ("text" if "text" in available_cols else None)
        )

        # 3. Read content and covariate columns for the sample
        content_cols = [
            "congress",
            "minority",
            "abs_dwnom1",
            "nominate_dim1",
            "match_score",
            "committee_name",
            "chamber",
            "vote_pct",
            "party",
            "state_abbrev",
            "chairspeech",
            "rankmemspeech",
            "freshman",
            "seniority",
            "speaker",
            "context_before",
            "context_after",
        ]
        if text_col:
            content_cols.append(text_col)

        content_cols_to_read = [c for c in content_cols if c in available_cols]

        if content_cols_to_read:
            num_row_groups = parquet_file.num_row_groups
            rows_per_group = max(5000, sample_size // num_row_groups)

            sample_dfs = []
            for rg in range(num_row_groups):
                try:
                    df_rg = parquet_file.read_row_group(rg, columns=content_cols_to_read).to_pandas()
                    if not df_rg.empty:
                        sample_dfs.append(df_rg.sample(min(rows_per_group, len(df_rg)), random_state=42))
                except Exception:
                    continue

            if sample_dfs:
                sample_df = pd.concat(sample_dfs, ignore_index=True)
                # Rename target_sentence to text for compatibility with the insights tab
                if "target_sentence" in sample_df.columns:
                    sample_df = sample_df.rename(columns={"target_sentence": "text"})
                if len(sample_df) > sample_size:
                    sample_df = sample_df.sample(sample_size, random_state=42)
            else:
                sample_df = pd.DataFrame(columns=content_cols_to_read)
        else:
            sample_df = (
                df_meta.sample(min(100000, len(df_meta)), random_state=42) if not df_meta.empty else pd.DataFrame()
            )

        stats = {
            "total_rows": total_rows,
            "available_cols": available_cols,
            "null_counts": df_meta.isnull().sum() if not df_meta.empty else pd.Series(),
            "dtypes": df_meta.dtypes if not df_meta.empty else pd.Series(),
            "avg_match_score": df_meta["match_score"].mean() if "match_score" in df_meta.columns else 0,
            "memory_usage": df_meta.memory_usage(deep=True).sum() / (1024 * 1024) if not df_meta.empty else 0,
            "sample_df": sample_df,
        }

        # Optional categorical counts
        if "party" in df_meta.columns:
            stats["party_counts"] = df_meta["party"].value_counts()
        if "chamber" in df_meta.columns:
            stats["chamber_counts"] = df_meta["chamber"].value_counts()
        if "congress" in df_meta.columns:
            stats["congress_counts"] = df_meta["congress"].value_counts().sort_index()

        return stats
    except Exception as e:
        st.error(f"Error computing global stats: {e}")
        import traceback

        st.code(traceback.format_exc())
        return None


# Sidebar - Configuration
with st.sidebar:
    st.subheader("Data Loading Options")

    sampling_mode = st.radio(
        "Sampling Method", ["Top N", "Random"], index=1, help="Random sample avoids bias if the file is sorted."
    )

    row_limit = st.select_slider("Number of Rows", options=[10000, 50000, 100000, 250000, 500000], value=100000)

    # Manual Regenerate Button
    if st.button("Force Regenerate Optimized Data", help="Manually re-run the CSV -> Parquet conversion"):
        st.info("Re-optimizing data...")
        progress_bar = st.progress(0, text="Optimizing data for performance...")

        def update_progress(percent):
            progress_bar.progress(percent, text=f"Optimizing data... {int(percent * 100)}%")

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

    if st.button("Refresh Data Cache"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    with st.expander("Empirical Heuristics", expanded=True):
        keyword_presets = {
            "Empirical Core (Paper Baseline)": "data, evidence, statistics, statistical, study, report, research, analysis, percent, %, number, increase, decrease, caused, correlation, fact, empirical",
            "Hard Sciences & Research": "data, evidence, statistics, statistical, study, report, research, analysis, percent, %, scientific, study, publication, peer-reviewed, laboratory",
            "Economics & Finance": "deficit, inflation, revenue, budget, cost, gdp, dollars, billion, million, tax, interest, employment, trade, gross, market",
            "Causality & Evidence": "caused, correlation, effect, because, result, consequence, fact, empirical, prove, demonstration, findings, variable, experimental",
            "Custom (Edit below)": "",
        }

        selected_preset = st.selectbox(
            "Keyword Preset Library",
            options=list(keyword_presets.keys()),
            index=0,
            help="Choose a pre-defined set of research keywords or build a custom one.",
        )

        default_keywords = keyword_presets[selected_preset]

        # If user chooses custom, we don't overwrite if they already typed something
        if "custom_keywords_input" not in st.session_state:
            st.session_state.custom_keywords_input = default_keywords

        if selected_preset != "Custom (Edit below)":
            st.session_state.custom_keywords_input = default_keywords

        keywords_input = st.text_area(
            "Keywords (comma-separated)",
            value=st.session_state.custom_keywords_input,
            help="Keywords to classify a sentence as empirical discourse.",
            height=120,
            key="keywords_text_area",
        )

        # Sync session state
        st.session_state.custom_keywords_input = keywords_input
        empirical_keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

        whole_words_toggle = st.checkbox(
            "Match Whole Words Only",
            value=True,
            help="If checked, matches keywords only on exact word boundaries to avoid false positives (e.g. 'fact' matching 'factory').",
        )

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

# Header - Premium styled gradient title block
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <h1 class="gradient-text" style="font-size: 2.8rem; margin-bottom: 5px; font-weight: 800; letter-spacing: -0.5px;">
            BICAM Dataset Explorer
        </h1>
        <p style="font-size: 1.05rem; color: rgba(255, 255, 255, 0.65); font-weight: 400; margin-top: 0;">
            Temporal Robustness and Power/Knowledge Analysis of U.S. Congressional Discourse (1997-2025)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"Currently viewing **{len(filtered_df):,}** sentences from a **{len(df):,}** row sample (Total dataset: **{total_rows:,}** rows)."
)

# Main Tabs
tabs = st.tabs(["Overview", "Transcript Search", "Matching Diagnostics", "Data Science & Insights"])

with tabs[0]:
    render_overview_tab(filtered_df)

with tabs[1]:
    render_search_tab(filtered_df)

with tabs[2]:
    render_diagnostics_tab(filtered_df)

with tabs[3]:
    render_insights_tab(global_stats, empirical_keywords=empirical_keywords, whole_words=whole_words_toggle)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data Dictionary available in docs/project/data_dictionary.md.")
