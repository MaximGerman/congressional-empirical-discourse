"""
End-to-end pipeline for processing BICAM congressional hearing transcripts.

Steps:
1. Load hearings data, filter to 115th-118th Congress (House only)
2. Load transcripts and inspect format
3. Process transcripts: speaker segmentation -> sentence splitting
4. Enrich with political metadata using BICAM's hearing-member mappings
5. Create a representative sample for silver labeling (~10K sentences)

Usage:
    cd <project_root>
    .venv/bin/python -m src.pipeline
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from src.data import (
    TEXT_COLUMN,
    filter_hearings_by_congress,
    load_hearings,
    load_hearings_texts_chunked,
)
from src.pipeline_enrich import step4_enrich_metadata
from src.preprocess import (
    download_nltk_deps,
    process_single_hearing,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def step1_explore_data():
    """Step 1: Load, explore, and filter hearings metadata."""
    logger.info("STEP 1: Loading and exploring hearings data")

    hearings_df = load_hearings()
    logger.info("Total hearings in dataset: %d", len(hearings_df))
    logger.info("Columns: %s", list(hearings_df.columns))
    logger.info("Congress range: %d - %d", hearings_df["congress"].min(), hearings_df["congress"].max())

    # Filter to target period (House only, 115th+)
    new_era = filter_hearings_by_congress(hearings_df, min_congress=115, chamber="house")
    logger.info("House hearings from 115th Congress onwards: %d", len(new_era))
    logger.info("Breakdown by congress:\n%s", new_era["congress"].value_counts().sort_index().to_string())

    return hearings_df, new_era


def step2_load_transcripts(new_era):
    """Step 2: Load transcripts for target hearings."""
    logger.info("STEP 2: Loading transcripts for target hearings")

    target_ids = new_era["hearing_id"].tolist()
    logger.info("Loading transcripts for %d hearings (reading ~5.4GB file in chunks)...", len(target_ids))

    texts_df = load_hearings_texts_chunked(target_ids)
    logger.info("Loaded %d transcript records", len(texts_df))
    logger.info("Columns: %s", list(texts_df.columns))

    # Check text availability
    has_text = texts_df[TEXT_COLUMN].notna() & (texts_df[TEXT_COLUMN].str.len() > 100)
    logger.info("Records with substantial text: %d / %d", has_text.sum(), len(texts_df))

    # Show a sample
    sample_row = texts_df[has_text].iloc[0]
    text = str(sample_row[TEXT_COLUMN])
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if (
            line.strip().startswith(("Chairman", "Mr.", "Mrs.", "Ms."))
            and "...." not in line
            and len(line.strip()) > 20
        ):
            start = max(0, i)
            end = min(len(lines), i + 15)
            sample_lines = "\n".join(f"  {lines[j]}" for j in range(start, end))
            logger.debug("Sample transcript (hearing_id=%s):\n%s", sample_row["hearing_id"], sample_lines)
            break

    return texts_df


def step3_process_transcripts(new_era, texts_df):
    """Step 3: Process all transcripts into sentence-level records."""
    logger.info("STEP 3: Processing transcripts into sentence-level format")

    download_nltk_deps()

    # Only process hearings with actual text
    has_text = texts_df[TEXT_COLUMN].notna() & (texts_df[TEXT_COLUMN].str.len() > 100)
    processable = texts_df[has_text]
    logger.info("Processing %d hearings with text...", len(processable))

    all_records = []
    failed_hearings = []
    empty_hearings = 0

    for _, row in tqdm(processable.iterrows(), total=len(processable), desc="Processing hearings"):
        hearing_id = row["hearing_id"]
        text = str(row[TEXT_COLUMN])

        try:
            records = process_single_hearing(hearing_id, text)
            if records:
                all_records.extend(records)
            else:
                empty_hearings += 1
        except Exception as e:
            failed_hearings.append({"hearing_id": hearing_id, "error": str(e)})

    sentences_df = pd.DataFrame(all_records)
    logger.info("Total sentences extracted: %d", len(sentences_df))
    logger.info("Hearings with no speakers found: %d", empty_hearings)
    logger.info("Failed hearings: %d", len(failed_hearings))

    if not sentences_df.empty:
        logger.info("Unique speakers: %d", sentences_df["speaker"].nunique())
        logger.info("Unique hearings with sentences: %d", sentences_df["hearing_id"].nunique())

    return sentences_df, failed_hearings


def step5_create_sample(legislators_df, sample_size=10000, seed=42):
    """Step 5: Create a representative sample for silver labeling."""
    logger.info("STEP 5: Creating representative sample for silver labeling")

    if legislators_df.empty:
        logger.warning("No data to sample from.")
        return pd.DataFrame()

    # Only sample from matched legislators
    matched = legislators_df[legislators_df["bioguide_id"].notna()].copy()
    logger.info("Matched legislator sentences available: %d", len(matched))

    if len(matched) <= sample_size:
        logger.info("Dataset smaller than target sample (%d), using all data", sample_size)
        sample = matched.copy()
    else:
        # Stratified sample by congress
        congresses = sorted(matched["congress"].unique())
        per_congress = sample_size // len(congresses)
        parts = []
        for c in congresses:
            congress_data = matched[matched["congress"] == c]
            n = min(len(congress_data), per_congress)
            parts.append(congress_data.sample(n, random_state=seed))

        sample = pd.concat(parts, ignore_index=True)

        # Top up if we didn't hit the target
        if len(sample) < sample_size:
            remaining = matched[~matched.index.isin(sample.index)]
            extra = remaining.sample(min(len(remaining), sample_size - len(sample)), random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=True)

    logger.info("Sample size: %d", len(sample))
    logger.info("Congress distribution:\n%s", sample["congress"].value_counts().sort_index().to_string())
    logger.info("Party distribution:\n%s", sample["party"].value_counts().to_string())
    logger.info("Minority distribution:\n%s", sample["minority"].value_counts().to_string())

    return sample


def run_pipeline():
    """Run the full pipeline end-to-end."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    ensure_output_dir()

    # Step 1: Explore and filter
    _hearings_df, new_era = step1_explore_data()

    # Step 2-3: Load and process transcripts (skip if raw CSV already exists)
    raw_path = os.path.join(OUTPUT_DIR, "sentences_raw.csv")
    if os.path.exists(raw_path):
        logger.info("Found existing raw sentences at %s, loading...", raw_path)
        sentences_df = pd.read_csv(raw_path)
        logger.info("Loaded %d sentences from cache", len(sentences_df))
    else:
        # Step 2: Load transcripts
        texts_df = step2_load_transcripts(new_era)

        # Step 3: Process transcripts
        sentences_df, _failed = step3_process_transcripts(new_era, texts_df)

        if sentences_df.empty:
            logger.error("No sentences extracted. Check transcript format.")
            return

        # Save raw sentences
        sentences_df.to_csv(raw_path, index=False)
        logger.info("Raw sentences saved to: %s", raw_path)

    # Step 4: Enrich with metadata
    legislators_df = step4_enrich_metadata(sentences_df, new_era)

    # Save enriched data
    enriched_path = os.path.join(OUTPUT_DIR, "sentences_enriched.parquet")
    legislators_df.to_parquet(enriched_path, index=False)
    logger.info("Enriched legislator sentences saved to Parquet: %s", enriched_path)

    # Step 5: Create sample
    sample = step5_create_sample(legislators_df)

    # Save sample
    sample_path = os.path.join(OUTPUT_DIR, "sample_for_labeling.csv")
    sample.to_csv(sample_path, index=False)
    logger.info("Silver labeling sample saved to: %s", sample_path)

    # Summary
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total hearings in target period: %d", len(new_era))
    logger.info("  Total sentences extracted: %d", len(sentences_df))
    logger.info("  Legislator sentences (filtered): %d", len(legislators_df))

    # Compute matched metrics
    matched_df = legislators_df[legislators_df["bioguide_id"].notna()]
    matched_count = len(matched_df)
    total_count = len(legislators_df)
    logger.info(
        "  Matched to member database: %d (%.1f%%)",
        matched_count,
        (matched_count / total_count * 100) if total_count else 0,
    )
    logger.info("  Silver labeling sample: %d", len(sample))
    logger.info("  Missing data checks (on matched):")

    # Check for missing covariates
    covariates_to_check = ["seniority_rs", "abs_dwnom1_rs", "vote_pct", "chairspeech", "rankmemspeech"]
    for col in covariates_to_check:
        if col in matched_df.columns:
            missing_count = matched_df[col].isna().sum()
            logger.info("    - %s: %d missing", col, missing_count)
        else:
            logger.info("    - %s: Column not found", col)

    logger.info("Output files in: %s/", OUTPUT_DIR)


if __name__ == "__main__":
    run_pipeline()
