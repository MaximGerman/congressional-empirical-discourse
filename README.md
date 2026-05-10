# Congressional Empirical Discourse — BICAM Data Pipeline

Data pipeline for processing U.S. congressional hearing transcripts from the [BICAM dataset](https://bicam.net/) into sentence-level records enriched with political metadata. Part of the workshop project *"Temporal Robustness of Empirical Discourse Classification in U.S. Congressional Hearings (1997-2025)"*.

## Project Structure

```text
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data.py                # Data loading, filtering, member lookup
│   ├── preprocess.py          # Speaker segmentation, sentence splitting, name matching
│   └── pipeline.py            # End-to-end pipeline (Steps 1-5)
├── notebooks/
│   └── 01_explore_bicam.ipynb # Interactive data exploration notebook
└── data/                      # Pipeline output (gitignored, ~5GB)
    ├── sentences_raw.csv      # All extracted sentences
    ├── sentences_enriched.csv # Legislator sentences with metadata
    └── sample_for_labeling.csv# 10K balanced sample for silver labeling
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK tokenizer data
python -c "import nltk; nltk.download('punkt_tab')"
```

## Running the Pipeline

The pipeline downloads BICAM data (~1.8GB hearings + members), processes transcripts, and outputs sentence-level CSVs:

```bash
python -m src.pipeline
```

This runs 5 steps:

1. **Load & filter** — Loads hearings metadata, filters to 115th-118th Congress (House only)
2. **Load transcripts** — Reads the 5.4GB transcript file in chunks for target hearings
3. **Process transcripts** — Speaker segmentation + sentence splitting with context windows
4. **Enrich metadata** — Matches speakers to members database, adds party/majority status
5. **Create sample** — Stratified 10K sample for silver labeling

Steps 2-3 are cached: if `data/sentences_raw.csv` exists, the pipeline skips straight to Step 4.

### Expected Output

| File | Size | Rows | Description |
|------|------|------|-------------|
| `sentences_raw.csv` | ~2.0 GB | ~6.08M | All sentences from all speakers |
| `sentences_enriched.csv` | ~2.7 GB | ~5.50M | Legislator sentences with metadata |
| `sample_for_labeling.csv` | ~5 MB | 10,000 | Balanced sample (2,500 per congress) |

### Key Columns in Enriched Output

| Column | Description |
|--------|-------------|
| `hearing_id` | BICAM hearing identifier |
| `speaker` | Speaker attribution from transcript (e.g. "Mr. Polis") |
| `context_before` | Preceding sentence |
| `target_sentence` | The sentence to classify |
| `context_after` | Following sentence |
| `congress` | Congress number (115-118) |
| `party` | Republican / Democratic / Independent |
| `minority` | 1 = minority party, 0 = majority party |
| `committee_name` | Committee that held the hearing |
| `hearing_date` | Date of the hearing |
| `bioguide_id` | Member's Biographical Directory ID |

## Exploration Notebook

```bash
jupyter notebook notebooks/01_explore_bicam.ipynb
```

The notebook walks through metadata inspection, transcript format, speaker segmentation, and sentence splitting interactively.

## Data Sources

- **BICAM** — Downloaded automatically by the pipeline to `~/.bicam/`
  - `hearings/` (~5.4 GB) — Transcripts, metadata, member/witness lists
  - `members/` (~5 MB) — Member info, terms, party history
  - `committees/` (~78 MB) — Committee names and codes
- **Original labeled data** (not in this repo):
  - `RA_merged_with_agreement.csv` — 1,945 human-labeled sentences (gold labels)
  - `df_withempirical.RData` — ~5.8M sentences from 1997-2015 era
