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
│   ├── preprocess.py          # Speaker segmentation, sentence splitting
│   ├── elections.py           # MIT Election Lab data enrichment
│   ├── leadership.py          # Committee leadership enrichment
│   ├── voteview.py            # NOMINATE, seniority, gender enrichment
│   └── pipeline.py            # Pipeline orchestration
├── scripts/
│   ├── explorer.py            # Streamlit Data Explorer app
│   ├── optimize_data.py       # Parquet conversion utility
│   ├── update_data_dict.py    # Automated data dictionary generator
│   └── components/            # Explorer UI modules (Tabs)
├── tests/                     # Unit & integration tests
├── notebooks/                 # Interactive analysis
└── data/                      # Local data storage
    ├── sentences_enriched.parquet # Optimized data for analysis
    └── ...
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

## BICAM Dataset Explorer

For interactive analysis and data quality diagnostics, we provide a Streamlit-based explorer:

```bash
make explorer
```

The explorer includes tabs for:
- **Overview**: High-level dataset statistics and distributions.
- **Search**: Keyword search across millions of sentences with metadata filtering.
- **Diagnostics**: Matching rate analysis and covariate coverage checks.
- **Insights**: Advanced visualizations and correlation analysis.

## Parquet Optimization

To handle the ~2.4GB enriched dataset efficiently, we use the Apache Parquet format. Parquet provides columnar storage and compression, reducing load times from minutes (CSV) to seconds.

If you have the CSV version but need the optimized Parquet file:

```bash
make optimize
```

## Automated Data Dictionary

The project maintains an automated [Data Dictionary](docs/project/data_dictionary.md) that synchronizes with the source code. It is updated automatically via pre-commit hooks, ensuring that documentation always reflects the latest column definitions and enrichment logic.

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
| `unified_government` | 1 = unified government, 0 = divided government |
| `minuni` | Interaction term (minority × unified_government) |
| `committee_name` | Committee that held the hearing |
| `hearing_date` | Date of the hearing |
| `bioguide_id` | Member's Biographical Directory ID |
| `chairspeech` | 1 if speaker is committee Chair, 0 otherwise |
| `rankmemspeech` | 1 if speaker is Ranking Member, 0 otherwise |
| `leader` | 1 if speaker is Chair or Ranking Member, 0 otherwise |
| `nominate_dim1` | DW-NOMINATE first dimension (liberal-conservative) |
| `seniority` | Number of terms served in Congress |
| `female` | 1 if member is female, 0 if male |
| `vote_pct` | Percentage of votes received in last election |
| `vote_pct_sq` | Squared vote percentage (for non-linear effects) |

## Development

The project includes a `Makefile` for common development tasks:

```bash
make check         # Run all quality checks (lint, format, typecheck)
make test          # Run tests with coverage report
make format        # Automatically format code and fix lint issues
make typecheck     # Run mypy static type analysis
make explorer      # Launch the Streamlit data explorer
make optimize      # Convert CSV data to optimized Parquet
make update-docs   # Synchronize the data dictionary with source code
```

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
