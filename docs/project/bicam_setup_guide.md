# BICAM Data Setup Guide

## Background

This document guides the setup of the BICAM data pipeline for the workshop project **"Temporal Robustness of Empirical Discourse Classification in U.S. Congressional Hearings (1997-2025)"**.

### Project Context

- **Original paper:** Haim & Barak-Corren analyzed empirical discourse in U.S. House Committee Hearings (1997-2016) using a fine-tuned RoBERTa model on ~5.8M sentences across 12,428 hearings. They found that minority-party legislators strategically use more empirical discourse ("Power or Knowledge" framework).
- **Our extension:** We are extending the analysis to 2015-2025 using modern models (DeBERTa-v3, Llama-3-8B) and need to scrape, preprocess, and prepare the new-era data from the BICAM dataset.
- **Proposal file:** `proposal.tex` in this directory.
- **Deadline:** Proposal due May 23, 2026. Final submission in July.

### What We Already Have

| File | Description |
|------|-------------|
| `RA_merged_with_agreement.csv` | 1,945 human-labeled sentences (1997-2015). Gold labels. 225 empirical (11.6%), 1720 non-empirical. |
| `df_withempirical.RData` | ~5.8M unlabeled sentences from 1997-2015 hearings (531MB R data file). Used for continual pre-training. |
| `empirical_evidence_congress.pdf` | The original Haim & Barak-Corren paper. |

### What We Need from BICAM

Congressional hearing **transcripts** for the **115th-119th Congresses** (roughly 2017-2025). These need to be:
1. Downloaded from BICAM
2. Segmented into per-speaker speech chunks
3. Further split into individual sentences with context windows (matching the original paper's format)
4. Enriched with political metadata (party, majority/minority status, committee, etc.)

---

## Step 1: Install BICAM and Download Hearings Data

### Install

```bash
pip install bicam
# or
uv pip install bicam
```

### Download the hearings dataset

```bash
# List available datasets
bicam list-datasets --detailed

# Download hearings (~1.7-3.5GB, will take a while)
bicam download hearings --confirm
```

Or via Python:

```python
import bicam

# Download
hearings_path = bicam.download_dataset('hearings', confirm=True)

# Check what files are in the download
import os
for f in os.listdir(hearings_path):
    print(f)
```

You can configure the cache directory:
```bash
export BICAM_DATA=/path/to/your/storage  # e.g. on the cluster
```

### What you'll get

The hearings dataset contains two CSV files:

| File | Contents |
|------|----------|
| `hearings_metadata.csv` | Hearing-level metadata: `hearing_id`, `committee_id`, `congress`, `title`, `date` |
| `hearings_texts.csv` | Full transcripts: `hearing_id`, `text` (the raw transcript body) |

**Congress coverage:** 105th-118th (some up to 119th). We need **115th-119th** (2017-2025).

---

## Step 2: Explore and Filter the Data

### Load and inspect

```python
import bicam
import pandas as pd

# Load metadata
meta_df = bicam.load_dataframe('hearings', 'hearings_metadata.csv', download=True)
print(f"Total hearings: {len(meta_df)}")
print(f"Columns: {list(meta_df.columns)}")
print(f"Congress range: {meta_df['congress'].min()} - {meta_df['congress'].max()}")

# Filter to our target period (115th Congress onwards = 2017+)
new_era = meta_df[meta_df['congress'] >= 115]
print(f"Hearings from 115th Congress onwards: {len(new_era)}")
print(f"Breakdown by congress:")
print(new_era['congress'].value_counts().sort_index())
```

### Load transcripts

```python
# WARNING: This file is large (potentially several GB). 
# Consider using polars or dask for memory efficiency.
texts_df = bicam.load_dataframe('hearings', 'hearings_texts.csv', 
                                 download=True, df_engine='polars')

# Or with pandas, load only the hearings we need:
target_ids = set(new_era['hearing_id'])
chunks = []
for chunk in pd.read_csv(hearings_texts_path, chunksize=1000):
    filtered = chunk[chunk['hearing_id'].isin(target_ids)]
    chunks.append(filtered)
new_texts = pd.concat(chunks)
print(f"Loaded {len(new_texts)} transcript records for 2017+ hearings")
```

### Key questions to answer during exploration

- [ ] What does the `text` field look like? Is it one giant string per hearing, or already segmented?
- [ ] Are speaker names embedded in the transcript text (e.g., "Mr. SMITH. I believe that...")?
- [ ] How does the format compare to the original paper's source files (named like `CHRG-109hhrg98564.txt`)?
- [ ] Are there any obvious data quality issues (empty transcripts, truncated text, encoding problems)?

---

## Step 3: Preprocess Transcripts into Sentence-Level Format

The original paper's pipeline works as follows (described in Section 3.1 of the paper):

1. **Hearing transcript** -> split into **per-speaker speech chunks** (based on speaker attribution)
2. **Each speech chunk** -> split into **individual sentences**
3. **Each sentence** gets a **context window**: the preceding sentence and following sentence

### 3a: Speaker Segmentation

BICAM transcripts likely contain raw text with embedded speaker markers. The typical format in congressional hearing transcripts is:

```
Mr. SMITH. Thank you, Mr. Chairman. I want to...
Mrs. JONES. I appreciate the gentleman's remarks...
```

You need to parse these into structured records:

```python
import re

def segment_speakers(transcript_text):
    """
    Split a raw hearing transcript into per-speaker chunks.
    
    Congressional transcripts typically use patterns like:
    - "Mr. LASTNAME." or "Mrs. LASTNAME."
    - "Chairman LASTNAME."
    - "The CHAIRMAN."
    - Full names in caps: "Senator FIRSTNAME LASTNAME."
    
    IMPORTANT: Inspect the actual BICAM data first and adjust 
    this regex accordingly. The format may differ from what's 
    described here.
    """
    # Common pattern - adjust after inspecting actual data
    speaker_pattern = re.compile(
        r'((?:Mr\.|Mrs\.|Ms\.|Dr\.|Chairman|Chairwoman|The CHAIRMAN|Senator|Representative)\s+[A-Z][A-Za-z\-]+)\.'
    )
    
    chunks = []
    current_speaker = None
    current_text = []
    
    for line in transcript_text.split('\n'):
        match = speaker_pattern.match(line.strip())
        if match:
            if current_speaker:
                chunks.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = match.group(1)
            remainder = line[match.end():].strip()
            current_text = [remainder] if remainder else []
        else:
            if line.strip():
                current_text.append(line.strip())
    
    if current_speaker:
        chunks.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text)
        })
    
    return chunks
```

### 3b: Sentence Splitting with Context Windows

```python
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def create_sentence_records(speech_chunks, hearing_id):
    """
    Split speech chunks into individual sentences with context.
    Matches the format from the original paper's labeled data:
    - context_before: preceding sentence
    - target_sentence: the sentence to classify
    - context_after: following sentence
    """
    records = []
    for chunk in speech_chunks:
        sentences = sent_tokenize(chunk['text'])
        for i, sent in enumerate(sentences):
            record = {
                'hearing_id': hearing_id,
                'speaker': chunk['speaker'],
                'context_before': sentences[i-1] if i > 0 else '',
                'target_sentence': sent,
                'context_after': sentences[i+1] if i < len(sentences)-1 else '',
                'speech_text': chunk['text']  # keep full speech for reference
            }
            records.append(record)
    return records
```

### 3c: Output Format

The final sentence-level dataset should match the structure of the existing labeled data:

```
hearing_id | congress | committee_id | title | date | speaker | context_before | target_sentence | context_after
```

---

## Step 4: Enrich with Political Metadata

The original paper's analysis requires per-speaker metadata: party affiliation, majority/minority status, seniority, etc. BICAM provides a **members** dataset that can help:

```python
# Download members data
members_df = bicam.load_dataframe('members', 'members.csv', download=True)
print(members_df.columns.tolist())
```

### What you need to link:

| Field | Source | Notes |
|-------|--------|-------|
| `congress` | hearings_metadata | Which Congress session |
| `committee_id` | hearings_metadata | Which committee |
| `speaker` / `member_id` | Parsed from transcript + members dataset | Need fuzzy matching of speaker names to member records |
| `party` (dem/rep) | members dataset | Party affiliation |
| `minority` (0/1) | Derived | Requires knowing which party controlled the House in each Congress |
| `seniority` | members dataset or external | May need external data source |

### House control by Congress (for majority/minority coding):

| Congress | Years | House Majority |
|----------|-------|---------------|
| 115th | 2017-2019 | Republican |
| 116th | 2019-2021 | Democrat |
| 117th | 2021-2023 | Democrat |
| 118th | 2023-2025 | Republican |
| 119th | 2025-2027 | Republican |

### Important considerations:

- **Witnesses vs. members:** The original paper FILTERED OUT witnesses and only kept legislators. You need to do the same. Witnesses are typically identified by titles like "Dr.", professional affiliations, or by not matching any known member name.
- **Name matching:** Congressional transcripts use inconsistent name formats. You'll likely need fuzzy matching between transcript speaker names and the members database. Consider using `fuzzywuzzy` or `rapidfuzz`.
- **Some metadata may need external sources:** The original paper used data from Park (2021) and Ban, Park, and You (2022) for detailed political metadata. Ask Amit if he can share the metadata files he used -- they're likely separate from the transcript data.

---

## Step 5: Create a Representative Sample for Silver Labeling

For Phase 2 of the project (LLM silver labeling), we need ~10,000 sentences from the new era:

```python
# After preprocessing, sample strategically:
# - Proportional to congress (not all from one session)
# - Include multiple committees
# - Include both majority and minority speakers

sample = new_era_sentences.groupby('congress').apply(
    lambda x: x.sample(min(len(x), 2500), random_state=42)
).reset_index(drop=True)
print(f"Sample size: {len(sample)}")
print(f"Congress distribution:\n{sample['congress'].value_counts().sort_index()}")
```

---

## Priorities and Risk Assessment

### Must-do this week (before May 23 proposal deadline):

1. **Install BICAM, download hearings data, inspect the raw format.** This determines whether our data pipeline plan is realistic. If the transcripts are not speaker-segmented or are in an unexpected format, we need to know NOW.
2. **Write a small script that processes ONE hearing end-to-end:** download -> parse speakers -> split sentences -> output as CSV. This proves the pipeline works.
3. **Count how many hearings we have for 115th-119th Congress.** This goes into the proposal.

### Medium priority (before implementation):

4. Figure out speaker-to-member matching.
5. Test BICAM members dataset for metadata coverage.
6. Decide on a sampling strategy for silver labeling.

### Known risks:

| Risk | Mitigation |
|------|-----------|
| BICAM transcripts may not have speaker attribution | Fall back to raw GPO/GovInfo transcripts, which DO have speaker markers. The `gpo_tools` package (https://github.com/rbshaffer/gpo_tools) has a parser that handles speaker segmentation. |
| Download is very large (1.7-3.5GB) | Use the cluster storage at `/home/yandex/DLWorkShop2026b` for large files, not local machine |
| Speaker names don't match members database | Use fuzzy matching (`rapidfuzz`) + manual inspection of top mismatches |
| Some congresses have sparse coverage | Check coverage immediately after download; if 119th is too sparse, limit to 115th-118th |
| Transcript format changed across congresses | Compare a few transcripts from different congresses to check format consistency |

---

## Useful Resources

- **BICAM Python docs:** https://py.docs.bicam.net/en/latest/
- **BICAM data portal:** https://bicam.net/
- **BICAM GitHub:** https://github.com/bicam-data
- **BICAM paper (Nature Scientific Data):** https://www.nature.com/articles/s41597-025-05737-8
- **Alternative scraping tool (GPO):** https://github.com/rbshaffer/gpo_tools -- has a `Parser` class that segments hearing transcripts into individual statements with speaker-level metadata. May be useful as a fallback.
- **Original paper's data sources:** Ban, Park, and You (2022); Park (2021) -- the original authors relied on these for hearing transcripts and political metadata.
- **Contact:** Amit Haim (TAD mentor) -- can clarify data format questions and may share additional metadata files.

---

## Quick-Start Checklist

```
[ ] pip install bicam
[ ] bicam download hearings --confirm
[ ] Inspect hearings_metadata.csv: columns, congress range, count per congress
[ ] Inspect hearings_texts.csv: look at 2-3 raw transcripts, understand the format
[ ] Write speaker segmentation regex based on actual observed format
[ ] Process one hearing end-to-end into sentence-level CSV
[ ] Compare output format to RA_merged_with_agreement.csv
[ ] Count total available hearings for 115th-119th Congress
[ ] Report findings back -- update proposal if needed
```
