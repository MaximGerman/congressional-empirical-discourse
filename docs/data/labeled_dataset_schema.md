# Labeled Dataset Schema: `RA_merged_with_agreement.csv`

*Documentation of the gold-labeled training dataset from Haim & Barak-Corren (2026)*

---

## Overview

- **File:** `RA_merged_with_agreement.csv`
- **Rows:** 1,945 labeled sentences
- **Period:** 1997--2015 (105th--114th Congress)
- **Class distribution:** 225 empirical (11.6%) / 1,720 non-empirical (88.4%)
- **Source:** Human annotation by research assistants with author adjudication (inter-rater Cohen's Kappa = 0.44)

## Column Descriptions

| # | Column | Description |
|---|--------|-------------|
| 0 | `congress_rater1` | Congress number (e.g., 109 = 109th Congress, 2005--2007) |
| 1 | `committee_code2_rater1` | Committee code (e.g., "HSSY" = House Science and Technology) |
| 2 | `title_rater1` | Hearing title |
| 3 | `file_name_rater1` | Original transcript filename (e.g., `CHRG-109hhrg98564.txt`) |
| 4 | `thomas_name_rater1` | Speaker name in THOMAS format ("Last, First") |
| 5 | `govtrack_rater1` | GovTrack ID for the speaker |
| 6 | `speech_rater1` | Full speech text (the complete speech chunk the sentence comes from) |
| 7 | `rownumber_rater1` | Row number identifier |
| 8 | `gscore_rater1` | Grandstanding score (from Park 2021) |
| 9 | `powercmt_rater1` | Power committee indicator |
| 10 | `security_rater1` | Security-related committee indicator |
| 11 | `year_rater1` | Year range of the Congress session (e.g., "2005-2007") |
| 12 | `minority_rater1` | Minority party status (0 = majority, 1 = minority) |
| 13 | `unified_rater1` | Unified government indicator (1 = same party controls House, Senate, Presidency) |
| 14 | `minuni_rater1` | Interaction: minority x unified |
| 15 | `partyloyalty_rater1` | Party loyalty score |
| 16 | `votepct100_rater1` | Vote percentage (scaled to 100) |
| 17 | `votepct_sq100_rater1` | Vote percentage squared |
| 18 | `seniority_rs_rater1` | Seniority (rescaled) |
| 19 | `seniority_sq_rs_rater1` | Seniority squared (rescaled) |
| 20 | `abs_dwnom1_rs_rater1` | Absolute DW-NOMINATE first dimension score (rescaled; ideological extremity) |
| 21 | `dem_rater1` | Democrat indicator (1 = Democrat, 0 = Republican) |
| 22 | `freshman_rater1` | Freshman member indicator |
| 23 | `female_rater1` | Female indicator |
| 24 | `chairspeech_rater1` | Chair speech indicator |
| 25 | `rankmemspeech_rater1` | Ranking member speech indicator |
| 26 | `leader_rater1` | Leadership position indicator |
| 27 | `salience_rater1` | Hearing salience score |
| 28 | `salience_rs_rater1` | Hearing salience (rescaled) |
| 29 | `polar_rater1` | Polarization score |
| 30 | `polar_rs_rater1` | Polarization (rescaled) |
| 31 | `context_before_rater1` | Preceding sentence (context window) |
| 32 | `target_sentence` | **The sentence to classify** |
| 33 | `context_after_rater1` | Following sentence (context window) |
| 34 | `empirical_binary` | **Gold label: 1 = empirical, 0 = non-empirical** |
| 35 | `empirical_quotes` | Quoted empirical evidence (if applicable) |
| 36 | `empirical_category` | **Category of empirical content** (causal, correlational, descriptive, monetary, statistical, qualitative, historical, other) |

## Key Columns for Model Training

For the binary classification task, the primary inputs and labels are:

- **Input:** `context_before_rater1` + `target_sentence` + `context_after_rater1` (concatenated with context window, matching the original paper's approach)
- **Label:** `empirical_binary` (0 or 1)

For the category classification task:
- **Input:** Same as above (only for sentences where `empirical_binary` = 1)
- **Label:** `empirical_category`

## Notes

- The `_rater1` suffix on most columns indicates these values come from the first rater's annotations (after adjudication/merging).
- The `speech_rater1` column contains the full speech chunk, which can be very long. The model uses only the target sentence with its context window.
- Political metadata columns (minority, seniority, party, etc.) are used for the downstream political analysis, not for classification training.
