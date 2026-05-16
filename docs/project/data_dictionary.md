# BICAM Enriched Data Dictionary

This document serves as a comprehensive reference for the `sentences_enriched.csv` dataset. The dataset contains sentence-level utterances from congressional hearings, enriched with political, demographic, and electoral metadata for the speakers.

## Core Utterance Data
Derived from the raw BICAM transcript files.

*   **`hearing_id`** (String): Unique identifier for the hearing (from BICAM).
*   **`sentence_id`** (Integer): Sequential ID for the sentence within the hearing.
*   **`speaker`** (String): Raw speaker name as it appears in the transcript (e.g., "Mr. SMITH").
*   **`text`** (String): The text of the spoken sentence.
*   **`char_start`** (Integer): Start character index of the sentence in the raw transcript.
*   **`char_end`** (Integer): End character index of the sentence in the raw transcript.

## Hearing Metadata
Hearing-level information derived from BICAM metadata files.

*   **`congress`** (Integer): The congressional session (e.g., 115, 116).
*   **`chamber`** (String): The congressional chamber (e.g., "House").
*   **`title`** (String): Title of the hearing.
*   **`committee_code`** (String): The code of the committee holding the hearing (e.g., "HSAS").
*   **`committee_name`** (String): The full name of the committee.
*   **`hearing_date`** (Date): The date the hearing took place.

## Speaker Matching Metadata
Metadata resulting from matching the raw `speaker` string to the official member database.

*   **`bioguide_id`** (String): The official biographical ID of the member of Congress.
*   **`member_first_name`** (String): The matched member's first name.
*   **`speaker_last_name`** (String): Parsed last name of the speaker from the transcript.
*   **`party`** (String): The member's political party (e.g., "Republican", "Democratic").
*   **`member_state`** (String): The state represented by the member.
*   **`match_score`** (Float): Confidence score of the name match (100 for exact).
*   **`match_type`** (String): The strategy used to match the speaker (e.g., "hearing_member_exact", "congress_exact").

## Political & Demographic Covariates
Covariates added during the enrichment pipeline, replicating variables from the original paper.

### Party and Majority Status
*   **`dem`** (Integer, 0/1): 1 if the member is a Democrat, 0 otherwise.
*   **`minority`** (Integer, 0/1): 1 if the member's party is in the minority in the House for that Congress.
*   **`unified`** (Integer, 0/1): 1 if there is a unified government (same party controls House, Senate, and Presidency), 0 otherwise.
*   **`minuni`** (Integer, 0/1): Interaction term `minority` * `unified`.

### Ideology and Demographics (Source: Voteview)
*   **`nominate_dim1`** (Float): The member's first dimension DW-NOMINATE score (economic/liberal-conservative).
*   **`nominate_dim2`** (Float): The member's second dimension DW-NOMINATE score.
*   **`abs_dwnom1`** (Float): The absolute value of `nominate_dim1`, capturing ideological extremity.
*   **`abs_dwnom1_rs`** (Float): Rank-standardized `abs_dwnom1` within the specific Congress (percentage rank).
*   **`gender`** (String): Member's gender ("M", "F", etc.).
*   **`female`** (Integer, 0/1): 1 if the member is female, 0 otherwise.

### Seniority (Source: Voteview)
*   **`seniority`** (Integer): The number of terms the member has served in the House (up to and including the current Congress).
*   **`seniority_sq`** (Integer): `seniority` squared.
*   **`seniority_rs`** (Float): Rank-standardized `seniority` within the specific Congress (percentage rank).
*   **`seniority_sq_rs`** (Float): The square of `seniority_rs`.
*   **`freshman`** (Integer, 0/1): 1 if this is the member's first term (`seniority` == 1), 0 otherwise.

### Committee Leadership (Source: Static Lookup Table)
*   **`chairspeech`** (Integer, 0/1): 1 if the speaker is the Chair of the committee holding the hearing.
*   **`rankmemspeech`** (Integer, 0/1): 1 if the speaker is the Ranking Member of the committee.
*   **`leader`** (Integer, 0/1): 1 if the speaker is either the Chair or Ranking Member.

### Electoral Safety (Source: MIT Election Lab)
*   **`state_abbrev`** (String): State abbreviation.
*   **`district_code`** (Integer): Congressional district number.
*   **`vote_pct`** (Float, 0-100): The percentage of the two-party vote the member received in the most recent election.
*   **`vote_pct_sq`** (Float): `vote_pct` squared, to capture non-linear electoral safety effects.
