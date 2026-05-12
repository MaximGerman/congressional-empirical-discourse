# BICAM Data Enrichment Plan

**Goal:** Add all covariates needed for Phase 4 temporal analysis (OLS regressions replicating the original paper's specification).

**Status:** The BICAM pipeline produces matched legislator sentences with basic metadata (`party`, `minority`, `congress`, `committee_code`, `bioguide_id`). This plan adds the missing covariates.

**Constraint:** All enrichment happens in `src/pipeline.py` step 4 or a new step. Output is the same `sentences_enriched.csv` with additional columns. No changes to steps 1-3.

**Reference columns from original dataset (`RA_merged_with_agreement.csv`):**
- `minority_rater1` — ✅ already have
- `seniority_rs_rater1`, `seniority_sq_rs_rater1` — ❌ missing
- `abs_dwnom1_rs_rater1` — ❌ missing
- `dem_rater1` — ✅ derivable from `party`
- `freshman_rater1` — ❌ missing
- `female_rater1` — ❌ missing
- `chairspeech_rater1`, `rankmemspeech_rater1` — ❌ missing
- `leader_rater1` — ❌ missing
- `unified_rater1`, `minuni_rater1` — ❌ missing
- `votepct100_rater1`, `votepct_sq100_rater1` — ❌ missing
- `salience_rater1` — ❌ missing (methodology unclear)
- `polar_rater1` — ❌ missing (methodology unclear)
- `partyloyalty_rater1` — ❌ missing
- `gscore_rater1` — ❌ missing (Park 2021 grandstanding score — likely out of scope)

---

## Task 1: Fix `hearing_date` field

**Status:** DONE
**Effort:** Small
**Dependencies:** None
**Files:** `src/data.py`, `src/pipeline.py`

### Problem
The `hearing_date` column has incorrect values — e.g., 1992 dates appearing for 115th Congress hearings (2017-2019). The dates likely come from BICAM's `hearings_metadata.csv` and are either misparsed or refer to a different field.

### Steps
- [ ] Inspect BICAM's `hearings_metadata.csv` to find the correct date column
- [ ] Check if `updated_at` vs another field is being used incorrectly
- [ ] Fix the date mapping in `src/data.py` (likely in `load_hearings_metadata()` or the join in the pipeline)
- [ ] Validate: all dates should fall within the expected congress range (115th: 2017-01-03 to 2019-01-03, etc.)
- [ ] Rerun pipeline and confirm dates are correct

### Validation
```python
# Every hearing date should fall within its congress term
congress_ranges = {
    115: ("2017-01-03", "2019-01-03"),
    116: ("2019-01-03", "2021-01-03"),
    117: ("2021-01-03", "2023-01-03"),
    118: ("2023-01-03", "2025-01-03"),
}
```

---

## Task 2: Voteview enrichment (DW-NOMINATE, gender, seniority, freshman, vote share)

**Status:** DONE
**Effort:** Medium
**Dependencies:** None (can run in parallel with Task 1)
**Files:** `src/voteview.py` (new), `tests/test_voteview.py` (new), `src/pipeline.py` (updated)

### Background
Voteview (voteview.com) publishes `HSall_members.csv` — a single CSV with every member of Congress, including NOMINATE scores, party, state, gender, and bioguide ID. This one file gives us most missing covariates.

### Steps
- [ ] Download `HSall_members.csv` from https://voteview.com/data (or use their API). Store in `data/external/` (gitignored) or fetch at pipeline runtime.
- [ ] Create `src/voteview.py` with a function `load_voteview_members(congress_range)` that:
  - Reads `HSall_members.csv`
  - Filters to `chamber == "House"` and target congresses (115-118)
  - Returns DataFrame with columns: `bioguide_id`, `congress`, `nominate_dim1`, `nominate_dim2`, `born`, `gender` (derived from `gender` column if available, otherwise from name/bio)
- [ ] Derive additional columns:
  - `abs_dwnom1` = absolute value of `nominate_dim1` (ideological extremity)
  - `seniority` = number of prior congresses served by that member (count rows with same `bioguide_id` and earlier congress)
  - `freshman` = 1 if this is the member's first congress, else 0
  - `vote_pct` = from Voteview's vote share columns if available, otherwise skip
- [ ] Join onto `sentences_enriched.csv` on `(bioguide_id, congress)`
- [ ] Add tests in `tests/test_voteview.py`

### Output columns added
- `nominate_dim1` (float)
- `nominate_dim2` (float)
- `abs_dwnom1` (float, derived)
- `gender` (str: "M" or "F")
- `seniority` (int, count of congresses served including current)
- `freshman` (int, 0/1)
- `vote_pct` — deferred to Task 6

### Validation
- No nulls in `nominate_dim1` for matched members (Voteview coverage is near-complete for recent congresses)
- Seniority >= 1 for all members
- Freshman members should have seniority == 1
- Cross-check: `party` from Voteview should match `party` from BICAM (flag discrepancies)

---

## Task 3: Unified government flag

**Status:** DONE
**Effort:** Small
**Dependencies:** None (can run in parallel with Tasks 1-2)
**Files:** `src/pipeline.py` or `src/data.py`

### Background
Unified government = same party controls House, Senate, and Presidency. This is a Congress-level variable, not member-level.

### Lookup table
| Congress | Years | House | Senate | President | Unified | Controlling party |
|----------|-------|-------|--------|-----------|---------|-------------------|
| 115 | 2017-2019 | R | R | R (Trump) | 1 | Republican |
| 116 | 2019-2021 | D | R | R (Trump) | 0 | — |
| 117 | 2021-2023 | D | D* | D (Biden) | 1 | Democrat |
| 118 | 2023-2025 | R | D | D (Biden) | 0 | — |

*Senate 117th: 50-50 split, VP Harris breaks tie → Democratic control.

### Steps
- [x] Add a constant dict in `src/data.py` mapping congress → unified (0/1)
- [x] Derive `minuni` interaction term: `minority * unified`
- [x] Join onto enriched data on `congress`
- [x] Add test

### Output columns added
- `unified` (int, 0/1)
- `minuni` (int, 0/1 — interaction of minority × unified)

---

## Task 4: Committee leadership (chair / ranking member)

**Status:** NOT STARTED
**Effort:** Medium-Large
**Dependencies:** Task 2 (needs `bioguide_id` confirmed reliable)
**Files:** New `src/leadership.py`, update `src/pipeline.py`

### Background
For each committee-congress pair, one member is Chair and one is Ranking Member. This data is available from:
- congress.gov API (structured but rate-limited)
- BICAM's `hearings_members.csv` (may have role info — check first)
- unitedstates/congress-legislators GitHub repo (has committee membership YAML with leadership roles)

### Steps
- [ ] First check if BICAM's `hearings_members.csv` already has a role/position column (cheapest option)
- [ ] If not, download committee membership data from `unitedstates/congress-legislators` repo (`committees-current.yaml`, `committee-membership-current.yaml`)
- [ ] Create `src/leadership.py` with `load_committee_leaders(congress_range)`:
  - Returns DataFrame: `bioguide_id`, `congress`, `committee_code`, `role` (chair/ranking_member/member)
- [ ] Derive columns:
  - `chairspeech` = 1 if speaker is committee chair for that hearing's committee
  - `rankmemspeech` = 1 if speaker is ranking member
  - `leader` = 1 if chair or ranking member
- [ ] Join onto enriched data on `(bioguide_id, congress, committee_code)`
- [ ] Add tests

### Output columns added
- `chairspeech` (int, 0/1)
- `rankmemspeech` (int, 0/1)
- `leader` (int, 0/1)

### Complication
Committee codes may differ between BICAM and congress-legislators. Need a mapping table or fuzzy match on committee names.

---

## Task 5: Hearing salience and polarization scores

**Status:** NOT STARTED — NEEDS METHODOLOGY CLARIFICATION
**Effort:** Unknown
**Dependencies:** Task 2 (polarization may derive from DW-NOMINATE)
**Files:** TBD

### Background
The original paper includes `salience_rater1` and `polar_rater1` but the computation methodology isn't documented in our references. These may be:
- **Salience:** Media coverage of the hearing topic, or number of witnesses, or CQ almanac coding
- **Polarization:** Could be committee-level or topic-level, possibly derived from DW-NOMINATE spread

### Steps
- [ ] Ask Amit Haim how `salience` and `polar` were computed in the original dataset
- [ ] If derivable from available data (e.g., polarization = std of DW-NOMINATE within committee), implement
- [ ] If requires external data (media coverage, CQ coding), assess feasibility and decide whether to drop from replication or find a proxy
- [ ] Same for `partyloyalty` and `gscore` (grandstanding score from Park 2021)

### Decision needed
These columns may not be essential for the core Phase 4 analyses (trend, minority gap, event study). They are control variables. If they can't be reproduced, the regression specification should note the omission rather than block the project.

---

## Task 6: Election vote share (vote_pct)

**Status:** NOT STARTED
**Effort:** Small-Medium
**Dependencies:** Task 2 (needs bioguide_id, congress, plus state/district from Voteview)
**Files:** New `src/elections.py`, update `src/pipeline.py`

### Background
The original paper uses `votepct100_rater1` and `votepct_sq100_rater1` as controls for electoral safety. Voteview does not carry election results. The MIT Election Lab publishes `1976-2022-house.csv` on Harvard Dataverse with candidate-level vote shares by state, district, and election year.

### Steps
- [ ] Download MIT Election Lab House elections data from Harvard Dataverse. Store in `data/external/`.
- [ ] Create `src/elections.py` with a function to load and filter to target election cycles
- [ ] Map election years to congresses (e.g., 2016 election → 115th Congress)
- [ ] Join on state + district (from Voteview) + congress to get each member's vote share
- [ ] Derive `vote_pct_sq` = vote_pct² (quadratic term for non-linear effects)
- [ ] Merge onto enriched data on `(bioguide_id, congress)`
- [ ] Add tests in `tests/test_elections.py`

### Output columns added
- `vote_pct` (float, 0-100 scale to match original paper's `votepct100`)
- `vote_pct_sq` (float, vote_pct²)

### Notes
- Members who ran unopposed will have vote_pct ~100%. This is valid, not an error.
- Special elections may need separate handling (mid-term replacements).
- MIT Election Lab data currently covers through 2022; 118th Congress members elected in 2022 should be covered. Check coverage for any 2024 special elections.

---

## Execution Order

```
Week 1 (parallel):
  Task 1 (hearing_date fix)     ████  [small, independent]
  Task 2 (Voteview enrichment)  ████████  [medium, independent]
  Task 3 (unified government)   ██  [small, independent]

Week 2:
  Task 4 (committee leadership) ████████████  [medium-large, needs bioguide_id]

When info available:
  Task 5 (salience/polarization) ████████  [blocked on methodology from Amit]
```

Tasks 1, 2, and 3 can all be done in parallel by separate agents.
Task 4 depends on Task 2 only loosely (confirmed bioguide_id reliability).
Task 5 is blocked on external input.

---

## Integration

After each task, rerun the pipeline (`python -m src.pipeline`) and verify:
1. No regressions in existing columns
2. New columns have expected null rates (should be near-zero for Voteview data)
3. Cross-check a few values manually against congress.gov

Final validation: compare column distributions between our enriched BICAM data and the original `RA_merged_with_agreement.csv` for overlapping congresses (if any) to sanity-check consistency.
