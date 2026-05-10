# Documentation

Transcriptions of project resources into agent-readable (plaintext/markdown) format.

## Structure

```
docs/
├── references/                          # Original research materials
│   ├── haim_barak_corren_2026.md        # Full paper (40-page PDF transcription)
│   └── presentation_slides.md           # Author presentation slides (30 slides)
│
├── workshop/                            # TAU Deep Learning Workshop materials
│   ├── course_information.md            # Workshop syllabus, schedule, deliverables
│   └── computing_resources.md           # Available GPU/compute resources
│
├── project/                             # Project-specific documentation
│   ├── proposal.tex                     # LaTeX proposal (due May 23, 2026)
│   ├── project_overview.md              # Phase plan and execution strategy
│   └── meeting_notes_kickoff.md         # Zoom kickoff meeting with Amit Haim
│
└── data/                                # Dataset documentation
    └── labeled_dataset_schema.md        # Schema for RA_merged_with_agreement.csv
```

## Files Not Transcribed

These resource files are either already text-readable, binary data, or too large to include:

| File | Reason | Location |
|------|--------|----------|
| `BICAM_SETUP_GUIDE.md` | Already markdown; exists at repo root | `./BICAM_SETUP_GUIDE.md` |
| `proposal.tex` | Already text; copied into `docs/project/` | `docs/project/proposal.tex` |
| `RA_merged_with_agreement.csv` | Tabular data (1,945 rows); schema documented | `docs/data/labeled_dataset_schema.md` |
| `df_withempirical.RData` | 531MB binary R data file; cannot be transcribed | N/A (use on cluster) |

## Source Mapping

| Transcribed File | Original Resource |
|------------------|-------------------|
| `references/haim_barak_corren_2026.md` | `empirical_evidence_congress.pdf` |
| `references/presentation_slides.md` | `presentation_empirical_evidence_congress.pptx` |
| `workshop/course_information.md` | `2026 Workshop Information (1).pptx` |
| `workshop/computing_resources.md` | `Computing Resources.pdf` |
| `project/meeting_notes_kickoff.md` | `Maxim German's Zoom Meeting.pdf` |
| `project/project_overview.md` | `gemini-code-1778318489222.md` |
| `data/labeled_dataset_schema.md` | `RA_merged_with_agreement.csv` (header analysis) |
