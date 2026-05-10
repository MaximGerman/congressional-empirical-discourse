# Presentation: In This House We Believe in... Empirical Evidence?

**Authors:** Amit Haim & Netta Barak-Corren

*Transcribed from: `presentation_empirical_evidence_congress.pptx` (30 slides)*

---

## Slide 1 -- Title
**In This House We Believe in... Empirical Evidence? Power and Knowledge Dynamics in Congressional Hearings**
Amit Haim & Netta Barak-Corren

## Slide 2 -- Research Question
Are legislators interested in empirical evidence, and how does it influence their work?

## Slide 3 -- The House of Representatives
> "All Legislative Powers herein granted shall be vested in a Congress of the United States, which shall consist of a Senate and House of Representatives."
> -- Article I, Section 1, United States Constitution

## Slide 4 -- The Legislative Process (in a nutshell)
1. A representative sponsors the Bill
2. Bill assigned for a committee
3. Public hearings (witnesses, etc.)
4. Mark-up session(s) (committee debates, votes on amendments)
5. Bill is reported (released for the House) or tabled (killed)
6. Committees also conduct oversight hearings, which can also conclude in reports
7. Bill released for vote, debate, or amendment (floor)
8. Bill dies / remanded / passes to the Senate

## Slide 5 -- House Committee Work and Hearings
- Legislative Hearings
- Oversight Hearings

## Slide 6 -- Methods: Data
- U.S. House Committee Hearings (1997--2015; 104th--115th)
- N=12,812 hearings parsed into speaker-speech chunks; 5.9 million sentences
- Metadata on political constellation (who controls what in gov't), members (party, seniority, political minority status, gender), committees (political control, committee topic), hearing types (oversight/legislative), and bill status (failed/passed/signed into law)
- *Note: Data originally from Ju Yeon Park. Currently being expanded to 2025.*

## Slide 7 -- Methods: Machine Learning Classification
- Human-annotated data (~2000 sentences) used to fine-tune RoBERTa model (LLM approach yielded unremarkable results)
- Validation with ClaimBuster (.58 correlation)
- Sentence-level classification of empirical content (binary) and empirical category (multi-class: causal, correlational, statistical, qualitative, historical, monetary, etc.)
- Analysis aggregated for speaker-chunk, speaker-hearing, and hearing

## Slides 8--9
*[Figures/visualizations]*

## Slide 10 -- Key Findings
- Empirical discourse has increased over time, but not necessarily due to evidence-based policymaking
- Instead, empirical discourse is strategically used by the minority party to challenge the majority (vis-a-vis increased polarization)
- Minority legislators (from both parties) use more empirical discourse than majority members
- The larger the power gap, the more the minority relies on empirical discourse

## Slide 11 -- Empirical Discourse Over Time & Overall Distribution
- Model performance:
  - Accuracy: 0.91
  - Precision: 0.79
  - Recall: 0.76

## Slides 12--13 -- Power Dynamics and Empirical Discourse 1
*[Figures showing party-level trends with house control background shading]*

## Slide 14 -- Power Dynamics and Empirical Discourse 2
- **Majority party:** Controls the legislative agenda
- **Minority party:** Lacks power, so it relies on empirical discourse as a substitute

## Slide 15 -- Power Dynamics and Empirical Discourse 3
- **Seniority Effect:**
  - Senior minority members use more empirical discourse
  - Senior majority members use less, while junior majority members use slightly more

## Slides 16--18 -- Power Dynamics and Empirical Discourse 4
*[Placeholder content -- event study figures and trifecta analysis]*

## Slides 19--20
*[Additional figures]*

## Slide 21 -- Legislative Stage and Outcomes
- Legislative hearings are less empirical compared to oversight hearings
- Hearings with specific bills on the table are less empirical than general legislative hearings (in line with Ban, Park & You's findings on witnesses)
- Bills that passed a House vote had less empirical discussion in committee, compared to failed bills

## Slide 22 -- Legislative Stage and Outcomes
*[Placeholder -- additional outcome figures]*

## Slide 23 -- Why Do Bills with Empirical Background Fail?
**Hypothesis:**
- Minority legislators bring evidence as weapon against the bill
- Evidence is an effective weapon
- Raising doubts, concerns, ultimately leading to the bill's failure (at least in borderline cases)

**Supporting finding:** As bills progress, empirical discourse declines. Possibly not only because less information is needed (Ban, Park & You 2022), also because the majority may be posing stricter controls on the conversation.

**Need to explore the hypothesis against:**
- Bills sponsored by both parties
- Bills sponsored by minority members

## Slide 24
*[Figure]*

## Slide 25 -- Substantial Differences across Committees
*[Bar chart of empirical scores by committee]*

## Slide 26 -- And Hearing Topics
*[Bar chart of empirical scores by policy topic]*

## Slide 27 -- Constitutional Discourse -- Less Empirical
*[Figure]*

## Slide 28 -- Discussion
- "Power or Knowledge": majority control the committee agenda (policy outcomes and leverage political exposure)
- Minority members need alternative ways to influence; wield factual statements as challenging the dominant forces, somewhat successfully
- While use of empirical discourse has risen over time (against general expectations; but short time span?), may be driven more by partisanship than by an increased demand for evidence
- Increased partisanship -> Power through coalition-building declined -> Minority (mostly the Democrats, but sometimes Republicans, too) in stronger need for using knowledge as substitute for power
- Park (2021) finds that minority members "grandstand" more; but less comprehensive definition?
- Limitations: many

## Slide 29
Comments, Questions, please!

## Slide 30
*[End]*
