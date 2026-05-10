# In This House We Believe in... Empirical Evidence? Power and Knowledge Dynamics in Congressional Hearings

**Authors:** Amit Haim (Tel Aviv University), Netta Barak-Corren (Hebrew University of Jerusalem)

**Date:** May 6, 2026

**Status:** Draft -- Please do not circulate without permission

---

## Abstract

This study examines the use of empirical discourse in U.S. House of Representatives committee hearings (1997--2016, N = 12,428), analyzing over 5.8 million sentences using machine learning classification. We find that empirical discourse has increased over time, not necessarily as a marker of evidence-based policymaking, but as a strategic tool of the minority. Minority legislators -- both Republican and Democrat -- rely on empirical arguments more than majority legislators, and more so as the majority-minority control gap increases. Senior minority members use substantially more empirical discourse, indicating that empirical discourse is acquired as a substitute for power. We estimate the effect as ranging between 15-40% depending on seniority. Moreover, we find that majority party members (mostly Republicans) use even less empirical discourse when their party controls a trifecta of the branches of government, turning less to knowledge when fewer constraints on power exist. We find causal support for this phenomenon using an event study design: when power shifts (twice throughout the duration of our study), those who used less empirical discourse turn to using more of it, and vice versa; this is also true at the individual level for House members who survived reelection but switched from being in the minority to being in the majority or vice versa. Finally, minority legislators use especially high levels of empirical discourse when opposing majority-sponsored bills. Empirical discourse is associated with a lower likelihood of bill passage, for both majority and minority sponsored bills, suggesting that empirical arguments provide at least somewhat effective minority strategy to challenge and thwart majority initiatives. Overall, these findings support a "Power or Knowledge" framework, whereby the majority controls the agenda whereas the minority leverages empirical discourse to challenge dominant forces.

**Keywords:** Empirical evidence, legislative process, congressional hearings, political polarization

---

## 1. Introduction

Legislatures play an important role in modern democratic societies. They craft policy in various domains and subject matters, conduct oversight of government actions and processes, and allocate funds to empower government action. These responsibilities require legislators to be well informed about the topics they handle. The parliamentary process is organized primarily through committees (Shepsle and Weingast 1987), dedicated to specific topics, which conduct hearings, gather evidence from various witnesses, and make decisions on both oversight and legislation before bills are advanced to full chamber votes.

But what kind of information do legislators rely on? Empirical knowledge, from basic facts to descriptive statistics to causal relations, including scientific work across disciplines and issues, is critical. Legislators need evidence to design sound policies, challenge unsound policies, and resolve concerns involving empirical claims, such as the likely effectiveness of a policy, indirect consequences, or the magnitude of the problem. Legislators also need evidence to conduct effective oversight of the executive branch -- to understand what worked, what didn't, and how to make government more efficient and effective. Furthermore, legislators could use compelling evidence that demonstrates the importance and narrow tailoring of their policymaking to protect their laws from constitutional challenges. Do legislators, in fact, care for data and use it in parliamentary work?

We set out to answer this question using textual analysis of committee hearings. We analyzed all transcripts from U.S. House committee hearings spanning the 105th to 114th Congresses (1997--2016, N = 12,428) and coded the use of empirical discourse in legislative and oversight activities. Hearings were divided into per-speaker speech chunks and further into individual sentences to facilitate detailed classification. Meta-data were retained for each hearing, including political context (e.g., the President's party at the time, House and Senate control, committee composition) and speaker data (e.g., party affiliation, gender, and ranking status). Sentences were classified into two key variables: the presence or absence of empirical discourse -- a claim about empirical evidence, or demand for it -- and the type of empirical evidence discussed or sought. We began with human annotations guided by an iterative and refined annotation procedure until it achieved satisfactory reliability and informed the fine-tuning of a machine learning model. This model, fine-tuned on thousands of annotated sentences, achieved high classification accuracy and was used to classify all sentences in the dataset (N = 5,890,603). Using these sentence-level classifications, we calculated an empirical score for each speech-chunk by determining the proportion of sentences containing empirical discourse out of all sentences in the chunk. We then aggregated scores for each hearing and individual speaker, depending on the specific analysis. This score allowed us to analyze patterns in the use of empirical discourse over time, across different political settings, and within various committees and hearing types. We used several analytical methods, including linear regression, an event study with difference in difference estimates of the effect of power shifts on empirical discourse, and comparisons between different sub-groups (e.g. minority vs majority members, senior vs junior members, and more).

### Key Findings

Our analysis reveals notable patterns in the use of empirical discourse in congressional hearings:

1. **Empirical discourse has been on the rise** during the study period (1997--2016). However, this increase does not necessarily indicate more evidence-based lawmaking. Rather, empirical discourse appears to have become a strategic tool for lawmakers, particularly as polarization has intensified.

2. **Minority party members consistently use more empirical discourse** than members of the majority, regardless of partisan affiliation, suggesting that empirical statements serve as a substitute for formal power and a means to influence the legislative agenda. While Democrats tend to use more empirical discourse on average, this reflects their minority status for much of the studied period; when Republicans are in the minority, they too increase their reliance on empirical discourse.

3. **The minority-majority gap in empirical discourse widens as the power gap between the parties grows** -- the more acute the minority's disadvantage, the greater their use of empirical discourse. Seniority further sharpens this pattern: senior minority members make particularly frequent use of empirical discourse, while among the majority it is the junior members who show a slight tendency to rely more on empirical arguments.

4. **When Republicans hold the majority, they use even less empirical discourse when their party controls all branches of government** (a trifecta), turning less to knowledge when fewer constraints on power exist.

5. **Causal support via event study:** Losing or gaining power influences legislators' turn to or away from empirical discourse. We exploit the two power shifts that occurred during the period of our study to conduct an event study. Loss of power causes those who use less empirical discourse to turn to using more of it, and vice versa -- power gain causes a reduction in empirical discourse. This is also true at the individual level for House members who survived reelection but switched from being in the minority to being in the majority or vice versa.

6. **Empirical discourse and bill outcomes:** Although legislative hearings generally contain less empirical discourse than oversight hearings, minority legislators use especially high levels of empirical discourse when opposing majority-sponsored bills, and both majority and minority legislators rely more heavily on empirical argumentation when backing bills that ultimately fail. Empirical discourse is associated with a lower likelihood of bill passage, for both majority and minority sponsored bills.

Overall, these findings suggest that empirical discourse serves as an effective tool for the minority to challenge and thwart majority initiatives. The legislator appears to be working under a "Power or Knowledge" framework, whereby the majority controls the agenda whereas the minority leverages empirical discourse to challenge dominant forces.

---

## 2. Background

### 2.1 Empirical Discourse in Congressional Hearings

The scope of empirical studies of legislative processes is still limited, but has begun to develop in recent years with the advent of text analysis methods. When it comes to the use of evidence by legislatures, previous studies have attributed the varying demand for evidence to several factors: legislative stage, political incentives, and institutional change over time.

**Legislative Stage.** Willems et al (2024) survey Belgian and Canadian politicians and find that they prefer public opinion sources for the initial setting of their agenda -- which issues to tackle -- and prefer expert opinions during policy formulation -- how to tackle these issues. Similarly, Ban et al. (2022) and Mosley and Gibson (2017) observe that empirical discourse is often prioritized in early legislative stages to formulate policy and establish its credibility. As policymaking progresses, other forms of information, such as cost-benefit analyses and personal narratives, become prominent. Trade groups and unions, which can predict public responses and affect a bill's chances of passing, are preferred when specific bills are ripe for consideration. Conversely, witnesses from academia and nonprofits are more common in hearings without specific bills (Ban, Park, and You 2022).

**Political Incentives.** Political calculation also shapes Congress's use of information. First, the selection of witnesses is strategic, as evident in the selection of different witness types for different stages of legislation (Ban, Park, and You 2022). Second, legislators might prioritize political messaging over fact-seeking. Park (2021) finds that this behavior, which she calls "grandstanding," is prevalent among minority party members, non-chair members of prestigious committees, and committees addressing presidential priorities.

Third, the search for evidence varies with political power and level of conflict. Congress is more likely to invite analytical witnesses under unified government conditions (Ban, Park, and You 2022), perhaps because the evidence has less potential to undermine the government when its power is solidified. Ban et al. (2023) complement this finding by showing that bureaucrats are more likely to provide analytical testimony when there is partisan alignment between their agency and the committee chair, particularly under divided government. Esterling (2011) surveyed experts that testified before Congress as part of the legislation of Medicare and showed that witnesses adapted their testimony based on the levels of disagreement they expect to meet. Expecting low confrontation or extreme confrontation reduces their use of empirical statements, while expectations for moderate confrontation lead to the highest use of falsifiable evidence.

Fourth, political polarization significantly affects both the supply and demand for information in Congress. In addition to Ban et al (2023)'s finding regarding the selective sharing of bureaucratic evidence along partisan lines, Aroyehun et al (2024) reveal a decline in evidence-based rhetoric in congressional speeches during periods of heightened polarization (based on 8 million congressional speeches between 1879-2022). Furnas et al (2024), studying data from 1995 to 2021, show that Democrat-controlled committees are nearly twice as likely as Republican-controlled ones to cite scientific research in the policy documents they produce, and often of higher quality (based on citations data).

Caspi and Stiglitz (2023) further illuminate how political incentives shape information use in Congress by showing that legislators' discourse responds strategically to observability. Using evidence from U.S. Senate proceedings, they demonstrate that when legislators' statements are more publicly observable -- such as during high-salience or closely watched debates -- senators are less likely to engage in reasoned, evidence-based argumentation and more likely to adopt messaging-oriented rhetoric. In contrast, lower-observability settings are associated with more deliberative and substantively grounded discourse.

**Temporal changes as they interact with political changes.** The declining role of congressional committees as information-processing hubs adds another dimension. Congress implemented a series of reforms throughout the 1970s (Shaw 1981 and again in the mid 1990s) that greatly impacted its organization, work processes, and available professional support. Lewallen et al (2016) document a significant reduction in witnesses, exploratory hearings, and solution-focused hearings since the 1970s, limiting the range and depth of insights available to Congress. However, Ban et al (2023), relying on newer and more comprehensive datasets, show that the numbers of witnesses actually increased through the 1970s and 1980s. The sharp drop began after the 1995 Contract With America reform that shut down the House's Office for Technology Assessment and laid off a third of all Congressional staff, including many analysts and economists who were the gateway for evidence for their committees. The weakening of the committees is also attributed to stronger party leadership, that pressures the committees to align their hearings and findings with the party's broader agenda, narrowing the scope of information considered (Lewallen, Theriault, and Jones 2016).

**The effects of evidence.** Not much is known about the implications of evidence in congressional hearings. Burstein and Hirsh (2007) examined this question using data from congressional committee and subcommittee hearings on 27 policy proposals from the 1970s--1990s, focusing on testimonies from nearly 1,000 witnesses, including experts, interest group representatives and government officials. They found that only a minority of witness testimonies relied on empirical research (12-16 percentage) and that not all types of evidence affected the probability of legislation. Specifically, only claims that the proposed policy will be effective increased the likelihood of legislation, whereas evidence on the importance of the issue or other aspects had no impact. In contrast, almost any evidence brought by the *opponents* of the policy proved effective in reducing the likelihood of enactment. Importantly, the amount of evidence mattered, too. The more evidence provided by witnesses -- whether in support of the policy or against it -- the *less* the policy was likely to be enacted.

Finally, the empirical competence of legislators, while endogenous, is another factor that likely influences the use and misuse of evidence in legislative processes. Pereira et al (2024) conducted a survey experiment that found that politicians across Western democracies struggle with statistical literacy, exhibiting limited ability to differentiate between correlational and causal evidence or assess sample representativeness.

### 2.2 Text Analysis of Hearing Transcripts

The rise of text analysis and natural language processing (NLP) has revolutionized the study of congressional hearings, enabling researchers to uncover patterns in how legislators engage with empirical discourse. Some studies relied on dictionary-based methods for their simplicity, using predefined keyword sets to identify concepts like "evidence-based arguments." For instance, Ban et al (2022; 2023) employed dictionaries to assess the prevalence of analytical sentences in witness testimony.

Recent advancements, including supervised machine learning and word embedding models, have expanded the analytical toolkit for studying congressional discourse. Supervised models, such as those developed by Park (2021) to calculate "grandstanding scores," leverage labeled datasets to capture subtler distinctions between different types of content. Embedding models like Word2Vec and BERT provide even deeper insights by representing the contextual relationships between words, enabling nuanced assessments such as Aroyehun et al.'s "Evidence Minus Intuition" (EMI) score (2024).

---

## 3. Methods

### 3.1 Data

We assembled a comprehensive dataset of Congressional committee hearings from the U.S. House of Representatives spanning the 105th through the 114th Congresses (1997-2016; N=12,428). The data include the speech of all members of congress who were present at a committee hearing. Each hearing transcript was divided into speech chunks based on speaker attribution to maintain speaker-specific contextual information. We obtained meta-data on each hearing, which includes the general political constellation at the time -- including the President's identity and party, control of the Senate and the House of Representatives, committee control and party ratio, and more. Additionally, we obtained meta-data on individual speakers, including political party, gender, and ranking status in the committee. We also classify each hearing as a legislative or oversight/investigative hearing.

Merging data across hearings and datasets resulted in a loss of 3.14% of the hearings, leaving 12428 unique hearings and 477752 unique speech-chunks. Merging data about bills -- adding information on their sponsorship and fate in Congress and linking individual bills to the hearings that addressed them -- resulted in additional loss of 18.03%.

To prepare the dataset for analysis, each speech chunk was further divided into individual sentences, along with a context window of preceding and succeeding sentences to provide context. This pre-processing step facilitated more granular classification and analysis of the content within the hearings.

**Table 1: Number of Hearings by Year/Congress**

| Year | Congress | Hearings |
|------|----------|----------|
| 1997 | 105 | 175 |
| 1998 | 105 | 94 |
| 1999 | 106 | 410 |
| 2000 | 106 | 391 |
| 2001 | 107 | 410 |
| 2002 | 107 | 380 |
| 2003 | 108 | 566 |
| 2004 | 108 | 496 |
| 2005 | 109 | 657 |
| 2006 | 109 | 546 |
| 2007 | 110 | 1015 |
| 2008 | 110 | 761 |
| 2009 | 111 | 1041 |
| 2010 | 111 | 730 |
| 2011 | 112 | 1102 |
| 2012 | 112 | 750 |
| 2013 | 113 | 830 |
| 2014 | 113 | 653 |
| 2015 | 114 | 842 |
| 2016 | 114 | 579 |
| **Total** | | **12,428** |

### 3.2 Detecting Empirical Discourse in Congressional Speech

We developed a classification procedure to classify sentences into *binary* and *categorical* variables.

**Binary classification.** We classified each sentence as indicating the presence or absence of empirically grounded content. We used the following definition:

> *Empirical evidence is information gathered directly or indirectly through observation or experimentation that may be used to confirm or disconfirm a theory or policy, or to help justify, or establish as reasonable, a person's belief in a given proposition or argument.*

Importantly, empirical statements do not necessarily come in the form of statistics or metrics and need not necessarily be true or validated in the epistemological sense to count as empirical under our scheme. We classified as empirical: descriptions of the empirical world, quests for knowledge on relations of causation and correlation, claims about effects, results, outcomes, and consequences, and historical evidence.

**Categorical classification.** Each empirical sentence was classified into a category specifying the type of empirical content present in a statement out of the following: causal, correlational, descriptive, monetary, statistical, qualitative, historical, other.

**Human Annotation.** The two authors started by annotating randomly selected hearings from the dataset based on the classification procedure and working definitions described above. Then, two research assistants (RAs) were tasked with classifying the same hearings, independently, according to the procedure. This dual-coding process ensured moderate inter-rater reliability (Cohen's Kappa = 0.44). Discrepancies between the RAs' classifications were resolved first by iterative discussions between them and the authors, and then through executive decisions made by the authors, ensuring consistency in the final annotated dataset.

**Classifier Development.** The annotated datasets from the human classification process, spanning several thousand sentences, were split into training and validation sets to develop machine learning classifiers. We used the RoBERTa-base model, a transformer-based language model pre-trained on a large corpus of text, and fine-tuned it using our classified datasets. The fine-tuned model achieved an accuracy rate of 0.91 on a held-out test set (average precision 0.79, recall 0.76, f1-score 0.77).

Using the fine-tuned RoBERTa model, we classified all sentences in the assembled dataset of congressional hearings. We then use the results of this classification, and use a Large Language Model (GPT-4o) to classify empirical sentences into one of the categories (See Appendix B).

**Validation.** We further validate our measure using the ClaimBuster API, a transformer-based tool to detect check-worthiness of statements. We obtain probabilities of check-worthiness on a sample of statements from our dataset (n=2000), and find a moderate positive correlation (.58) with our measure.

**Analysis.** After classifying all sentences in the dataset, we generated an empirical score for each speech chunk within hearing in the dataset, by calculating the share of the empirical evidence mentions of all sentences (hereinafter: empirical score). We used this measure for subsequent analyses of temporal trends, political differences, majority-minority relations, committee types, and more.

**Table 2: Descriptive Statistics of Hearing-Level Empirical Score**

| Statistic | N | Mean | St. Dev. | Median | Min | Max |
|-----------|------|------|----------|--------|-----|-----|
| Empirical Score (Hearing-Level Mean) | 12,428 | 0.110 | 0.054 | 0.103 | 0.000 | 0.428 |

---

## 4. Results

Overall, the use of empirical discourse in congressional committees has moderately increased during the studied period of 1997-2016. Following a slight decline leading to 2006 it has been steadily rising up to the end of the study period in 2016.

### 4.1 Power Dynamics

At first glance, and in line with prior work linking partisanship and evidentiary language (Furnas, LaPira, and Wang 2024), we observe distinct aggregate trends for Republicans and Democrats. Democrats generally rely more heavily on empirical statements, with a notable exception during the 2007--2010 period, when Republicans briefly exceed Democrats in empirical discourse. However, these partisan differences largely mask a more fundamental structural dynamic related to majority and minority status.

Upon closer inspection, variation in empirical discourse is explained to a substantial degree by whether members belong to the majority or minority party. For most of the study period, Republicans controlled the House, with Democrats holding the majority only during the 110th and 111th Congresses (2007--2010). Across parties, minority members consistently exhibit higher rates of empirical discourse than majority members, suggesting that empirical claims function as a strategic resource when formal agenda-setting power is absent.

This pattern is borne out in the regression results. Minority status is associated with a statistically significant and substantively meaningful increase in empirical discourse. In Model 2, minority members score approximately 0.016 points higher on the empirical discourse measure than majority members (p < 0.001). Relative to the baseline empirical score of roughly 0.106, this corresponds to an increase of about 15%. Even in the most conservative specification with member fixed effects (Model 5), minority status remains associated with a 0.010 increase, which corresponds to roughly a 14--15% increase relative to the baseline level of empirical discourse in that model.

**Table 3: The Relationship between Party Affiliation and Minority Status on Rate of Empirical Discourse**

*Dependent variable: Empirical Score*

| Variable | (1) | (2) | (3) | (4) | (5) |
|----------|-----|-----|-----|-----|-----|
| Intercept | 0.108*** | 0.106*** | 0.111*** | 0.082*** | 0.068*** |
| Party (D) | 0.007*** | -0.0001 | 0.001 | 0.002*** | -0.045*** |
| House Control (D) | 0.039*** | 0.045*** | 0.046*** | 0.055*** | 0.055*** |
| Minority Status | | 0.016*** | 0.007*** | 0.007*** | 0.010*** |
| Seniority | | | -0.031*** | -0.038*** | -0.005 |
| Seniority x Minority | | | 0.050*** | 0.047*** | 0.037*** |
| Year Fixed Effects | Yes | Yes | Yes | Yes | Yes |
| Committee Fixed Effects | No | No | No | Yes | Yes |
| Topic Fixed Effects | No | No | No | Yes | Yes |
| Member Fixed Effects | No | No | No | No | Yes |
| Observations | 477,752 | 477,752 | 477,752 | 477,752 | 477,752 |
| R-squared | 0.001 | 0.002 | 0.002 | 0.008 | 0.030 |

*Note: \*p < 0.05; \*\*p < 0.01; \*\*\*p < 0.001. Unit of analysis is the speech chunk. OLS estimates with standard errors in parentheses.*

**The degree of partisan imbalance matters.** Using the majority-minority seat gap as a measure of control asymmetry, we find a strong positive association between the size of the control gap and differences in empirical discourse between minority and majority members. As the majority's seat advantage grows, minority members increasingly outpace majority members in empirical discourse, consistent with the interpretation that evidence is deployed strategically to compensate for diminished institutional leverage.

**Substantive Effects.** Moving from majority to minority status is associated with a 14--15% increase in empirical discourse. Among minority members, an additional unit of seniority further increases empirical discourse by roughly 18%, while senior majority members exhibit comparatively lower levels of empirical discourse. Taken together, these effects imply that a senior minority legislator produces empirical discourse at rates approximately 35--40% higher than a senior majority counterpart.

#### 4.1.1 Event Study

To investigate whether shifts from minority to majority status and vice versa has a causal impact on empirical speech, we conduct an event study around two such shifts of power. Republicans held control of the House of Representatives until losing the 2006 elections (110-111th Congress), before regaining control in the 2010 elections.

The event study shows that both Republicans and Democrats respond to position switching: gaining power (Democrats in 2007; Republicans in 2011) leads to a reduction in empirical discourse (Republicans reacting more strongly). In contrast, losing power (Republicans in 2007; Democrats in 2011) leads to increased empirical discourse (Republicans also reacting more strongly).

### 4.2 Legislative Stage and Legislation Outcomes

We find that legislative hearings are associated with less empirical discussion compared to oversight hearings. Similarly, legislative hearings in more advanced stages, when a specific bill is on the table, contain less empirical discussion compared to hearings where no bill is being considered.

#### 4.2.1 Bill Outcomes

We find a negative association between the prevalence of empirical statements and the chances of a bill to pass and become law. Key patterns in the use of empirical discourse in relation to bill passage:

1. Empirical discourse is more prevalent in legislative initiatives that ultimately failed, compared to those that passed.
2. Minority legislators consistently employ higher levels of empirical discourse than majority legislators across nearly all sponsorship categories.
3. Even majority legislators appear to use more empirical discourse when majority-sponsored bills fail.
4. When examining minority-sponsored legislation, minority members use substantially less empirical discourse in cases of minority bills that pass, compared to the amount they use to oppose majority-sponsored bills.

### 4.3 Variation across Committees and Topics

We find substantial differences across committees and hearing topics with regards to the use of empirical discourse. Appropriations and budget committees, as well as science and commerce committees, are the more "heavy users" of empirical discourse; while rules, administration, and judiciary committees are on the farther end of the spectrum. Energy, environment, health and agriculture are the topics most associated with empirical discourse; while civil rights hearings are the least empirically informed.

---

## 5. Discussion

In this study, we analyzed the use of empirical discourse in U.S. House of Representatives committee hearings from 1997 to 2016 (N = 12,428 hearings; over 5.8 million sentences) and uncovered several layers of strategic evidence use by legislators. We found that minority party members, whether Democrat or Republican, consistently employed more empirical discourse than their majority counterparts, and this tendency grew stronger as the majority-minority power gap widened. Senior minority members were especially likely to rely on empirical statements, suggesting that seniority within the minority amplifies the use of knowledge as a compensatory tool for power they otherwise lack. Our event study analyses of the 2007 and 2011 flips in House control demonstrated that legislators increase their use of empirical argumentation when moving into the minority and reduce it when gaining majority status, highlighting a causal link between power loss and reliance on empirical statements.

Overall, our findings suggest that house members operate under a "Power or Knowledge" scheme, whereby those in the majority control the committee agenda and can affect policy outcomes and leverage political exposure in their favor. Members of the minority, left to find alternative ways of influence, turn to wield factual statements as a way of challenging the dominant forces.

While our findings suggest that the use of empirical discourse has risen over time, this may be driven more by partisan politics than by an increase demand for evidence; that is, as the House's ability to carry out legislative actions has declined due to increased partisanship, the minority -- mostly Democrats, but at a time also Republicans -- were in stronger need for the knowledge when they were lacking in power.

---

## 6. Conclusion and Future Research

This study suggests that empirical discourse in congressional hearings serves both as a tool for informed policymaking and as a strategic advantage -- the weapon of the powerless. Minority party members, particularly those with seniority, rely more on empirical discourse, suggesting that factual statements compensate for lack of formal legislative power.

Future research can address these and other questions. For example, by using metrics like NOMINATE that measure the level of extremity or moderation of each House representative over time, we could test whether empirical discourse is used more by moderates or extremists and whether it helps or thwarts legislation backed by each of these camps.

Finally, politics is constantly adapting and changing. Our dataset ends in 2016, before the covid pandemic and the Presidencies of Trump, Biden, and Trump again, each taking very different orientations towards facts, knowledge, and power. Future research can look into these changes and compare the usage of empirical discourse across and between different periods in American politics.

---

## Appendix A: Classifying Empirical Content

**Empirical Binary:** Indicates the presence or absence of empirically grounded content. Definition:

> *Empirical evidence is information gathered directly or indirectly through observation or experimentation that may be used to confirm or disconfirm a theory or policy, or to help justify, or establish as reasonable, a person's belief in a given proposition or argument.*

Important clarifications from the annotation guidelines:
- Empirical evidence may regard specific events, factors and causes, or more generally relationships between factors such as causality or correlation.
- Abstract statements such as "where there is smoke there is fire" are not included.
- Statements about research to establish facts or descriptions of reality are included.
- Testimonial statements about the experience of a single person or a handful of people are not included, but large groups may be included.

**Empirical Category:** Specifies the type of empirical content present in a statement in a non-exclusive way (i.e. more than one category). Categories:

- **Monetary:** Facts about the economy, budgets, or costs. *Example: "The costs of Shuttle missions to service Hubble have never been charged to the science program."*
- **Causal:** Statements showing cause-and-effect relationships. *Example: "We did so because veterans consistently told us their evidence of toxic exposures was being minimized or ignored."*
- **Descriptive:** Factual descriptions of situations, events, or populations. *Example: "There are medical organizations now who are treating tens of thousands of people overdosed with chemicals."*
- **Qualitative:** Non-quantitative observations, interviews, or narratives. *Example: "Diverse array of symptoms including fatigue, skin rash, headache, muscle and joint pain, memory problems, shortness of breath, sleep disturbances, gastrointestinal symptoms, and chest pain."*
- **Historical:** References to past events or timelines. *Example: "1995 is the point at which you began to ask our troops if they were exposed to chemicals."*
- **Statistical:** Use of statistics, numbers, or comparisons. *Example: "Have you encountered -- in your study dealing with these servicemen and women, have you encountered in other instances that you might think would be the higher percentage than average population?"*
- **Other:** If the evidence does not fit into the above categories.

## Appendix B: Empirical Categories

We classify a randomly generated sample of empirical sentences, derived from our dataset, using OpenAI's GPT-4o model, using a prompt instructing the model to help in analyzing congressional hearings to identify empirical evidence. The distribution of categories shows descriptive evidence is the most common (~45%), followed by historical (~19%), monetary (~17%), statistical (~11%), causal (~7%), and qualitative (~2%).

## Appendix D: Bill Sponsorship

We extract information about all bills that were introduced in Congress within our study period regarding their sponsors and co-sponsors. We construct a *bipartisanship intensity* measure defined as the proportion of bill supporters (sponsor and cosponsors) who are affiliated with the chamber's minority party:

> Bipartisanship Intensity = (Number of minority party cosponsors + indicator if sponsor is minority) / (Total number of sponsors and cosponsors)

Categories:
- **Partisan majority:** bipartisanship intensity <= 0.10
- **Bipartisan:** bipartisanship intensity between 0.45 and 0.55
- **Partisan minority:** bipartisanship intensity >= 0.90
- **Other:** bills that do not fall within these thresholds

## Appendix F: The Impact of Full Trifecta Control

A party in control of all political branches of government (President, Senate and House of Representatives), called trifecta, wields enormous power. For Democrats, holding the majority in the house leads to less empirical discourse, yet the size of the effect is smaller under full control of government. Republicans, while also utilizing empirical discourse less when in the majority, do so less often when they hold a trifecta, suggesting they are trading more facts for more power.

---

## References

- Aroyehun, S. T. et al. (2024). "Computational Analysis of US Congressional Speeches Reveals a Shift from Evidence to Intuition." *arXiv preprint*.
- Ban, Pamela, Ju Yeon Park, and Hye Young You (2022). *Bureaucrats in Congress: Strategic Information Sharing in Policymaking.*
- Ban et al. (2023). "How Are Politicians Informed? Witnesses and Information Provision in Congress." *American Political Science Review* 117.1, pp. 122--140.
- Bimber, Bruce Allen et al. (1996). *The politics of expertise in Congress: The rise and fall of the Office of Technology Assessment.* SUNY Press.
- Burstein, Paul and C. Elizabeth Hirsh (2007). "Interest Organizations, Information, and Policy Innovation in the U.S. Congress." *Sociological Forum* 22.2, pp. 273--299.
- Caspi, Aviv and Edward Stiglitz (2023). "Observability and Reasoned Discourse: Evidence from the U.S. Senate." Working paper, SSRN.
- Esterling, Kevin M. (2011). "Deliberative Disagreement in U.S. Health Policy Hearings." *Legislative Studies Quarterly* 36.2, pp. 169--198.
- Furnas, Alexander, Timothy Michael LaPira, and Dashun Wang (2024). *Partisan Disparities in the Use of Science in Policy.*
- Heitshusen, Valerie (2017). *Senate Committee Hearings: Arranging Witnesses.*
- Jones, Bryan D. et al. (2023). *Policy Agendas Project: Congressional Hearings.*
- Lewallen, Jonathan, Sean M. Theriault, and Bryan D. Jones (2016). "Congressional dysfunction: An information processing perspective." *Regulation & Governance* 10.2, pp. 179--190.
- McGrath, Robert J. (2013). "Congressional Oversight Hearings and Policy Control." *Legislative Studies Quarterly* 38.3, pp. 349--376.
- Meng, Kevin et al. (2024). "Gradient-Based Adversarial Training on Transformer Networks for Detecting Check-Worthy Factual Claims." *ACM Trans. Intell. Syst. Technol.* 15.6.
- Mosley, Jennifer E. and Katherine Gibson (2017). "Strategic use of evidence in state-level policymaking: matching evidence type to legislative stage." *Policy Sciences* 50.4, pp. 697--719.
- Oleszek, Walter J. (1989). *Congressional Procedures and the Policy Process.* Washington, DC: CQ Press.
- Park, Ju Yeon (2021). "When Do Politicians Grandstand? Measuring Message Politics in Committee Hearings." *The Journal of Politics* 83.1, pp. 214--229.
- Pereira, Miguel M. et al. (2024). *Politicians and Evidence.*
- Shaw, Malcolm (1981). "Congress in the 1970s: A Decade of Reform." *Parliamentary Affairs* 34.3, pp. 272--290.
- Shepsle, Kenneth A. and Barry R. Weingast (1987). "The Institutional Foundations of Committee Power." *American Political Science Review* 81.1, pp. 85--104.
- Suzgun, Mirac, Tal Gur, Federico Bianchi, et al. (2025). "Language models cannot reliably distinguish belief from knowledge and fact." *Nature Machine Intelligence* 7, pp. 1780--1790.
- Willems, E., B. Maes, and S. Walgrave (2024). "Mechanisms of Political Responsiveness: The Information Sources Shaping Elected Representatives' Policy Actions." *Political Research Quarterly* 77.3, pp. 851--865.
