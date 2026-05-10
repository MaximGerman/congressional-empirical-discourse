# Kickoff Meeting Notes -- Zoom with Amit Haim

*Transcribed and translated from: `Maxim German's Zoom Meeting.pdf` (Hebrew/English, auto-generated Zoom summary)*

---

## Quick Recap

The meeting focused on a research project analyzing the use of empirical information in U.S. Congressional hearings. Amit presented their current work, which included training a RoBERTa model on 13,000 Congressional hearings between 1997--2015, resulting in a 58% correlation with empirical sentence identification using the ClaimBuster instrument. The team discussed possible next steps, including either improving the existing model's performance or expanding the analysis to include data from 2015--2025, which would require working with Congressional IPA data or using an existing dataset. Maxim and Ilay expressed interest in the project and asked questions about data annotation requirements and methodology. The discussion also touched on the possibility of incorporating video data analysis and using AI tools like Claude for labeling tasks. Shimon (Dr. Moni Shahar) emphasized the importance of producing publishable research for career development and encouraged the team to aim for high-quality scientific work.

## Next Steps

- **Amit:** Send the presentation, paper, dataset, and relevant project materials to Maxim and Ilay (via shared Google Drive folder).
- **Maxim and Ilay:** Review the supplied paper, dataset, and materials shared by Amit.
- **Maxim and Ilay:** Discuss internally and decide on the project direction (e.g., improving model performance, expanding the dataset to 2025, or exploring multimodal/video analysis).
- **Maxim and Ilay:** Submit an initial project proposal to Amit (and possibly also to Shimon) for feedback before the final submission.
- **Amit:** Be available for questions, suggestions, and advice as Maxim and Ilay progress.
- **Amit:** Arrange for research assistants to label additional data samples if Maxim and Ilay find the current dataset too thin (contingent on their request, with an estimated timeline of one to two weeks).

## Detailed Summary

### Deep Learning and NLP Workshop

The meeting focused on a workshop discussion and a potential project involving deep learning and NLP. Shimon explained the purpose of the workshop, noting that it is not part of the course team and participants need to pay attention to submission deadlines in July. Maxim shared his background in deep learning and LLM degrees and mentioned his experience with NLP courses taught by Nadav Cohen. Amit planned to present the project details and discuss next steps with Maxim, and offered to share his screen to better present the information.

### Empirical Data in Congressional Hearings

Amit presented a research project examining the use of empirical information in U.S. Congressional hearings, focusing on legislative discussions from 1997 to 2015. The project, which analyzed approximately 13,000 hearings resulting in 6 million sentences, found that minority party members, including Republicans and Democrats when in the minority, tend to use empirical data more compared to their majority counterparts. Amit noted that when a bill is on the table, discussions become less empirical as legislators focus more on passing legislation rather than data-based arguments.

### Language Model Training Methodology

Amit discussed their methodology for training a language model, explaining that they started with a simple approach using a random sample of hearings divided into paragraphs per speaker, resulting in approximately 2,000 labeled sentences. They reported achieving performance that meets requirements and later validated the model using the ClaimBuster instrument, finding a 58% correlation. The team concluded that while performance was adequate, using an LLM was not worth it for their specific needs.

### RoBERTa Analysis Presentation

Amit presented an analysis using RoBERTa on a full dataset, including party affiliation and regression studies. Shimon highlighted the project's advantage of having a baseline and suggested it as an opportunity to improve RoBERTa training without introducing new scientific concepts. Amit proposed two options: providing data and labels for better results or focusing on action, and noted that their data ended in 2015--2016.

### Trump Election Discourse Research Project

Amit and Shimon discussed a potential research project examining changes in discourse in the decade after Trump's election. Shimon expressed interest in the academic aspects of the project and suggested using Claude to streamline technical work, noting that previous projects were burdened with unnecessary technical tasks. Shimon recommended continuing with both technical and academic components, emphasizing that the project could answer interesting questions about language models and discourse changes. Maxim indicated his agreement to continue the project.

### Congressional Data Collection Methods

Amit discussed working with transcripts and videos from Congress from the last 15 years, and mentioned the possibility of exploring video scripts despite his experience being limited to text. Shimon expressed interest in the multimodal approach but noted its complexity. Amit explained two methods for data collection: working directly with the Congressional IPA, which has some errors, or using an existing project that already processed the data. The team received a link to a relevant project called **bicam.net**.

### Dataset Discussion Meeting

The team discussed a dataset containing approximately 2,000 labeled sentences from ten hearings spanning 1997--2015, which Amit and his colleagues used to define empirical discourse and proposals. Maxim expressed concerns about the effort required for annotation and data processing, while Shimon suggested using existing labeled samples for training and testing the model, with the option to expand the dataset if needed. The group agreed to share the dataset via Google Drive, and Shimon emphasized the importance of producing a high-quality scientific paper from this work, which could benefit team members' career prospects.
