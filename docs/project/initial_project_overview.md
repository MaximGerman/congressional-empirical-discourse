# Project Overview: Temporal Robustness and Domain Adaptation in Congressional Empirical Discourse

*Transcribed from: `gemini-code-1778318489222.md` (early project planning notes)*

---

## 1. Project Overview

This project builds upon existing research analyzing the use of empirical evidence in U.S. House Committee Hearings (1997--2015). The original research established a baseline using a RoBERTa model. Our project aims to modernize this pipeline, expand the analysis to the modern political era (2015--2025), and extract deeper categorical insights.

To achieve an ambitious scope within the constraints of limited manual labeling capacity and single-GPU compute, this project will leverage modern deep learning techniques including **Knowledge Distillation (LLM-as-a-labeler)**, **Unsupervised Domain Adaptation (Continual Pre-training)**, and **Multi-Task Learning**.

---

## 2. The Gameplan (Execution Steps)

### Phase 1: Baseline Replication & Modernization
- **Task:** Establish a fair comparison between the original architecture and modern encoder models.
- **Action:** Train the original `RoBERTa-base` and a modern `DeBERTa-v3-base` on the existing 1997--2015 labeled dataset (`RA_merged_with_agreement.csv`).
- **Goal:** Demonstrate improved F1/Accuracy using a modernized architecture on the same legacy distribution.

### Phase 2: Zero-Shot Expansion (Overcoming the Labeling Gap)
- **Task:** Expand the dataset to the 2015--2025 era without manual data tagging.
- **Action:** Use the `bicam` Python package to ingest modern transcripts. Deploy a locally hosted, quantized LLM (`Llama-3-8B-Instruct` via 4-bit quantization) to generate "Silver Labels" for a representative sample (~10,000 sentences) based on the definitions from the original paper.
- **Validation:** Calculate a "Trust Score" (Cohen's Kappa) by running the LLM on a subset of the legacy human-labeled data to scientifically justify the silver labels.

### Phase 3: Unsupervised Domain Adaptation
- **Task:** Bridge the linguistic and structural gap between the 1997--2015 data and the newer BICAM data.
- **Action:** Perform Masked Language Modeling (MLM) Continual Pre-training on a combined corpus (the 500MB unlabeled `df_withempirical` file + new BICAM text) using the `DeBERTa-v3` model.
- **Goal:** Teach the model the specific vocabulary and formatting shifts of the modern era prior to classification fine-tuning.

### Phase 4: Multi-Task Expansion (Optional/Stretch Goal)
- **Task:** Move beyond binary classification (Empirical: Yes/No).
- **Action:** Modify the DeBERTa model head to perform Multi-Task Learning (MTL), simultaneously predicting if a statement is empirical *and* what category of evidence it represents (e.g., statistical, anecdotal).

---

## 3. Workshop Proposal Points (Due May 23)

- **Team:** Maxim German, Ilay
- **End Product:** A robust, automated NLP pipeline capable of ingesting raw, modern congressional transcripts, standardizing the text, and outputting classified empirical statements with their corresponding evidence categories.
- **Training and Inference Schemes:**
  - *Training (Supervised):* Fine-tuning `DeBERTa-v3` with a standard classification head (and potentially a multi-task head) using PyTorch/HuggingFace Trainer.
  - *Training (Unsupervised):* Continual pre-training via Masked Language Modeling (MLM) to address covariate shift.
  - *Inference:* Batch inference using a 4-bit quantized `Llama-3-8B` for knowledge distillation/silver labeling.
- **Datasets:**
  - `RA_merged_with_agreement.csv`: Legacy labeled dataset (1997-2015).
  - `df_withempirical`: 500MB unlabeled legacy corpus (1997-2015).
  - *BICAM Dataset:* Newly scraped data for the 2015-2025 era.
- **Compute and Storage Requirements:**
  - **Compute:** CS Department GPU Cluster (Slurm). A single GPU is sufficient. High VRAM requirements for the LLM are mitigated via 4-bit quantization (~5.5GB VRAM footprint).
  - **Storage:** Personal directory under `/home/yandex/DLWorkShop2026b` to house the 500MB corpus, newly scraped BICAM data, and saved model checkpoints.
- **Third-Party Tools and Models:**
  - **Models:** `RoBERTa-base`, `DeBERTa-v3-base`, `Llama-3-8B-Instruct`.
  - **Libraries:** HuggingFace `transformers`, `datasets`, `bitsandbytes` (or `unsloth`) for quantization, `vLLM`/`Ollama` for high-throughput inference, and `bicam` for data scraping.
- **Existing Solutions & Related Work:**
  - Building directly upon the methodology presented in *"In This House We Believe in... Empirical Evidence? Power and Knowledge Dynamics in Congressional Hearings"* (Haim & Barak-Corren).
  - Literature regarding LLM-as-a-judge, weak supervision, and unsupervised domain adaptation in NLP.
