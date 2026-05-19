from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Starting classification job...")

# =======================================================
# 0. Paths
# =======================================================
path = r"C:\Users\USER\OneDrive - Tel-Aviv University\Documents\empirical_evidence_corren"
PROJECT_ROOT = Path(path)

def project_path(*args):
    return PROJECT_ROOT.joinpath(*args)

sentences_file = project_path("hearings_testing", "all_sentences_with_meta.csv")
predictions_file = project_path("hearings_testing", "sentence_predictions_all.csv")

# =======================================================
# 1. Load sentence data + FILTER (member + House)
# =======================================================
df_sentences_all = pd.read_csv(sentences_file, low_memory=False)
print("Loaded sentences:", len(df_sentences_all))

def clean_chamber(ch):
    if not isinstance(ch, str):
        return "Other"
    ch = ch.lower().strip()

    if "joint" in ch:
        return "Joint"
    if "commission" in ch:
        return "Commission"
    if "legislative branch" in ch:
        return "Legislative Branch"

    if "house" in ch and "senate" not in ch:
        return "House"
    if "senate" in ch and "house" not in ch:
        return "Senate"

    return "Other"

df_sentences_all["chamber_clean"] = df_sentences_all["chamber"].apply(clean_chamber)

# Filter
df_to_classify = df_sentences_all[
    (df_sentences_all["speaker_type"] == "member") &
    (df_sentences_all["chamber_clean"] == "House")
].copy()

print("Filtered (member + House):", len(df_to_classify))

# Clean sentence
df_to_classify = df_to_classify[df_to_classify["sentence"].notna()].copy()
df_to_classify["sentence"] = df_to_classify["sentence"].astype(str).str.strip()
df_to_classify = df_to_classify[df_to_classify["sentence"] != ""].copy()
df_to_classify = df_to_classify.reset_index(drop=True)

print("After cleaning sentence:", len(df_to_classify))
print("Sample sentences:", df_to_classify["sentence"].head(5).tolist())

# =======================================================
# 2. Load model & tokenizer
# =======================================================
model_directory = project_path("hearings_testing", "model_output", "checkpoint-760")
print("Model directory:", model_directory)

tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)
model.eval()

id2label = getattr(model.config, "id2label", None)

# =======================================================
# 3. Classification helper
# =======================================================
def classify_sentences(
    df: pd.DataFrame,
    text_column: str = "sentence",
    batch_size: int = 32,
    max_length: int = 256,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    print("Classifying", n, "sentences")

    if n == 0:
        return df

    df["pred_label_id"] = None
    df["pred_label"] = None
    df["pred_confidence"] = None

    softmax = torch.nn.Softmax(dim=-1)
    n_batches = (n + batch_size - 1) // batch_size

    for start in tqdm(range(0, n, batch_size), total=n_batches, desc="Batches"):
        end = min(start + batch_size, n)
        batch_texts = df[text_column].iloc[start:end].astype(str).tolist()

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = softmax(logits)

        batch_pred_ids = probs.argmax(dim=-1).cpu().tolist()
        batch_pred_probs = probs.max(dim=-1).values.cpu().tolist()

        if id2label:
            batch_pred_labels = [id2label[i] for i in batch_pred_ids]
        else:
            batch_pred_labels = [str(i) for i in batch_pred_ids]

        df.iloc[start:end, df.columns.get_loc("pred_label_id")] = batch_pred_ids
        df.iloc[start:end, df.columns.get_loc("pred_label")] = batch_pred_labels
        df.iloc[start:end, df.columns.get_loc("pred_confidence")] = batch_pred_probs

    return df

# =======================================================
# 4. Run classification & save (OVERWRITE)
# =======================================================
df_classified = classify_sentences(df_to_classify, text_column="sentence", batch_size=32)

df_classified.to_csv(predictions_file, index=False)
print("\n✅ Saved predictions to:")
print(predictions_file)
