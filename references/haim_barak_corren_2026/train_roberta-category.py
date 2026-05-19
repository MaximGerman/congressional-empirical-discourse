import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pathlib import Path
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SpeechDataset(Dataset):
    def __init__(self, tokenizer, speeches, labels, max_length=512):
        self.encodings = tokenizer(speeches, truncation=True, padding=True, max_length=max_length)
        self.labels = labels  # Multi-label one-hot encoded

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(file_path, tokenizer, args):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    speeches = df['target_sentence'].tolist()  # Column with speeches
    # Extract multi-labels from the appropriate columns
    labels = df[['category_historical', 'category_correlational', 'category_causal',
                 'category_monetary', 'category_qualitative', 'category_testimonial',
                 'category_statistical', 'category_descriptive']].values.tolist()
    dataset = SpeechDataset(tokenizer, speeches, labels)
    return dataset

def evaluate_model(args, model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.sigmoid(logits)  # Apply sigmoid for multi-label
        all_preds += predictions.cpu().tolist()
        all_labels += batch['labels'].cpu().tolist()

    # Convert predictions to binary using a threshold of 0.5
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    binary_labels = np.array(all_labels)

    # Calculate metrics
    f1 = f1_score(binary_labels, binary_preds, average="macro")
    return f1

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_dataset = load_dataset(args.train_file, tokenizer, args)
    test_dataset = load_dataset(args.test_file, tokenizer, args)

    # Update the model for multi-label classification
    num_labels = 8  # Number of categories
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_labels)
    model.config.problem_type = "multi_label_classification"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=args.logging_dir,
        report_to="none",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    f1 = evaluate_model(args, model, test_dataset)
    print(f"Test F1 Score: {f1}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="classifier_categories/train.csv")
    parser.add_argument("--test_file", type=str, default="classifier_categories/test.csv")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)


