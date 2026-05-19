import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import pickle
import sys
import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run predictions on a dataset.")
    parser.add_argument("--sample_size", type=float, default=1.0, help="Percentage of the dataset to sample (e.g., 0.01 for 1%).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing predictions.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save progress checkpoints.")
    args = parser.parse_args()

    # Set up paths
    PROJECT_ROOT = Path("C:\\Users\\mitha\\Documents\\empirical_evidence_corren")

    def project_path(*args):
        return PROJECT_ROOT.joinpath(*args)

    # Specify the base model used for training
    base_model = "roberta-base"  # Replace with the model you used

    # Define model directory
    model_directory = project_path("hearings_testing", "model_output", "checkpoint-760")

    if not model_directory.exists():
        print(f"Error: Model directory '{model_directory}' does not exist.")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Save the tokenizer files to your checkpoint directory
    tokenizer.save_pretrained(model_directory)

    print("Loading model and tokenizer from the checkpoint...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        sys.exit(1)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load data
    data_file = project_path("data", "merged_hearings.pkl")
    if not data_file.exists():
        print(f"Error: Data file '{data_file}' does not exist.")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_pickle(data_file)

    if 'target_sentence' not in df.columns:
        print("Error: 'target_sentence' column is missing in the dataset.")
        sys.exit(1)

    # Apply sampling if specified
    if args.sample_size < 1.0:
        sample_size = max(1, int(len(df) * args.sample_size))
        print(f"Sampling {sample_size} rows ({args.sample_size * 100:.2f}% of the dataset)...")
        df = df.sample(n=sample_size, random_state=42)

    # Initialize checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if available
    checkpoint_file = checkpoint_dir / "predictions_checkpoint.pkl"
    if checkpoint_file.exists():
        print("Resuming from checkpoint...")
        with open(checkpoint_file, "rb") as file:
            checkpoint = pickle.load(file)
        predictions = checkpoint.get("predictions", [])
        probabilities = checkpoint.get("probabilities", [])
        start_idx = checkpoint.get("start_idx", 0)
    else:
        predictions = []
        probabilities = []
        start_idx = 0

    # Process in batches
    print("Performing predictions in batches...")
    batch_size = args.batch_size
    for batch_start in range(start_idx, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        print(f"Processing batch {batch_start} to {batch_end}...")
        batch_texts = df['target_sentence'].iloc[batch_start:batch_end].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()

        predictions.extend(batch_predictions)
        probabilities.extend(batch_probabilities)

        # Save checkpoint
        checkpoint = {
            "predictions": predictions,
            "probabilities": probabilities,
            "start_idx": batch_end
        }
        with open(checkpoint_file, "wb") as file:
            pickle.dump(checkpoint, file)

    # Save predictions to DataFrame
    print("Saving predictions to DataFrame...")
    df['predictions'] = predictions
    df['probabilities'] = probabilities

    # Save the updated DataFrame to a pickle file
    output_file = project_path("data", "merged_hearings_roberta_predict.pkl")
    print(f"Saving updated DataFrame to '{output_file}'...")
    try:
        with open(output_file, "wb") as file:
            pickle.dump(df, file)
        print("Process completed successfully.")
    except Exception as e:
        print(f"Error saving the output file: {e}")
        sys.exit(1)

    # Remove checkpoint after successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()

if __name__ == "__main__":
    main()
