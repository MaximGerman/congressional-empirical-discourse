import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CSV_PATH = "data/sentences_enriched.csv"
PARQUET_PATH = "data/sentences_enriched.parquet"
CHUNK_SIZE = 250000  # Process in chunks to keep memory usage low


def convert_csv_to_parquet(progress_callback=None):
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return False

    print(f"Optimizing {CSV_PATH} for dashboard performance...")

    # Estimate total rows for progress bar
    print("Estimating file size...")
    try:
        with open(CSV_PATH, "rb") as f:
            total_rows = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b"")) - 1
    except Exception:
        # Fallback to slower line counting if needed
        with open(CSV_PATH, encoding="utf-8") as f:
            total_rows = sum(1 for _ in f) - 1
    print(f"Total rows to process: {total_rows:,}")

    # Define types to ensure consistency and minimize memory
    dtype = {
        "congress": "int16",
        "chamber": "category",
        "party": "category",
        "match_type": "category",
        "dem": "Int8",
        "minority": "Int8",
        "unified": "Int8",
        "minuni": "Int8",
        "freshman": "Int8",
        "chairspeech": "Int8",
        "rankmemspeech": "Int8",
        "leader": "Int8",
        "member_state": "category",
    }

    writer = None

    # Use chunksize to keep memory usage low and constant
    reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, dtype=dtype)

    total_chunks = (total_rows // CHUNK_SIZE) + 1
    for i, chunk in enumerate(tqdm(reader, total=total_chunks)):
        # Convert date column if it exists
        if "hearing_date" in chunk.columns:
            chunk["hearing_date"] = pd.to_datetime(chunk["hearing_date"])

        # Rename target_sentence to text if needed
        if "text" not in chunk.columns and "target_sentence" in chunk.columns:
            chunk = chunk.rename(columns={"target_sentence": "text"})

        # Initialize the Parquet writer with the schema of the first chunk
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(PARQUET_PATH, table.schema, compression="snappy")

        writer.write_table(table)

        # Report progress if callback provided
        if progress_callback:
            progress_callback((i + 1) / total_chunks)

    if writer:
        writer.close()

    csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024)
    pq_size = os.path.getsize(PARQUET_PATH) / (1024 * 1024)

    print("\nOptimization Complete!")
    print(f"  CSV Size:     {csv_size:.1f} MB")
    print(f"  Parquet Size: {pq_size:.1f} MB ({(pq_size / csv_size * 100):.1f}% of original)")
    print(f"  Saved to:     {PARQUET_PATH}")
    return True


if __name__ == "__main__":
    convert_csv_to_parquet()
