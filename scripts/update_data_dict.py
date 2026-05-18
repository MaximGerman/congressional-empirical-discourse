import os
import sys

# Ensure the project root is in sys.path to resolve cross-module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pyarrow.parquet as pq

from scripts.optimize_data import PARQUET_PATH

DICT_PATH = "docs/project/data_dictionary.md"


def generate_summary_table(parquet_path: str) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(parquet_path)
    columns = parquet_file.schema.names
    total_rows = parquet_file.metadata.num_rows

    stats = []
    for col in columns:
        # Load only this single column from the parquet file
        col_df = pd.read_parquet(parquet_path, columns=[col])
        series = col_df[col]

        dtype = str(series.dtype)
        null_count = series.isnull().sum()
        null_pct = (null_count / total_rows) * 100
        unique_vals = series.nunique()

        # Get a few sample values
        sample = series.dropna().unique()[:3].tolist()
        sample_str = ", ".join([str(x) for x in sample])

        stats.append(
            {
                "Column": f"`{col}`",
                "Type": dtype,
                "Null %": f"{null_pct:.1f}%",
                "Unique": f"{unique_vals:,}",
                "Samples": sample_str,
            }
        )

    return pd.DataFrame(stats)


def update_dictionary():
    if not os.path.exists(PARQUET_PATH):
        print(f"Skipping update: {PARQUET_PATH} not found.")
        return

    print(f"Updating data dictionary from {PARQUET_PATH}...")
    summary_df = generate_summary_table(PARQUET_PATH)

    markdown_table = summary_df.to_markdown(index=False)

    with open(DICT_PATH, encoding="utf-8") as f:
        content = f.read()

    marker = "## Automated Schema Summary"

    # Check if the table content actually changed before writing
    # We compare the table part specifically to avoid timestamp-only updates
    if marker in content:
        base_content = content.split(marker)[0].strip()
        existing_table = content.split(marker)[1].split("\n\n", 1)[-1].strip()
        if existing_table == markdown_table.strip():
            print("No changes detected in schema. Skipping update.")
            return
    else:
        base_content = content.strip()

    header = f"\n\n{marker}\n*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    new_content = base_content + header + markdown_table + "\n"

    with open(DICT_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully updated {DICT_PATH}")


if __name__ == "__main__":
    update_dictionary()
