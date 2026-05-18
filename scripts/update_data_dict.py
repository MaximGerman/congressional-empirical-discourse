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

    heavy_cols = {"text", "target_sentence", "context_before", "context_after"}
    small_cols = [col for col in columns if col not in heavy_cols]

    # Read all small columns in a single optimized batch
    if small_cols:
        small_df = pd.read_parquet(parquet_path, columns=small_cols, engine="pyarrow", dtype_backend="pyarrow")
    else:
        small_df = pd.DataFrame()

    stats = []
    for col_idx, col in enumerate(columns):
        if col in small_df.columns:
            series = small_df[col]
            dtype = str(series.dtype)
            null_count = int(series.isnull().sum())
            unique_vals = int(series.nunique())
            sample = series.dropna().unique()[:3].tolist()
        else:
            # For heavy text columns, fetch stats from metadata and slice for samples
            arrow_field = parquet_file.schema.to_arrow_schema().field(col)
            dtype = str(arrow_field.type)

            null_count = 0
            has_stats = True
            for rg in range(parquet_file.num_row_groups):
                col_meta = parquet_file.metadata.row_group(rg).column(col_idx)
                if col_meta.statistics and col_meta.statistics.null_count is not None:
                    null_count += col_meta.statistics.null_count
                else:
                    has_stats = False
                    break

            if not has_stats:
                sample_table = parquet_file.read_row_group(0, columns=[col])
                sample_series = sample_table.to_pandas()[col]
                null_count = int(sample_series.isnull().sum() * (total_rows / len(sample_series)))

            # Load only the first row group for sample preview
            sample_table = parquet_file.read_row_group(0, columns=[col])
            sample_series = sample_table.to_pandas()[col]

            unique_vals = total_rows - null_count
            sample = sample_series.dropna().unique()[:3].tolist()

        null_pct = (null_count / total_rows) * 100
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

    marker = "## Automated Schema Summary"

    # Shortcut: If data_dictionary.md is newer than the Parquet file, skip execution
    if os.path.exists(DICT_PATH):
        pq_mtime = os.path.getmtime(PARQUET_PATH)
        dict_mtime = os.path.getmtime(DICT_PATH)
        if dict_mtime >= pq_mtime:
            with open(DICT_PATH, encoding="utf-8") as f:
                if marker in f.read():
                    print("Data dictionary is already up-to-date with parquet file. Skipping update.")
                    return

    print(f"Updating data dictionary from {PARQUET_PATH}...")
    summary_df = generate_summary_table(PARQUET_PATH)

    markdown_table = summary_df.to_markdown(index=False)

    with open(DICT_PATH, encoding="utf-8") as f:
        content = f.read()

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
