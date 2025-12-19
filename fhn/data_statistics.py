import pandas as pd
import glob
import os
from typing import List

def get_col_counts(files: List[str], target_col="symbol"):
    """
    This function counts occurrences of values in a target column across multiple files.

    Parameters:
    - files: list of file paths to CSV files
    - target_col: name of the column to count values from

    Returns:
    - symbols_set: set of unique values found in the target column
    - counts_df: dataframe of value counts per file
    """
    counts_list = []
    symbols_set = set()

    for f in files:
        df = pd.read_csv(f)
        counts = df[target_col].value_counts()
        counts_list.append(counts)
        symbols_set.update(counts.index)

    # change into dataframe
    symbols = sorted(symbols_set)
    counts_df = pd.DataFrame(0, index=[os.path.basename(f) for f in files], columns=symbols)

    for i, counts in enumerate(counts_list):
        for sym, val in counts.items():
            counts_df.loc[counts_df.index[i], sym] = val

    return symbols_set, counts_df


def save_filtering_summary(
    df_raw,
    df_filt,
    df_balanced,
    loss_threshold,
    r2_threshold,
    output_folder,
    filename="filtering_summary.csv"
):
    os.makedirs(output_folder, exist_ok=True)

    datasets = {
        "raw": df_raw,
        "filtered": df_filt,
        "balanced": df_balanced
    }

    rows = []

    for name, df in datasets.items():
        total_rows = len(df)

        # Extract metrics
        L = df["loss"].dropna()
        R = df["r2"].dropna()

        # Apply validity rules
        L = L[(L >= 0) & (L <= (loss_threshold if name != "raw" else 5))]
        R = R[(R >= (r2_threshold if name != "raw" else 0)) & (R <= 1)]

        rows.append({
            "dataset": name,
            "metric": "loss",
            "total_rows": total_rows,
            "valid_rows": len(L),
            "mean": L.mean(),
            "median": L.median()
        })

        rows.append({
            "dataset": name,
            "metric": "r2",
            "total_rows": total_rows,
            "valid_rows": len(R),
            "mean": R.mean(),
            "median": R.median()
        })

    summary_df = pd.DataFrame(rows)

    out_path = os.path.join(output_folder, filename)
    summary_df.to_csv(out_path, index=False)

    return summary_df