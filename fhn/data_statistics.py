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