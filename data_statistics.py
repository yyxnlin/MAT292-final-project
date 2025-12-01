import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from typing import List

def get_col_counts(files: List[str], target_col="annotation_symbol"):
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
            counts_df.iloc[i][sym] = val

    return symbols_set, counts_df

def plot_counts_stacked(
    df: pd.DataFrame,
    output_folder: str = "plots",
    filename: str = "annotations_count"
):
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{filename}.png")

    df.plot(kind='bar', stacked=True, figsize=(18, 8), colormap='tab20')
    plt.xlabel("File")
    plt.ylabel("Number of annotations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.show()


if __name__ == "__main__":
    # get all files in annotations folder
    folder_path = "data/"
    pattern = os.path.join(folder_path, "*_annotations_1*.csv")

    files = glob.glob(pattern)
    files.sort()

    _, counts_df = get_col_counts(files)
    plot_counts_stacked(counts_df)


