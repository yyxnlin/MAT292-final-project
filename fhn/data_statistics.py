import pandas as pd
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


# if __name__ == "__main__":
#     # get all files in annotations folder in the format of xxx_annotations_1.csv
#     folder_path = "data/"
#     pattern = os.path.join(folder_path, "*_annotations_1*.csv")

#     files = glob.glob(pattern)
#     files.sort()

#     _, counts_df = get_col_counts(files)
#     plot_counts_stacked(counts_df)


