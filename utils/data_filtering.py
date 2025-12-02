import pandas as pd
import os 


def filter_df_by_threshold(df, loss_col='loss', loss_threshold=0.1, r2_col = 'r2', r2_threshold = 0.8):
    df_filtered=df.copy()
    # Keep only rows with loss < 0.1
    if loss_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[loss_col] < loss_threshold]
    
    # Filter by r2 if column exists
    if r2_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[r2_col] > r2_threshold]


    return df_filtered


def categorize_symbols(df, category_map, symbol_col="symbol", new_col="symbol_categorized"):
    """
    Create a new column that maps each symbol into a category based on category_map.
    
    category_map is a dictionary of the form {category : List[symbols]}

    for example:
        {
            "N": ["N"],
            "LRB": ["L", "R", "B"],
            ...
        }
    this will result in the new column being "N" if the original column was "N", 
    "LRB" if any of the symbols in the original column was L, R, or B, 
    "Other" otherwise.
    """

    # build inverse lookup: symbol -> category
    # print(category_map.items())
    lookup = {}
    for category, symbol_list in category_map.items():
        for sym in symbol_list:
            lookup[sym] = category

    # map using the lookup; default is "Other" if it is unmapped
    df[new_col] = df[symbol_col].map(lambda s: lookup.get(s, "Other"))

    return df
