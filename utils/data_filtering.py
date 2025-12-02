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
