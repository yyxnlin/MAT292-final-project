import pandas as pd

def filter_low_loss(df, loss_col='loss', threshold=0.1):
    """
    Keep only rows where the loss column is below the given threshold.
    """
    if loss_col not in df.columns:
        return df.copy()
    return df[df[loss_col] < threshold].copy()