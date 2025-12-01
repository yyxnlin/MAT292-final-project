import pandas as pd

def balance_classes(waves_df, max_per_class=300):
    """
    Balance the number of normal ('N') and abnormal beats in the DataFrame, capped at 300 entries each.
    """
    normal = waves_df[waves_df["symbol"] == "N"]
    abnormal = waves_df[waves_df["symbol"] != "N"]
    n_samples = min(len(normal), len(abnormal), max_per_class)
    normal_sample = normal.sample(n=n_samples, random_state=0)
    abnormal_sample = abnormal.sample(n=n_samples, random_state=0)

    counts = {"N": len(normal_sample), "abnormal": len(abnormal_sample)}
    balanced_df = pd.concat([normal_sample, abnormal_sample]).sample(frac=1, random_state=0).reset_index(drop=True)

    return balanced_df, counts
