import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

def balance_classes(waves_df, max_per_class, method="undersample_normal"):
    """
    method = "undersample_normal" or "oversample_abnormal"
    """

    if method == "undersample_normal":
        return balance_classes_undersample(waves_df, max_per_class)
    elif method == "oversample_abnormal":
        return balance_classes_bootstrap(waves_df, max_per_class)
    elif method == "smote":
        return balance_classes_smote(waves_df, max_per_class)
    
def balance_classes_undersample(waves_df, max_per_class=None):
    """
    Balance the number of normal ('N') and abnormal beats in the DataFrame, capped at 300 entries each.
    """
    normal = waves_df[waves_df["symbol"] == "N"]
    abnormal = waves_df[waves_df["symbol"] != "N"]

    if (max_per_class != None):
        n_samples = min(len(normal), len(abnormal), max_per_class)
    else:
        n_samples = min(len(normal), len(abnormal))
    normal_sample = normal.sample(n=n_samples, random_state=0)
    abnormal_sample = abnormal.sample(n=n_samples, random_state=0)

    counts = {"N": len(normal_sample), "abnormal": len(abnormal_sample)}
    balanced_df = pd.concat([normal_sample, abnormal_sample]).sample(frac=1, random_state=0).reset_index(drop=True)

    return balanced_df, counts


def balance_classes_bootstrap(waves_df, max_per_class=None):
    """
    Balance the number of normal ('N') and abnormal beats by bootstrapping the minority class,
    capped at max_per_class entries per class.
    """
    if "symbol" not in waves_df.columns:
        raise RuntimeError("waves_df must contain a 'symbol' column")

    # Separate classes
    normal = waves_df[waves_df["symbol"] == "N"]
    abnormal = waves_df[waves_df["symbol"] != "N"]

    # Determine target number per class
    if (max_per_class != None):
        target_samples = min(len(normal), len(abnormal), max_per_class)
    else:
        target_samples = min(len(normal), len(abnormal))

    # Oversample minority class
    if len(normal) < len(abnormal):
        normal_sample = normal.sample(n=target_samples, replace=True, random_state=0)
        abnormal_sample = abnormal.sample(n=target_samples, replace=False, random_state=0)
    else:
        abnormal_sample = abnormal.sample(n=target_samples, replace=True, random_state=0)
        normal_sample = normal.sample(n=target_samples, replace=False, random_state=0)

    balanced_df = pd.concat([normal_sample, abnormal_sample]).sample(frac=1, random_state=0).reset_index(drop=True)
    counts = {"N": len(normal_sample), "abnormal": len(abnormal_sample)}

    return balanced_df, counts



def balance_classes_smote(waves_df, feature_cols=['a','b','tau','I','v0','w0'], max_per_class=300):
    """
    Balance the number of normal ('N') and abnormal beats using SMOTE,
    capped at max_per_class entries per class.
    Returns:
        balanced_df  - dataframe of balanced samples
        counts        - dict: {"N": n_normal, "abnormal": n_abnormal}
    """
    if "symbol" not in waves_df.columns:
        raise RuntimeError("waves_df must contain a 'symbol' column")

    # Separate classes (same method as your bootstrap version)
    normal = waves_df[waves_df["symbol"] == "N"].copy()
    abnormal = waves_df[waves_df["symbol"] != "N"].copy()

    # Label encoding for binary classification
    #   0 -> N
    #   1 -> abnormal
    df = waves_df.copy()
    df["binary_label"] = (df["symbol"] != "N").astype(int)

    # Extract feature matrix
    df_clean = df.dropna(subset=feature_cols + ["binary_label"]).copy()
    X = df_clean[feature_cols].values
    y = df_clean["binary_label"].values

    # -------------------------
    # Run SMOTE (on binary label)
    # -------------------------
    sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_resample(X, y)

    # Reconstruct dataframe
    res_df = pd.DataFrame(X_res, columns=feature_cols)
    res_df["symbol"] = np.where(y_res == 0, "N", "abnormal")

    # -------------------------
    # Apply max_per_class cap
    # -------------------------
    normal_res = res_df[res_df["symbol"] == "N"]
    abnormal_res = res_df[res_df["symbol"] == "abnormal"]

    # Limit samples to max_per_class (same logic as your bootstrap version)
    normal_final = normal_res.sample(
        n=min(max_per_class, len(normal_res)), 
        random_state=0, 
        replace=False
    )
    abnormal_final = abnormal_res.sample(
        n=min(max_per_class, len(abnormal_res)), 
        random_state=0, 
        replace=False
    )

    # Final balanced dataset
    balanced_df = pd.concat([normal_final, abnormal_final]) \
                    .sample(frac=1, random_state=0) \
                    .reset_index(drop=True)

    counts = {
        "N": len(normal_final),
        "abnormal": len(abnormal_final)
    }

    return balanced_df, counts