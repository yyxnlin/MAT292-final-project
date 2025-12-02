import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

def balance_classes(waves_df, label_col, max_per_class, method="undersample"):
    """
    method = "undersample" or "oversample"
    """

    if method == "undersample":
        return balance_classes_undersample(waves_df, label_col, max_per_class)
    elif method == "oversample":
        return balance_classes_bootstrap(waves_df, label_col, max_per_class)
    
def balance_classes_undersample(df, label_col, max_per_class=None):
    """
    Balance multiple classes by undersampling each class to the size of the smallest class
    (capped to max_per_class cap if provided)
    """
    # group by class
    class_groups = {cls: grp for cls, grp in df.groupby(label_col)}

    # sample size will be the size of the category with the least number of entries, capped at max_per_class if it is not None
    min_class_size = min(len(grp) for grp in class_groups.values())
    if max_per_class is not None:
        sample_size = min(min_class_size, max_per_class)
    else:
        sample_size = min_class_size

    # sample each class
    balanced_parts = []
    counts = {}
    for cls, grp in class_groups.items():
        sampled = grp.sample(n=sample_size, random_state=0)
        balanced_parts.append(sampled)
        counts[cls] = len(sampled)

    # combine
    balanced_df = (
        pd.concat(balanced_parts)
          .sample(frac=1, random_state=0)
          .reset_index(drop=True)
    )
    return balanced_df, counts


def balance_classes_bootstrap(df, label_col, max_per_class=None):
    """
    Balance multiple classes by oversampling (bootstrapping) each class to the size of the largest class
    (capped to max_per_class if provided).
    """
    # group by class
    class_groups = {cls: grp for cls, grp in df.groupby(label_col)}

    # sample size will be the size of the category with the largest number of entries, capped at max_per_class if provided
    max_class_size = max(len(grp) for grp in class_groups.values())
    if max_per_class is not None:
        sample_size = min(max_class_size, max_per_class)
    else:
        sample_size = max_class_size

    # sample each class with replacement if needed
    balanced_parts = []
    counts = {}
    for cls, grp in class_groups.items():
        replace = len(grp) < sample_size  # only replace if group is smaller than target
        sampled = grp.sample(n=sample_size, replace=replace, random_state=0) # duplicate smaller categories to reach desired sample size
        balanced_parts.append(sampled)
        counts[cls] = len(sampled)

    # combine
    balanced_df = (
        pd.concat(balanced_parts)
          .sample(frac=1, random_state=0)
          .reset_index(drop=True)
    )

    return balanced_df, counts