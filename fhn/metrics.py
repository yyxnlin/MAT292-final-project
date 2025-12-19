from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(ecg_sub, pred):
    metrics = {}
    try:
        metrics["r2"] = r2_score(ecg_sub, pred)
    except Exception:
        metrics["r2"] = np.nan
    try:
        metrics["mse"] = np.mean((ecg_sub - pred) ** 2)
    except Exception:
        metrics["mse"] = np.nan
    try:
        metrics["mae"] = mean_absolute_error(ecg_sub, pred)
    except Exception:
        metrics["mae"] = np.nan
    return metrics


def compute_fhn_metric_averages(df, metrics=("loss", "r2", "mae")):
    """
    Compute average values of selected FHN fitting metrics.
    """
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    return df[list(metrics)].mean()