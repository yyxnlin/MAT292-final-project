from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import numpy as np

def compute_metrics(ecg_sub, pred):
    metrics = {}
    try:
        metrics["r2"] = r2_score(ecg_sub, pred)
    except Exception:
        metrics["r2"] = np.nan
    try:
        metrics["rmse"] = root_mean_squared_error(ecg_sub, pred, squared=False)
    except Exception:
        metrics["rmse"] = np.nan
    try:
        metrics["mae"] = mean_absolute_error(ecg_sub, pred)
    except Exception:
        metrics["mae"] = np.nan
    return metrics