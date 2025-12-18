import numpy as np
import pandas as pd
from tqdm import tqdm

from fhn.ecg_processing import attach_symbols, detect_waves
from fhn.metrics import compute_metrics
from fhn.simulation import fit_fhn_to_segment, loss, simulate_fhn


def preprocess_waves(ecg, annotations, fs=360):
    """
    This function detects ECG waves and attaches annotation symbols.

    Parameters:
    - ecg: list or numpy array of ECG signal values
    - annotations: dataframe of ECG annotations
    - fs: sampling frequency of the ECG signal

    Returns:
    - waves_df: dataframe containing detected ECG wave features and symbols
    - rpeaks: list of detected R-peak indices
    """
    # detect ECG waves and R-peaks
    waves_df, rpeaks = detect_waves(ecg, fs)

    # replace missing data
    waves_df = waves_df.replace("", np.nan).dropna(how="any")

    # ensure wave index columns are numeric
    for c in ["ECG_Q_Peaks", "ECG_S_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"]:
        if c in waves_df.columns:
            waves_df[c] = pd.to_numeric(waves_df[c], errors="coerce")

    # select sample of waves to run fhn on
    waves_df = attach_symbols(waves_df, annotations)

    return waves_df, rpeaks

def process_record(df, annotations, ecg, number):
    """
    This function preprocesses a single ECG record and assigns metadata.

    Parameters:
    - df: dataframe containing record-level metadata
    - annotations: dataframe of ECG annotations
    - ecg: list or numpy array of ECG signal values
    - number: identifier for the ECG recording
    - ekg_dict: dictionary of ECG signals
    - ann_dict: dictionary of annotation data

    Returns:
    - waves_df: dataframe containing processed ECG wave features
    """
    # run wave detection
    waves_df, _ = preprocess_waves(ecg, annotations)

    # flatten symbol lists to single label
    waves_df['symbol'] = waves_df['symbol'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # add recording id
    waves_df["recording"] = number
    return waves_df

    
def fit_beats(waves_bal, ecg, prefix="", keep_cols=None):
    """
    This function fits the FitzHugh–Nagumo model to ECG beats.

    Parameters:
    - waves_bal: dataframe of ECG beats to be fitted
    - ecg: list or numpy array of ECG signal values
    - prefix: string prefix used for logging/filenames
    - keep_cols: list of column names to retain from the input dataframe

    Returns:
    - fhn_df: dataframe containing fitted FHN parameters and metrics
    """
    results = []

    # iterate through ECG beats and fit FHN to each beat
    for _, row in tqdm(waves_bal.iterrows(), total=len(waves_bal), desc="Fitting FHN"):
        # skip beats with missing Q or S peak indices
        if pd.isna(row["ECG_Q_Peaks"]) or pd.isna(row["ECG_S_Peaks"]):
            continue
        q_idx, s_idx = int(row["ECG_Q_Peaks"]), int(row["ECG_S_Peaks"])

        # ensure valid QRS segment ordering (ie. S must come after Q, otherwise it's an error in detection so skip it)
        if s_idx <= q_idx:
            continue

        # extract QRS segment
        ecg_segment = ecg[q_idx:s_idx+1]
        if len(ecg_segment) < 3:
            continue

        # subsample 10 points to fit FHN model
        indices = np.linspace(0, len(ecg_segment)-1, 10, dtype=int)
        ecg_sub, t_sub = ecg_segment[indices], np.arange(len(ecg_segment))[indices]

        # fit FHN model
        res = fit_fhn_to_segment(ecg_sub, t_sub)
        if res is None:
            continue

        # compare against actual solution, get metrics
        a, b, tau, I, v0, w0 = res.x
        pred = simulate_fhn(res.x, t_sub)
        metrics = compute_metrics(ecg_sub, pred)
        
        # add widths
        qrs_width = row["ECG_S_Peaks"] - row["ECG_Q_Peaks"]
        pt_width = (row["ECG_T_Offsets"] - row["ECG_P_Onsets"]
                    if pd.notna(row["ECG_P_Onsets"]) and pd.notna(row["ECG_T_Offsets"]) else np.nan)

        # store fitted parameters
        row_dict = {
            "qrs_width": qrs_width,
            "pt_width": pt_width,
            "a": a,
            "b": b,
            "tau": tau,
            "I": I,
            "v0": v0,
            "w0": w0,
            "symbol": row["symbol"],
            "r2": metrics.get("r2", np.nan),
            "loss": loss(res.x, t_sub, ecg_sub),
            "mse": metrics.get("mse", np.nan),
            "mae": metrics.get("mae", np.nan),
        }

        # Include extra columns if specified
        for col in keep_cols:
            if col in row:
                row_dict[col] = row[col]

        results.append(row_dict)


    # plot_ecg_beats(ecg, rpeaks, waves_df, filename=f"ecg_{prefix}")
    fhn_df = pd.DataFrame(results)
    print(f"Fitted {len(fhn_df)} beats for {prefix}.")
    return fhn_df