import numpy as np
import pandas as pd
import neurokit2 as nk

from typing import List

def detect_waves(ecg, fs=360):
    """
    Detect ECG waves and R-peaks from a signal.
    """
    _, info = nk.ecg_process(ecg, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]
    _, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs, method="dwt")
    waves_df = pd.DataFrame(waves)
    waves_df["rpeak"] = rpeaks
    return waves_df, rpeaks

def attach_symbols(waves_df: pd.DataFrame, 
                   annotations: List):
    """ 
    Attach an annotation symbol to each row in the waves_df based on annotations given in dataset
    """
    if "symbol" in waves_df.columns:
        return waves_df

    if annotations is None:
        raise RuntimeError("No annotations found.")

    ann = annotations.sort_values("index").copy()

    # drop rows with no P_Onset or T_Offset column, then get the intervals of each wave (P to T)
    waves_valid = waves_df.dropna(subset=["ECG_P_Onsets", "ECG_T_Offsets"]).copy()
    intervals = pd.IntervalIndex.from_arrays(
        waves_valid["ECG_P_Onsets"], waves_valid["ECG_T_Offsets"], closed="both"
    )

    # try to match each index in the annotations file to an interval; if there is no match then skip
    matches, _ = intervals.get_indexer_non_unique(ann["index"])
    waves_valid["symbol"] = np.nan
    for ann_idx, wave_idx in enumerate(matches):
        if wave_idx != -1:
            waves_valid.iloc[wave_idx, waves_valid.columns.get_loc("symbol")] = ann.iloc[ann_idx]["annotation_symbol"]

    # attach symbol column to waves_df
    waves_df["symbol"] = waves_valid["symbol"]
    return waves_df

