from fhn.ecg_processing import detect_waves, attach_symbols
import pandas as pd
import numpy as np
from tqdm import tqdm
from fhn.simulation import fit_fhn_to_segment, simulate_fhn, loss, simulate_ap, fit_ap_to_segment, loss_ap
from fhn.metrics import compute_metrics

def preprocess_waves(ecg, annotations, fs=360):
    waves_df, rpeaks = detect_waves(ecg, fs)

    waves_df = waves_df.replace("", np.nan).dropna(how="any")
    for c in ["ECG_Q_Peaks", "ECG_S_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"]:
        if c in waves_df.columns:
            waves_df[c] = pd.to_numeric(waves_df[c], errors="coerce")


    # select sample of waves to run fhn on
    waves_df = attach_symbols(waves_df, annotations)

    return waves_df, rpeaks

def process_record(df, annotations, ecg, number, ekg_dict, ann_dict):
    waves_df, rpeaks = preprocess_waves(ecg, annotations)

    # Flatten symbol lists
    waves_df['symbol'] = waves_df['symbol'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Add patient ID
    waves_df["recording"] = number
    return waves_df

    
def fit_beats_fhn(waves_bal, ecg, prefix="", keep_cols=None):
    results = []
    for _, row in tqdm(waves_bal.iterrows(), total=len(waves_bal), desc="Fitting FHN"):
        if pd.isna(row["ECG_Q_Peaks"]) or pd.isna(row["ECG_S_Peaks"]):
            continue
        q_idx, s_idx = int(row["ECG_Q_Peaks"]), int(row["ECG_S_Peaks"])
        if s_idx <= q_idx:
            continue

        ecg_segment = ecg[q_idx:s_idx+1]
        if len(ecg_segment) < 3:
            continue

        # Subsample 10 points
        indices = np.linspace(0, len(ecg_segment)-1, 10, dtype=int)
        ecg_sub, t_sub = ecg_segment[indices], np.arange(len(ecg_segment))[indices]

        res = fit_fhn_to_segment(ecg_sub, t_sub)
        if res is None:
            continue

        a, b, tau, I, v0, w0 = res.x
        pred = simulate_fhn(res.x, t_sub)
        metrics = compute_metrics(ecg_sub, pred)

        qrs_width = row["ECG_S_Peaks"] - row["ECG_Q_Peaks"]
        pt_width = (row["ECG_T_Offsets"] - row["ECG_P_Onsets"]
                    if pd.notna(row["ECG_P_Onsets"]) and pd.notna(row["ECG_T_Offsets"]) else np.nan)

        row_dict = {
            "qrs_width": qrs_width,
            "pt_width": pt_width,
            "a": a,
            "b": b,
            "tau": tau,
            "I": I,
            "v0": v0,
            "w0": w0,
            "loss": loss(res.x, t_sub, ecg_sub),
            **metrics
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



def fit_beats_ap(waves_bal, ecg, prefix="", keep_cols=None):
    results = []
    for _, row in tqdm(waves_bal.iterrows(), total=len(waves_bal), desc="Fitting AP"):
        if pd.isna(row["ECG_Q_Peaks"]) or pd.isna(row["ECG_S_Peaks"]):
            continue
        q_idx, s_idx = int(row["ECG_Q_Peaks"]), int(row["ECG_S_Peaks"])
        if s_idx <= q_idx:
            continue

        ecg_segment = ecg[q_idx:s_idx+1]
        if len(ecg_segment) < 3:
            continue



        # Normalize beat for fitting
        ecg_min = np.min(ecg_segment)
        ecg_max = np.max(ecg_segment)
        ecg_norm = (ecg_segment - ecg_min) / (ecg_max - ecg_min)
        
        # Subsample normalized points
        indices = np.linspace(0, len(ecg_norm)-1, 10, dtype=int)
        ecg_sub = ecg_norm[indices]
        t_sub = np.arange(len(ecg_segment))[indices]
        # print (ecg_sub)

        res = fit_ap_to_segment(ecg_sub, t_sub)
        if res is None:
            continue

        fitted_params = res.x
        # print("fit:", fitted_params)

        v_fit_norm = simulate_ap(fitted_params, t_sub)
        v_fit = v_fit_norm * (ecg_max - ecg_min) + ecg_min
        metrics = compute_metrics(ecg_sub, v_fit_norm)

        qrs_width = row["ECG_S_Peaks"] - row["ECG_Q_Peaks"]
        pt_width = (row["ECG_T_Offsets"] - row["ECG_P_Onsets"]
                    if pd.notna(row["ECG_P_Onsets"]) and pd.notna(row["ECG_T_Offsets"]) else np.nan)

        row_dict = {
            "qrs_width": qrs_width,
            "pt_width": pt_width,
            "a": fitted_params[0],
            "k": fitted_params[1],
            "epsilon_low": fitted_params[2],
            "epsilon_high": fitted_params[3],
            "u0": fitted_params[4],
            "v0": fitted_params[5],
            "loss": loss_ap(res.x, t_sub, ecg_sub),
            **metrics
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