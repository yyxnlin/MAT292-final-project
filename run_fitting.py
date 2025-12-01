import os
import time
import re
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from fhn.simulation import fit_fhn_to_segment, simulate_fhn, loss
from fhn.ecg_processing import detect_waves, attach_symbols
from fhn.utils import balance_classes
from fhn.metrics import compute_metrics
from fhn.plots import plot_ecg_beats, plot_single_beat

def run_file(ecg, annotations, prefix, fs=360):
    # get waves df
    count = 0

    waves_df, _ = detect_waves(ecg, fs)
    waves_df.to_parquet(f"waves/{prefix}_waves.parquet")

    waves_df = waves_df.replace("", np.nan).dropna(how="any")
    for c in ["ECG_Q_Peaks", "ECG_S_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"]:
        if c in waves_df.columns:
            waves_df[c] = pd.to_numeric(waves_df[c], errors="coerce")


    # select sample of waves to run fhn on
    waves_df = attach_symbols(waves_df, annotations)

    waves_df.to_parquet("kdfjdisfjisfjsfjdsijds.parquet")
    waves_bal, counts = balance_classes(waves_df)

    print(counts)
    results = []

    # test plotting
    # for _, row in tqdm(waves_bal.iterrows(), total=len(waves_bal), desc="waves_bal iteration:"):
        # if (count < 10):
        #     print(row["ECG_Q_Peaks"], row["ECG_S_Peaks"], row["rpeak"], row["symbol"])
        #     plot_single_beat(ecg, row, output_folder="plots", filename=f"ecg_{prefix}_beat_{count}")
        #     count += 1

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

        results.append({
            "qrs_width": qrs_width,
            "pt_width": pt_width,
            "a": a,
            "b": b,
            "tau": tau,
            "I": I,
            "v0": v0,
            "w0": w0,
            "symbol": row["symbol"],
            "loss": loss(res.x, t_sub, ecg_sub),
            **metrics
        })

        


    # plot_ecg_beats(ecg, rpeaks, waves_df, filename=f"ecg_{prefix}")
    fhn_df = pd.DataFrame(results)
    print(f"Fitted {len(fhn_df)} beats for {prefix}.")
    return fhn_df


if __name__ == "__main__":
    log_file = "error_log.txt"
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output2"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = os.listdir(DATA_FOLDER)

    ekg_files = [f for f in files if re.search(r"_ekg\.csv$", f)]
    ann_files = [f for f in files if re.search(r"_annotations_1\.csv", f)]

    def get_number(filename):
        m = re.match(r"(\d+)_", filename)
        return int(m.group(1)) if m else None

    ekg_dict = {get_number(f): f for f in ekg_files}
    ann_dict = {get_number(f): f for f in ann_files}

    # set1 = [106, 111, 119, 200, 201, 203, 208, 210, 207, 214, 118, 124, 212, 231]
    # set2 = [100, 109, 116, 122, 118, 119, 232]
    set3 = [100]

    # numbers_to_run = sorted(set(set1 + set2))
    numbers_to_run = sorted(set(set3))


    numbers = sorted(set(numbers_to_run) & set(ekg_dict.keys()) & set(ann_dict.keys()))
    # numbers = sorted(set(ekg_dict.keys()) & set(ann_dict.keys()))

    for number in tqdm(numbers, desc="Processing files"):
        print(f"Processing number: {number}")
        output_path = os.path.join(OUTPUT_FOLDER, f"{number}_results.parquet")

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {number}: results already exist.")
            continue

        try:
            ekg_path = os.path.join(DATA_FOLDER, ekg_dict[number])
            ann_path = os.path.join(DATA_FOLDER, ann_dict[number])

            df = pd.read_csv(ekg_path)
            ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
            annotations = pd.read_csv(ann_path)

            fhn_df = run_file(ecg, annotations, number)

            fhn_df.to_parquet(output_path)
            print(f"Finished {number}")

        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number}: {repr(e)}\n")
            print(f"Error processing {number}: {e}")
            continue