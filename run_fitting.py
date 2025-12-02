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
from fhn.fhn_processing import preprocess_waves, fit_beats
from utils.balancing import balance_classes, balance_classes_bootstrap
from utils.data_filtering import filter_df_by_threshold
from utils.file_utils import build_record_dicts
from fhn.metrics import compute_metrics
from fhn.plots import plot_ecg_beats, plot_single_beat, plot_confusion_matrix
from classification.knn_classifier import *


def process_record(number, ekg_dict, ann_dict, data_folder="data"):
    ekg_path = os.path.join(data_folder, ekg_dict[number])
    ann_path = os.path.join(data_folder, ann_dict[number])

    df = pd.read_csv(ekg_path)
    ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
    annotations = pd.read_csv(ann_path)

    waves_df, rpeaks = preprocess_waves(ecg, annotations)

    # Flatten symbol lists
    waves_df['symbol'] = waves_df['symbol'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Add patient ID
    waves_df["recording"] = number
    return waves_df


def run_knn(df, label_col):
    features_fhn = ['a', 'b', 'tau', 'I', 'v0', 'w0']
    features_width = ['qrs_width', 'pt_width']
    X_scaled, y, groups, _, _, target_names = prepare_knn_data_general(df=df, features=features_fhn+features_width, label_col=label_col)
    
    group_counts = pd.Series(groups).value_counts()
    print("Number of datapoints per group:")
    print(group_counts)

    y_preds = knn_leave_one_group_out(X_scaled, y, groups)
    acc, report, cm = classification_metrics(y, y_preds, target_names)
    plot_confusion_matrix(cm, labels = target_names)
    print(f"Accuracy: {acc:.4f}")
    print(report)
    

if __name__ == "__main__":
    log_file = "error_log.txt"
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output_global_2"
    MAX_PER_CLASS = None

    LOSS_THRESHOLD = 0.3
    R2_THRESHOLD = 0.5

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    ekg_dict, ann_dict = build_record_dicts(DATA_FOLDER)

    # ------------------------ COMBINING ALL WAVES ------------------------
    all_waves = []

    for number in tqdm(sorted(ekg_dict.keys()), desc="Processing all records"):
        try:
            waves = process_record(number, ekg_dict, ann_dict, DATA_FOLDER)
            all_waves.append(waves)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number}: {repr(e)}\n")
            print(f"Error processing {number}: {e}")

    all_waves_df = pd.concat(all_waves, ignore_index=True)
    all_waves_df.to_parquet(f"{OUTPUT_FOLDER}/all_waves_raw.parquet")

    # ------------------------ FHN ------------------------
    raw_waves_df = pd.read_parquet(f"{OUTPUT_FOLDER}/all_waves_raw.parquet")
    raw_waves_df["symbol_binary"] = raw_waves_df["symbol"].map(
        lambda x: 'N' if x=='N' else 'Not N'
    )

    ecg_cache = {}
    
    for number in tqdm(sorted(ekg_dict.keys()), desc="Loading ECGs"):
        try:
            df = pd.read_csv(os.path.join(DATA_FOLDER, ekg_dict[number]))
            ecg_cache[number] = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number} (ECG load error): {repr(e)}\n")
            print(f"Error loading ECG for record {number}: {e}")
            continue

    fhn_rows = []

    for idx, row in tqdm(raw_waves_df.iterrows(), total=len(raw_waves_df), desc="Fitting FHN"):
        try:
            ecg = ecg_cache[row["recording"]]
            fhn_params_df = fit_beats(pd.DataFrame([row]), ecg, keep_cols=["recording", "symbol_binary"])
            fhn_rows.append(fhn_params_df)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{row['recording']} (FHN fit error, row {idx}): {repr(e)}\n")
            print(f"Skipping row {idx} in recording {row['recording']} due to error: {e}")
            continue

    if fhn_rows:
        fhn_df = pd.concat(fhn_rows, ignore_index=True)
        fhn_df.to_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_raw.parquet")
    else:
        print("No FHN results to save.")

    # ------------------------ FILTER -----------------------------
    fhn_df = pd.read_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_raw.parquet")
    fhn_df_filtered = filter_df_by_threshold(fhn_df, "loss", LOSS_THRESHOLD, "r2", R2_THRESHOLD)
    fhn_df_filtered.to_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_filtered.parquet")

    # ------------------------ BALANCING ------------------------
    fhn_df_filtered = pd.read_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_filtered.parquet")
    balanced_waves_df, counts = balance_classes(fhn_df_filtered, max_per_class=MAX_PER_CLASS, method="undersample_normal")
    balanced_waves_df.to_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_filtered_balanced.parquet")

    # ------------------------ KNN ------------------------
    balanced_waves_df = pd.read_parquet(f"{OUTPUT_FOLDER}/all_fhn_data_filtered_balanced.parquet")
    run_knn(balanced_waves_df, label_col = "symbol_binary")

    