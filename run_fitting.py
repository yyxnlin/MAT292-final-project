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
    OUTPUT_FOLDER = "output_global_1"
    MAX_PER_CLASS = 100

    LOSS_THRESHOLD = 0.3
    R2_THRESHOLD = 0.5

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    set3 = [100, 109, 116, 118, 119, 122, 124, 207, 208, 210, 212, 232]
    records_to_run = sorted(set3)

    ekg_dict, ann_dict = build_record_dicts(DATA_FOLDER)

    # ------------------------ COMBINING ALL WAVES ------------------------

    # all_waves = []

    # for number in tqdm(records_to_run):
    #     try:
    #         waves = process_record(number, ekg_dict, ann_dict, DATA_FOLDER)
    #         all_waves.append(waves)
    #     except Exception as e:
    #         with open(log_file, "a") as f:
    #             f.write(f"{number}: {repr(e)}\n")

    # all_waves_df = pd.concat(all_waves, ignore_index=True)
    # all_waves_df.to_parquet(f"{OUTPUT_FOLDER}/all_waves_raw.parquet")


    # ------------------------ BALANCING ------------------------
    # all_waves_df = pd.read_parquet(f"{OUTPUT_FOLDER}/all_waves_raw.parquet")

    # # balancing
    # # Define your binary or multi-class map
    # mapping = { 'N':'N' }

    # all_waves_df["symbol_binary"] = all_waves_df["symbol"].map(
    #     lambda x: 'N' if x=='N' else 'Not N'
    # )

    # # Apply global balancing
    # balanced_waves_df, counts = balance_classes(all_waves_df, max_per_class=300, method="undersample_normal")
    # print(counts)

    # balanced_waves_df.to_parquet("all_waves_balanced.parquet")

    # ------------------------ FHN ------------------------
    balanced_waves_df = pd.read_parquet("all_waves_balanced.parquet")
    # raw_waves_df = pd.read_parquet("output_global_1/all_waves_raw.parquet")
    
    df = balanced_waves_df
    # fit FHN
    ecg_cache = {}
    for number in records_to_run:
        df = pd.read_csv(os.path.join(DATA_FOLDER, ekg_dict[number]))
        ecg_cache[number] = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values

    fhn_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ecg = ecg_cache[row["recording"]]
        fhn_params_df = fit_beats(pd.DataFrame([row]), ecg, keep_cols=["recording", "symbol_binary"])
        fhn_rows.append(fhn_params_df)
        
    fhn_df = pd.concat(fhn_rows, ignore_index=True)

    fhn_df.to_parquet("all_fhn_data_raw.parquet")

    # ------------------------ FILTER -----------------------------
    # fhn_df = pd.read_parquet("all_fhn_data.parquet")
    # fhn_df_filtered = filter_df_by_threshold(fhn_df, "loss", LOSS_THRESHOLD, "r2", R2_THRESHOLD)
    # fhn_df_filtered.to_parquet("all_fhn_data_filtered.parquet")

    # # ------------------------ KNN ------------------------
    # fhn_df_filtered = pd.read_parquet("all_fhn_data_filtered.parquet")

    # run_knn(fhn_df_filtered, "symbol_binary")

    