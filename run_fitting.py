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
from utils.data_filtering import aggregate_and_filter_outputs, balanced_resample
from utils.file_utils import build_record_dicts
from fhn.metrics import compute_metrics
from fhn.plots import plot_ecg_beats, plot_single_beat, plot_confusion_matrix
from classification.knn_classifier import *


def process_record(number, ekg_dict, ann_dict, data_folder="data", output_folder="output3", max_per_class=300, method="undersample_normal"):
    ekg_path = os.path.join(data_folder, ekg_dict[number])
    ann_path = os.path.join(data_folder, ann_dict[number])

    df = pd.read_csv(ekg_path)
    ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
    annotations = pd.read_csv(ann_path)

    waves_df, rpeaks = preprocess_waves(ecg, annotations)




    def flatten_symbol(x):
        if isinstance(x, list):
            return x[0]  # or choose the first symbol
        return x

    waves_df['symbol'] = waves_df['symbol'].apply(flatten_symbol)



    waves_bal, counts = balance_classes(waves_df, max_per_class=max_per_class, method=method)
    fhn_df = fit_beats(waves_bal, ecg)

    # plot_single_beat(ecg, waves_bal.iloc[0], output_folder="plots", filename=f"ecg_{number}_beat_0")

    output_path = os.path.join(output_folder, f"{number}_results.parquet")
    fhn_df.to_parquet(output_path)
    print(f"Finished {number}")

def run_knn(data_dir, loss_threshold = 0.2, r2_threshold = 0.8):
    all_df = aggregate_and_filter_outputs(data_dir, "loss", loss_threshold, "r2", r2_threshold)

    categories = {
        'N': 'N',
        'A': 'AJS', 'J': 'AJS', 'S': 'AJS'
        # any symbols not listed will default to None
    }


    binary = {
        'N': 'N',
    }

    col = "symbol_binary"

    all_df[col] = all_df['symbol'].apply(lambda x: binary.get(x, 'Not N')) # default
    all_df = all_df[all_df[col].notna() & (all_df[col] != 'Other')]

    all_df.to_parquet("jdsfjskfjds.parquet")
    # Keep only relevant columns
    # all_df = all_df[['symbol', col]]

    # all_df combines all the rows of data with loss column > 0.1
    # all_df['symbol_binary'] = all_df['symbol'].apply(lambda x: 'N' if x=='N' else 'not N')

    # resample to make categories even
    df_bal, counts = balanced_resample(all_df, col)
    print("Balanced symbol counts:\n", counts)

    df_bal.to_parquet("12222.parquet")

    features_fhn = ['a', 'b', 'tau', 'I', 'v0', 'w0']
    features_width = ['qrs_width', 'pt_width']
    X_scaled, y, groups, _, _, target_names = prepare_knn_data_general(df=df_bal, features=features_fhn+features_width, label_col="symbol_binary")
    
    y_preds = knn_leave_one_group_out(X_scaled, y, groups)
    acc, report, cm = classification_metrics(y, y_preds, target_names)
    plot_confusion_matrix(cm, labels = target_names)
    print(f"Accuracy: {acc:.4f}")
    print(report)
    

if __name__ == "__main__":
    log_file = "error_log.txt"
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output2"
    MAX_PER_CLASS = 300


    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Prepare record lookup
    # ekg_dict, ann_dict = build_record_dicts(DATA_FOLDER)

    # # set1 = [106, 111, 119, 200, 201, 203, 208, 210, 207, 214, 118, 124, 212, 231]
    # # set2 = [100, 109, 116, 122, 118, 119, 232]
    # set3 = [100, 109, 116, 118, 119, 122, 124, 207, 208, 210, 212, 232]
    # records_to_run = sorted(set3)


    # for number in tqdm(records_to_run, desc="Processing files"):
    #     try:
    #         process_record(number, ekg_dict, ann_dict, DATA_FOLDER, OUTPUT_FOLDER, 
    #                        max_per_class=MAX_PER_CLASS, 
    #                        method="undersample_normal")
    #     except Exception as e:
    #         with open(log_file, "a") as f:
    #             f.write(f"{number}: {repr(e)}\n")
    #         print(f"Error processing {number}: {e}")

    # KNN -----------------------------------
    run_knn(OUTPUT_FOLDER, loss_threshold = 0.1, r2_threshold = 0.8)
    