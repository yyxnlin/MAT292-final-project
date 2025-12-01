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
from utils.data_filtering import filter_low_loss
from fhn.metrics import compute_metrics
from fhn.plots import plot_ecg_beats, plot_single_beat
from classification.knn_classifier import *


def process_record(number, ekg_dict, ann_dict, data_folder="data", output_folder="output3", max_per_class=300):
    ekg_path = os.path.join(data_folder, ekg_dict[number])
    ann_path = os.path.join(data_folder, ann_dict[number])

    df = pd.read_csv(ekg_path)
    ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
    annotations = pd.read_csv(ann_path)

    waves_df, rpeaks = preprocess_waves(ecg, annotations)
    waves_bal, counts = balance_classes_bootstrap(waves_df, max_per_class=max_per_class)
    fhn_df = fit_beats(waves_bal, ecg)

    # plot_single_beat(ecg, waves_bal.iloc[0], output_folder="plots", filename=f"ecg_{number}_beat_0")

    output_path = os.path.join(output_folder, f"{number}_results.parquet")
    fhn_df.to_parquet(output_path)
    print(f"Finished {number}")


if __name__ == "__main__":
    log_file = "error_log.txt"
    DATA_FOLDER = "data"
    OUTPUT_FOLDER = "output3"
    MAX_PER_CLASS = 300

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
    set3 = [100, 109, 116, 118, 119, 122, 124, 207, 208, 210, 212, 232]

    # numbers_to_run = sorted(set(set1 + set2))
    numbers_to_run = sorted(set(set3))


    numbers = sorted(set(numbers_to_run) & set(ekg_dict.keys()) & set(ann_dict.keys()))
    # numbers = sorted(set(ekg_dict.keys()) & set(ann_dict.keys()))

    for number in tqdm(numbers, desc="Processing files"):
        try:
            process_record(number, ekg_dict, ann_dict, DATA_FOLDER, OUTPUT_FOLDER, max_per_class=MAX_PER_CLASS)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number}: {repr(e)}\n")
            print(f"Error processing {number}: {e}")