import argparse
import os
import pandas as pd
from tqdm import tqdm

from fhn.simulation import fit_fhn_to_segment, simulate_fhn, loss
from fhn.ecg_processing import detect_waves, attach_symbols
from fhn.fhn_processing import preprocess_waves, fit_beats, process_record
from utils.balancing import balance_classes
from utils.data_filtering import filter_df_by_threshold
from utils.file_utils import build_record_dicts
from fhn.data_statistics import get_col_counts
from fhn.metrics import compute_metrics
from fhn.plots import plot_ecg_beats, plot_single_beat, plot_confusion_matrix, plot_counts_stacked
from classification.knn_classifier import *


def run_data_stats(data_folder, output_folder):
    ekg_dict, _ = build_record_dicts(data_folder)

    filenames = ekg_dict.values()
    files = [os.path.join(data_folder, f) for f in filenames]
    
    _, counts_df = get_col_counts(files)
    plot_counts_stacked(counts_df, output_folder=output_folder)


def run_combine(data_folder, output_folder, log_file):
    os.makedirs(output_folder, exist_ok=True)
    ekg_dict, ann_dict = build_record_dicts(data_folder)

    all_waves = []
    for number in tqdm(sorted(ekg_dict.keys()), desc="Processing all records"):
        try:
            ekg_path = os.path.join(data_folder, ekg_dict[number])
            ann_path = os.path.join(data_folder, ann_dict[number])

            df = pd.read_csv(ekg_path)
            ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
            annotations = pd.read_csv(ann_path)

            waves = process_record(df, annotations, ecg, number, ekg_dict, ann_dict)
            all_waves.append(waves)
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number}: {repr(e)}\n")
            print(f"Error processing {number}: {e}")

    all_waves_df = pd.concat(all_waves, ignore_index=True)
    all_waves_df.to_parquet(f"{output_folder}/all_waves_raw.parquet")


def run_fhn(data_folder, output_folder, log_file="error_log.txt"):
    ekg_dict, _ = build_record_dicts(data_folder) # change this later so you save all the numbers somewhere, no need to reconstruct ekg_dict again
    raw_waves_df = pd.read_parquet(f"{output_folder}/all_waves_raw.parquet")
    raw_waves_df["symbol_binary"] = raw_waves_df["symbol"].map(
        lambda x: 'N' if x=='N' else 'Not N'
    )

    # fit FHN
    ecg_cache = {}
    for number in tqdm(sorted(ekg_dict.keys()), desc="Loading ECGs"):
        try:
            df = pd.read_csv(os.path.join(data_folder, ekg_dict[number]))
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
        fhn_df.to_parquet(f"{output_folder}/all_fhn_data_raw.parquet")
    else:
        print("No FHN results to save.")


def run_filter(data_dir, loss_threshold, r2_threshold):
    fhn_df = pd.read_parquet(f"{data_dir}/all_fhn_data_raw.parquet")
    fhn_df_filtered = filter_df_by_threshold(fhn_df, "loss", loss_threshold, "r2", r2_threshold)
    fhn_df_filtered.to_parquet(f"{data_dir}/all_fhn_data_filtered.parquet")

def run_balance(data_dir, max_per_class, method):
    fhn_df_filtered = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered.parquet")
    balanced_waves_df, counts = balance_classes(fhn_df_filtered, max_per_class=max_per_class, method=method)
    balanced_waves_df.to_parquet(f"{data_dir}/all_fhn_data_filtered_balanced.parquet")

def run_model(data_dir, label_col):
    balanced_waves_df = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered_balanced.parquet")

    features_fhn = ['a', 'b', 'tau', 'I', 'v0', 'w0']
    features_width = ['qrs_width', 'pt_width']
    X_scaled, y, groups, _, _, target_names = prepare_knn_data_general(df=balanced_waves_df, features=features_fhn+features_width, label_col=label_col)
    
    group_counts = pd.Series(groups).value_counts()
    print("Number of datapoints per group:")
    print(group_counts)

    y_preds = knn_leave_one_group_out(X_scaled, y, groups)
    acc, report, cm = classification_metrics(y, y_preds, target_names)
    plot_confusion_matrix(cm, labels = target_names)
    print(f"Accuracy: {acc:.4f}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="ECG Classification Pipeline")

    parser.add_argument("--step", type=str, required=True,
                        choices=["data_stats", "combine", "fhn", "filter", "balance", "model"])

    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument("--log_file", type=str, default="error_log.txt")

    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--method", type=str, default="undersample_normal")

    parser.add_argument("--loss_threshold", type=float, default=0.1)
    parser.add_argument("--r2_threshold", type=float, default=0.6)

    parser.add_argument("--classification_type", type=str, default="symbol_binary")


    args = parser.parse_args()

    if args.step == "data_stats":
        run_data_stats(args.data_folder, args.plots_folder)

    elif args.step == "combine":
        run_combine(args.data_folder,
                    args.output_folder, args.log_file)

    elif args.step == "fhn":
        run_fhn(args.data_folder, args.output_folder,
                args.input, args.output)

    elif args.step == "filter":
        run_filter(args.output_folder,
                   args.loss_threshold, args.r2_threshold)
        
    elif args.step == "balance":
        run_balance(args.output_folder,
                    args.max_per_class, args.method)

    elif args.step == "model":
        run_model(args.output_folder, args.classification_type)


if __name__ == "__main__":
    main()
