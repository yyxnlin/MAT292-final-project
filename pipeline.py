import argparse
import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from classification.knn_classifier import *
from fhn.data_statistics import get_col_counts, save_filtering_summary
from fhn.fhn_processing import fit_beats,  process_record
from fhn.metrics import compute_fhn_metric_averages
from fhn.plots import (plot_confusion_matrix, plot_counts_stacked,
                       plot_filtering_summary,
                       plot_tsne_sample_by_symbol)
from utils.balancing import balance_classes
from utils.data_filtering import categorize_symbols, filter_df_by_threshold
from utils.file_utils import build_record_dicts

import warnings
warnings.filterwarnings("ignore")

def log_step(step_idx, total_steps, message):
    print(f"\n[{step_idx}/{total_steps}] {message}")

def run_data_stats(data_folder, plots_folder):
    # load record filenames without loading full ECG signals
    ekg_dict, _ = build_record_dicts(data_folder)

    # build full paths to csv files
    filenames = ekg_dict.values()
    files = [os.path.join(data_folder, f) for f in filenames]

    # compute symbol counts per record and plot summary
    _, counts_df = get_col_counts(files)
    plot_counts_stacked(counts_df, output_folder=plots_folder)


def run_combine(data_folder, output_folder, log_file):
    out_path = os.path.join(output_folder, "all_waves_raw.parquet")

    # if cached, skip
    if os.path.exists(out_path):
        print(f"   [i] Found existing combined wave data at {out_path}. Skipping combine step.")
        return
    
    # ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # load mappings from record number to ECG / annotation files
    ekg_dict, ann_dict = build_record_dicts(data_folder)

    all_waves = []

    # process each record independently
    for number in tqdm(sorted(ekg_dict.keys()), desc="Processing all records"):
        try:
            ekg_path = os.path.join(data_folder, ekg_dict[number])
            ann_path = os.path.join(data_folder, ann_dict[number])

            # load ECG signal and annotations
            df = pd.read_csv(ekg_path)
            ecg = df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
            annotations = pd.read_csv(ann_path)

            # extract wave features for this record
            waves = process_record(df, annotations, ecg, number)
            all_waves.append(waves)

        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"{number}: {repr(e)}\n")

    # combine all records into a single dataframe
    all_waves_df = pd.concat(all_waves, ignore_index=True)
    all_waves_df.to_parquet(f"{output_folder}/all_waves_raw.parquet")


def run_fhn(data_folder, output_folder, log_file="error_log.txt", n_jobs=-1):
    out_path = os.path.join(output_folder, "all_fhn_data_raw.parquet")

    # if cached, skip
    if os.path.exists(out_path):
        print(f"   [i] Found existing FHN data at {out_path}. Skipping FHN fitting.")
        return
    
    # reload record dictionary to map recordings to ECG files
    ekg_dict, _ = build_record_dicts(data_folder)

    # load preprocessed wave-level data
    raw_waves_df = pd.read_parquet(f"{output_folder}/all_waves_raw.parquet")

    # cache ECG signals
    ecg_cache = {}
    for number in tqdm(sorted(ekg_dict.keys()), desc="Loading ECGs"):
        try:
            df = pd.read_csv(os.path.join(data_folder, ekg_dict[number]))
            ecg_cache[number] = (
                df["MLII"].values if "MLII" in df.columns else df.iloc[:, 1].values
            )
        except Exception as e:
            # log ECG loading errors separately
            with open(log_file, "a") as f:
                f.write(f"{number} (ECG load error): {repr(e)}\n")
            continue

    # helper function for parallel FHN fitting
    def run_one(i, row):
        import warnings
        warnings.filterwarnings("ignore")

        try:
            ecg = ecg_cache[row["recording"]]
            return fit_beats(
                pd.DataFrame([row]),
                ecg,
                keep_cols=["recording", "symbol"]
            )
        except Exception as e:
            # log per-row fitting failures
            with open(log_file, "a") as f:
                f.write(f"{row['recording']} (FHN fit error, row {i}): {repr(e)}\n")
            return None

    # run FHN fitting in parallel
    with tqdm_joblib(tqdm(desc="Running batch", total=len(raw_waves_df))):
        fhn_rows = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(i, row)
            for i, row in raw_waves_df.iterrows()
        )

    # save results
    if fhn_rows:
        fhn_df = pd.concat(fhn_rows, ignore_index=True)
        fhn_df.to_parquet(out_path)


def run_filter(data_dir, loss_threshold, r2_threshold):
    # load raw FHN fits
    fhn_df = pd.read_parquet(f"{data_dir}/all_fhn_data_raw.parquet")

    # filter beats based on fit quality
    fhn_df_filtered = filter_df_by_threshold(
        fhn_df, "loss", loss_threshold, "r2", r2_threshold
    )

    fhn_df_filtered.to_parquet(f"{data_dir}/all_fhn_data_filtered.parquet")

def run_balance(data_dir, max_per_class, method, category_map):
    # load filtered FHN data
    fhn_df_filtered = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered.parquet")

    # map raw symbols into higher-level categories
    fhn_df_filtered_categorized = categorize_symbols(
        df=fhn_df_filtered,
        category_map=category_map,
        symbol_col="symbol",
        new_col="symbol_categorized"
    )

    # balance class distributions
    balanced_waves_df, counts = balance_classes(
        waves_df=fhn_df_filtered_categorized,
        label_col="symbol_categorized",
        max_per_class=max_per_class,
        method=method
    )
    balanced_waves_df.to_parquet(f"{data_dir}/all_fhn_data_filtered_balanced.parquet")

def run_filtered_fhn_stats(data_dir, plots_dir, loss_threshold, r2_threshold, features):
    # load raw and filtered datasets
    fhn_df = pd.read_parquet(f"{data_dir}/all_fhn_data_raw.parquet")
    fhn_df_filtered = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered.parquet")
    fhn_df_balanced = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered_balanced.parquet")

    # plot tsne by symbol
    plot_tsne_sample_by_symbol(fhn_df_filtered, plots_dir, max_sample_size=1000, filename="tsne_filtered")
    plot_tsne_sample_by_symbol(fhn_df_balanced, plots_dir, max_sample_size=1000, filename="tsne_balanced", col="symbol_categorized", features = features)

    # summarize filtering stats
    # plot_filtering_summary(
    #     fhn_df,
    #     fhn_df_filtered,
    #     loss_threshold,
    #     r2_threshold,
    #     plots_dir
    # )

    _ = save_filtering_summary(
        fhn_df,
        fhn_df_filtered,
        fhn_df_balanced,
        loss_threshold,
        r2_threshold,
        plots_dir
    )

    # compute averages
    raw_avgs = compute_fhn_metric_averages(fhn_df)
    filtered_avgs = compute_fhn_metric_averages(fhn_df_filtered)

    # combine into one table
    avg_df = pd.DataFrame(
        [raw_avgs, filtered_avgs],
        index=["raw", "filtered"]
    )

    # save to CSV
    avg_df.to_csv(f"{plots_dir}/fhn_metric_averages.csv")

def run_model(data_dir, plot_folder, class_names ,label_col="symbol_categorized", features=['a', 'b', 'tau', 'I', 'v0', 'w0', 'qrs_width', 'pt_width']):
    balanced_waves_df = pd.read_parquet(f"{data_dir}/all_fhn_data_filtered_balanced.parquet")

    X_scaled, y, groups = prepare_knn_data(df=balanced_waves_df, 
                                                    label_col=label_col, 
                                                    class_names=class_names,
                                                    features=features)

    # Get/save group counts to CSV
    # group_counts = pd.Series(groups).value_counts()
    # group_counts_df = group_counts.reset_index()
    # group_counts_df.columns = ["group", "count"]
    # group_counts_df.to_csv(f"{plot_folder}/group_counts.csv", index=False)


    y_preds = knn_leave_one_group_out(X_scaled, y, groups)
    acc, report, cm = classification_metrics(y, y_preds, class_names)

    # Save classification report as CSV
    with open(f"{plot_folder}/classification_report.txt", "w") as f:
        f.write(report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, labels = class_names, output_folder=plot_folder)

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(report)

def main():
    # predefined mappings from raw symbols to classification categories
    CATEGORY_MAPS = {
        "binary": {"N": ["N"]},
        "N/LRB": {"N": ["N"], "LRB": ["L", "R", "B"]},
        "N/L": {"N": ["N"], "L": ["L"]},
        "N/L/R/A": {"N": ["N"], "L/R": ["L", "R"], "A":["A"]},
    }
    VALID_CLASSIFICATION_TYPES = list(CATEGORY_MAPS.keys())

    ALL_FEATURES = [
        'a', 'b', 'tau', 'I', 'v0', 'w0', 'qrs_width', 'pt_width'
    ]

    parser = argparse.ArgumentParser(description="ECG Classification Pipeline")

    parser.add_argument("--step", type=str, required=True, nargs="+",
                        choices=["data_stats", "combine", "fhn", "filter", "filtered_fhn_stats", "balance", "model"])

    parser.add_argument("--data_folder", type=str, default="data")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--plots_folder", type=str, default="plots")
    parser.add_argument("--log_file", type=str, default="error_log.txt")

    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--method", type=str, default="undersample", help="oversample/undersample")

    parser.add_argument("--loss_threshold", type=float, default=0.1)
    parser.add_argument("--r2_threshold", type=float, default=0.6)

    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=ALL_FEATURES,
        choices=ALL_FEATURES,
        help="Subset of features to use"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        default="binary",
        choices=VALID_CLASSIFICATION_TYPES,
        help=f"Categories must be one of: {VALID_CLASSIFICATION_TYPES}"
    )


    args = parser.parse_args()
    # ensure output directories exist

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)

    category_map=CATEGORY_MAPS[args.categories]
    selected_features = args.features

    # run corresponding pipeline steps
    steps = args.step
    total_steps = len(steps)
    step_idx = 1

    if "data_stats" in steps:
        log_step(step_idx, total_steps, "Computing dataset statistics")
        run_data_stats(args.data_folder, args.plots_folder)
        step_idx += 1

    if "combine" in steps:
        log_step(step_idx, total_steps, "Combining raw ECG records")
        run_combine(args.data_folder, args.output_folder, args.log_file)
        step_idx += 1

    if "fhn" in steps:
        log_step(step_idx, total_steps, "Fitting FitzHugh–Nagumo model")
        run_fhn(args.data_folder, args.output_folder)
        step_idx += 1

    if "filter" in steps:
        log_step(step_idx, total_steps, "Filtering beats by fit quality")
        run_filter(args.output_folder, args.loss_threshold, args.r2_threshold)
        step_idx += 1

    if "balance" in steps:
        log_step(step_idx, total_steps, "Balancing class distributions")
        run_balance(
            args.output_folder,
            args.max_per_class,
            args.method,
            category_map
        )
        step_idx += 1

    if "filtered_fhn_stats" in steps:
        log_step(step_idx, total_steps, "Generating filtered FHN statistics and visualizations")
        run_filtered_fhn_stats(
            args.output_folder,
            args.plots_folder,
            args.loss_threshold,
            args.r2_threshold,
            selected_features
        )
        step_idx += 1

    if "model" in steps:
        log_step(step_idx, total_steps, "Training and evaluating KNN classifier")
        run_model(
            args.output_folder,
            plot_folder=args.plots_folder,
            class_names=list(category_map.keys()) + ["Other"],
            label_col="symbol_categorized",
            features=selected_features
        )


    

if __name__ == "__main__":
    main()
