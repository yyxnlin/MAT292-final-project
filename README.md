# MAT292 Final Project: Predicting Heart Rhythms With ECG Data

This project implements a multi-step electrocardiogram-classification pipeline. Raw ECG waveforms are segmented into beats, fit with the FitzHugh-Nagumo (FHN) model, and used to train a KNN classifier to predict heartbeat categories.

## Data sources
The data used in this project comes from the [MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/protobioengineering/mit-bih-arrhythmia-database-modern-2023). 

The dataset includes:
* ECG waveform files (`*_ekg.csv`), containing raw ECG signal
* Annotation files (`*_annotations_1.csv`), containing beat-level labels corresponding to each ECG waveform file

Place all downloaded CSVs in the `data/` directory before running the pipeline.

**IMPORTANT:** To speed up preprocessing steps, the preprocessed data (```all_fhn_data_raw.parquet```) can be downloaded directly from this
[Google Drive](https://drive.google.com/drive/folders/1g3bKZenL-nE8pVDSUNLb76B8ccBv6Ad2?usp=drive_link)
link.

## Installation + setup
### 1. Install dependencies
```powershell
pip install -r requirements.txt
```

### 2. Directory structure
All raw input `.csv` files should be placed in:
```
data/
├── xxx_ekg.csv
├── xxx_annotations_1.csv
...
```

## Option 1: Quick start (using downloaded preprocessed data)
This option is recommended if you want to reproduce figures and classification results without running the long preprocessing steps.
Place the downloaded ```all_fhn_data_raw.parquet``` file under the ```output``` folder:
```
output/
├── all_fhn_data_raw.parquet
...
```

To get the results for the **binary** classification model, run the following in command line:
```
python -m pipeline \
    --step filter filtered_fhn_stats balance model \
    --data_folder data \
    --output_folder output \
    --plots_folder plots_binary \
    --categories binary \
    --method undersample \
    --loss_threshold 0.1 \
    --r2_threshold 0.8
```

To get the results for the **N/L/Other** classification model, run the following in command line:
```
python -m pipeline \
    --step filter filtered_fhn_stats balance model \
    --data_folder data \
    --output_folder output \
    --plots_folder plots_binary \
    --categories N/L \
    --method undersample \
    --loss_threshold 0.1 \
    --r2_threshold 0.8
```

* Produces intermediate Parquet files in `output/`
* Generates all plots and tables in `plots/`
* **Note:** This skips the ```combine``` and ```filter``` steps as the raw data has already been downloaded.
This is the minimal command to reproduce all results in the report.


## Option 2: Run everything
To run the **entire pipeline** in one command:
```
python -m pipeline \
    --step data_stats combine fhn filter filtered_fhn_stats balance model \
    --data_folder data \
    --output_folder output \
    --plots_folder plots \
    --categories binary \
    --method undersample \
    --loss_threshold 0.1 \
    --r2_threshold 0.8
```
* Produces intermediate Parquet files in `output/`
* Generates plots (t-SNE, confusion matrices, counts) in `plots/`

This is the minimal command to reproduce all results in the report.

## Option 3: Running the pipeline step-by-step
The workflow is controlled through the `--step` argument, in the following format:
```
python -m pipeline --step <steps...> [options]
```

You can run **one or multiple steps** at a time.

Example:
```powershell
python -m pipeline --step data_stats combine 
```
The above code will execute the `data_stats` and `combine` functions of the pipeline.

### Pipeline steps
| Step                 | Description                                                       | Output File / Folder                                      |
|----------------------|-------------------------------------------------------------------|------------------------------------------------------------|
| `data_stats`         | Computes basic dataset statistics (symbol counts).                | Plots saved in `plots/`                                   |
| `combine`            | Extracts waves and combines all records into a single dataset.    | `output/all_waves_raw.parquet`                            |
| `fhn`                | Fits FHN parameters to each beat.                                 | `output/all_fhn_data_raw.parquet`                         |
| `filter`             | Filters beats by loss and R² thresholds.                          | `output/all_fhn_data_filtered.parquet`                    |
| `filtered_fhn_stats` | Generates t-SNE visualizations of FHN parameters.                 | Plots saved in `plots/`                                   |
| `balance`            | Balances the dataset using over/undersampling.                    | `output/all_fhn_data_filtered_balanced.parquet`           |
| `model`              | Trains/evaluates KNN classifier (leave-one-record-out).           | Confusion matrix in `plots/`, metrics printed to terminal |


### Feature definitions

In total, there are 8 features defined:  
`a`, `b`, `tau`, `I`, `v0`, `w0`, `qrs_width`, `pt_width`.

A subset of them can be chosen for the classification model using the `--features` setting. A description of them is provided below.

### FitzHugh–Nagumo (FHN) parameters

Each beat’s QRS segment is fit with the FitzHugh–Nagumo model:

```math
\begin{aligned}
\dot v &= v - \frac{v^3}{3} - w + I \\
\dot w &= \frac{v + a - b w}{\tau}
\end{aligned}
```

Thus, there are features `a`, `b`, `tau`, `I`, `v0`, `w0`

### Time-domain waveform features
| Feature | Description |
|------|------------|
| `qrs_width` | duration between Q and S peaks |
| `pt_width` | duration between P onset and T offset |

## Pipeline options
These have defaults, but you may override them:
| Argument           | Default         | Description                                                       |
| ------------------ | --------------- | ----------------------------------------------------------------- |
| `--data_folder`    | `data`          | where ECG + annotation CSVs are stored                            |
| `--output_folder`  | `output`        | where parquet files are written                                   |
| `--plots_folder`   | `plots`         | where figures are saved                                           |
| `--log_file`       | `error_log.txt` | logging file                                                      |
| `--categories`     | `binary`        | one of `VALID_CLASSIFICATION_TYPES` (e.g. `binary`, `N/L`, etc. defined in `main()`) |
| `--features`     | all        | FHN/width features used for classifier (`a`, `b`, `tau`, `I`, `v0`, `w0`, `qrs_width`, `pt_width`)|
| `--max_per_class`  | `None`          | cap on per-class size during balancing                            |
| `--method`         | `undersample`   | balancing method (`oversample` / `undersample`)                   |
| `--loss_threshold` | `0.1`           | FHN loss filter threshold                                         |
| `--r2_threshold`   | `0.6`           | Minimum R² value to keep a beat                                   |

### Examples for individual steps
```
# Dataset statistics
python -m pipeline --step data_stats

# Combine all records
python -m pipeline --step combine

# Fit FHN
python -m pipeline --step fhn

# Filter by quality
python -m pipeline --step filter --loss_threshold 0.1 --r2_threshold 0.6

# t-SNE plots of FHN parameters
python -m pipeline --step filtered_fhn_stats

# Balance the dataset
python -m pipeline --step balance --method undersample --max_per_class 1000

# Train & evaluate KNN classifier
python -m pipeline --step model
```
