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
### 1. Cloning the repository
Clone this repository to your local machine:

```bash
git clone https://github.com/yyxnlin/MAT292-final-project
cd MAT292-final-project
```

Alternatively, download this repository as a ```.zip``` and extract it. **The project root directory should be ```MAT292-final-project```.**:
```
mat292-final-project/
├── classification
├── data
  ├── 100.csv
  ├── 101.csv
  ├── ...
├── demo
...
├── pipeline.py
├── run_config_1.ps1
...
```

### 2. Virtual environment (recommended)
It is strongly recommended to use a virtual environment to ensure dependency isolation.

From the **root directory**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Dependencies
All required Python packages are listed in ```requirements.txt```.

When you run any of the provided PowerShell scripts (```run_*.ps1```), dependencies are automatically installed if needed using the active Python environment.

## Data setup
### Option 1: Preprocessed data (recommended for quick reproduction)
To skip the most time-consuming steps (wave extraction + FHN fitting), you can **download the preprocessed dataset** from the Google Drive link above (also [here](https://drive.google.com/drive/folders/1g3bKZenL-nE8pVDSUNLb76B8ccBv6Ad2?usp=drive_link)):

Create an ```output``` folder **under the root directory**, and place the downloaded ```all_fhn_data_raw.parquet``` file inside:
```
output/
├── all_fhn_data_raw.parquet
...
```

### Option 2: Raw data (optional, slower)
If you want to run the entire preprocessing pipeline, place all raw ECG and annotation CSV files in:
```
data/
├── xxx_ekg.csv
├── xxx_annotations_1.csv
...
```

## Running experiments (PowerShell scripts)
Four PowerShell scripts are provided, each corresponding to a configuration used in the report.
All scripts automatically:
- Install dependencies if needed
- Skip preprocessing steps if cached files exist
- Generate plots and tables shown in the report in configuration-specific folders


⚠️ **IMPORTANT:** On some Windows systems, running `.ps1` scripts may be disabled by default.

If you see an error like:
> *Running scripts is disabled on this system*

Run the following command once in PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If it still does not work, the pipeline can still run manually as described in the [Advanced usage (Optional)](#advanced-usage-optional) section below.


### Available configurations
Run the following commands from the project root.

**1. Binary (Normal vs. Abnormal), all features**
```powershell
.\run_config_1.ps1
```

**2. Binary (Normal vs. Abnormal), FHN features only**
```powershell
.\run_config_2.ps1
```

**3. Binary (Normal vs. Abnormal), width features only**
```powershell
.\run_config_3.ps1
```

**4. N/L/Other, all features**
```powershell
.\run_config_4.ps1
```
##  Outputs
Each script produces:
- Intermediate Parquet datasets in output/
- Tables and plots (t-SNE, confusion matrices, class counts, classification reports) in corresponding ```config_*``` folder

**These outputs directly reproduce the figures and results reported in the paper.**

## Advanced usage (Optional)
> ⚠️ **Do not use for standard execution.**  
> The PowerShell scripts provided in this repository can already run experiments. Don't try this unless the powershell doesn't work.

After activating the virtual environment, install dependencies by running the following command:
```powershell
python -m pip install -r requirements.txt
```

The pipeline itself can be run with ```pipeline.py``` and controlled through the `--step` argument, in the following format:
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
