import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def plot_counts_stacked(
    df: pd.DataFrame,
    output_folder: str = "plots",
    filename: str = "annotations_count"
):
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{filename}.png")

    df.plot(kind='bar', stacked=True, figsize=(18, 8), colormap='tab20')
    plt.xlabel("File")
    plt.ylabel("Number of annotations")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.show()


def plot_ecg_beats(ecg, rpeaks, waves,
                    output_folder: str = "plots",
                    filename: str = "ecg_beats",
                    fs=360, 
                    num_beats_to_plot=3):
    """
    Plot the first few ECG beats with the different PQRST segments from Neurokit2 labelled.
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{filename}.png")


    if len(rpeaks) < num_beats_to_plot + 1:
        raise ValueError("Not enough R-peaks to plot multiple beats.")

    start = rpeaks[0] - int(0.2 * fs)  # start slightly before first R
    end   = rpeaks[num_beats_to_plot] + int(0.4 * fs)
    start = max(start, 0)
    end   = min(end, len(ecg))

    plt.figure(figsize=(14, 5))
    t = np.arange(start, end) / fs
    plt.plot(t, ecg[start:end], label="ECG")

    # Helper function to plot points
    def plot_points(indices, name, color):
        pts = [i for i in indices if start <= i < end]
        if pts:
            plt.scatter(np.array(pts) / fs, ecg[pts], s=40, label=name, color=color)

    # Plot R-peaks
    plot_points(rpeaks, "ECG_R_Peaks", "black")

    # Plot fiducial points
    colors = {
        "ECG_P_Peaks": "green",
        "ECG_P_Onsets": "lightgreen",
        "ECG_P_Offsets": "darkgreen",
        "ECG_Q_Peaks": "purple",
        "ECG_R_Onsets": "blue",
        "ECG_R_Offsets": "navy",
        "ECG_S_Peaks": "red",
        "ECG_T_Peaks": "orange",
        "ECG_T_Onsets": "gold",
        "ECG_T_Offsets": "brown",
    }

    for key in waves.keys():
        if key in colors:
            plot_points(waves[key], key, colors[key])

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.show()




def plot_single_beat(ecg, row, output_folder="plots", filename="ecg_beat", fs=360):
    """
    Plot a single ECG beat with PQRST segments labeled from a waves_df row.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal array.
    row : pd.Series
        One row from waves_df containing fiducial points for the beat.
    output_folder : str
        Folder to save the figure.
    filename : str
        Name of the saved figure (PNG).
    fs : int
        Sampling frequency in Hz.
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{filename}.png")

    # Define start and end indices for plotting
    rpeak = int(row["rpeak"])
    start = max(rpeak - int(0.2 * fs), 0)
    end = min(rpeak + int(0.4 * fs), len(ecg))

    plt.figure(figsize=(10, 4))
    t = np.arange(start, end) / fs
    plt.plot(t, ecg[start:end], label="ECG")

    # Helper function to plot points
    def plot_point(idx, name, color):
        if pd.notna(idx) and start <= idx < end:
            plt.scatter(idx/fs, ecg[int(idx)], s=50, label=name, color=color)

    # Fiducial point colors
    colors = {
        "ECG_P_Peaks": "green",
        "ECG_P_Onsets": "lightgreen",
        "ECG_P_Offsets": "darkgreen",
        "ECG_Q_Peaks": "purple",
        "ECG_R_Onsets": "blue",
        "ECG_R_Offsets": "navy",
        "rpeak": "black",
        "ECG_S_Peaks": "red",
        "ECG_T_Peaks": "orange",
        "ECG_T_Onsets": "gold",
        "ECG_T_Offsets": "brown",
    }

    # Plot each fiducial point
    for key, color in colors.items():
        if key in row.index:
            plot_point(row[key], key, color)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.show()




def plot_confusion_matrix(cm, labels=['N','not N'], title="Normalized Confusion Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()