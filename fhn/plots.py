import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_counts_stacked(
    df: pd.DataFrame,
    output_folder: str = "plots",
    filename: str = "annotations_count"
):
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, f"{filename}.png")

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for i in range(len(df)):
        # Sort values for this row ascending
        row_sorted = df.iloc[i].sort_values(ascending=True)
        for col, value in row_sorted.items():
            ax.bar(x[i], value, bottom=bottom[i], color=plt.cm.Set2(df.columns.get_loc(col)/len(df.columns)), label=col if i == 0 else "")
            bottom[i] += value

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_xlabel("File", fontsize=16)
    ax.set_ylabel("Number of annotations", fontsize=16)
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.08, 1),
        fontsize=16,
        ncol=1
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)


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




def plot_confusion_matrix(cm, labels=['N','not N'], output_folder="plots"):
    plt.figure(figsize=(5,4))
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"fontsize": 12})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    ax.tick_params(axis='x', labelsize=12, rotation=0) 
    ax.tick_params(axis='y', labelsize=12)

    plt.savefig(f"{output_folder}/confusion_matrix.png", dpi=300)


    
def plot_tsne_sample_by_symbol(df, output_folder, max_sample_size=1000, features=None):
    if features is None:
        features = ['a','b','tau','I','v0','w0']

    # --- Random subsample (without replacement) ---
    if len(df) > max_sample_size:
        df = df.sample(n=max_sample_size, random_state=42)
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Run t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    tsne_coords = tsne.fit_transform(X_scaled)

    # --- Build plot dataframe ---
    plot_df = pd.DataFrame({
        "tsne1": tsne_coords[:,0],
        "tsne2": tsne_coords[:,1],
        "symbol": df["symbol"].values
    })

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    for sym in plot_df["symbol"].unique():
        mask = plot_df["symbol"] == sym
        plt.scatter(
            plot_df.loc[mask, "tsne1"],
            plot_df.loc[mask, "tsne2"],
            label=sym,
            alpha=0.7,
            s=20
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.14, 1),
        fontsize=15,
        ncol=1
    )
    plt.grid(True)
    plt.savefig(f"{output_folder}/tsne.png", dpi=300)


def plot_filtering_summary(
    df_raw,
    df_filt,
    loss_threshold,
    r2_threshold,
    output_folder
):
    # Filtered data
    L_raw = df_raw["loss"].dropna()
    L_filt = df_filt["loss"].dropna()
    R_raw = df_raw["r2"].dropna()
    R_filt = df_filt["r2"].dropna()

    # Keep only reasonable ranges
    L_raw = L_raw[(L_raw >= 0) & (L_raw <= 5)]
    L_filt = L_filt[(L_filt >= 0) & (L_filt <= 5)]
    R_raw = R_raw[(R_raw >= 0) & (R_raw <= 1)]
    R_filt = R_filt[(R_filt >= 0) & (R_filt <= 1)]

    # Define plots info: (data, xlabel, xlim, color, filename)
    plots = [
        (L_raw, "Loss L", (0, 5), "#4C72B0", "loss_raw.png"),
        (L_filt, "Loss L", (0, loss_threshold), "#55A868", "loss_filtered.png"),
        (R_raw, "$R^2$", (0, 1), "#4C72B0", "r2_raw.png"),
        (R_filt, "$R^2$", (r2_threshold, 1), "#55A868", "r2_filtered.png")
    ]

    for data, xlabel, xlim, color, fname in plots:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(data, bins=50, color=color, alpha=0.7)
        
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel("Count", fontsize=20)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        
        # Reduce number of ticks
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
        
        # Format tick labels to max 2 decimal places
        fmt = lambda x, _: ('%.2f' % x).rstrip('0').rstrip('.')  # only keep decimals if necessary
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt))
        
        # Tick font size
        ax.tick_params(axis='both', labelsize=19)
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{fname}")
        plt.close()