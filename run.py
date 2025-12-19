import subprocess
import sys

PYTHON = sys.executable

COMMON_STEPS = [
    "data_stats", "combine", "fhn",
    "filter", "filtered_fhn_stats",
    "balance", "model"
]

CONFIGS = {
    "1": {
        "name": "Binary (Normal vs Abnormal), all features",
        "categories": "binary",
        "features": ["a", "b", "tau", "I", "v0", "w0", "pt_width", "qrs_width"],
        "plots_folder": "config_1_all_features",
    },
    "2": {
        "name": "Binary (Normal vs Abnormal), FHN features only",
        "categories": "binary",
        "features": ["a", "b", "tau", "I", "v0", "w0"],
        "plots_folder": "config_2_fhn_only",
    },
    "3": {
        "name": "Binary (Normal vs Abnormal), width features only",
        "categories": "binary",
        "features": ["pt_width", "qrs_width"],
        "plots_folder": "config_3_width_only",
    },
    "4": {
        "name": "N/L/Other, all features",
        "categories": "N/L",
        "features": ["a", "b", "tau", "I", "v0", "w0", "pt_width", "qrs_width"],
        "plots_folder": "config_4_N_L",
    },
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python run.py <config_number>\n")
        print("Available configurations:")
        for k, v in CONFIGS.items():
            print(f"  {k}: {v['name']}")
        sys.exit(1)

    cfg_id = sys.argv[1]

    if cfg_id not in CONFIGS:
        print(f"Unknown config: {cfg_id}\n")
        print("Available configurations:")
        for k, v in CONFIGS.items():
            print(f"  {k}: {v['name']}")
        sys.exit(1)

    cfg = CONFIGS[cfg_id]

    cmd = [
        PYTHON, "-m", "pipeline",
        "--step", *COMMON_STEPS,
        "--data_folder", "data",
        "--output_folder", "output",
        "--plots_folder", cfg["plots_folder"],
        "--categories", cfg["categories"],
        "--features", *cfg["features"],
        "--loss_threshold", "0.1",
        "--r2_threshold", "0.8",
    ]

    print("\n==============================================")
    print(f"Running configuration {cfg_id}")
    print(cfg["name"])
    print("==============================================")
    print("Command:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
