import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
HORIZON_EXP_DIR = PROJECT_DIR / "experiments" / "horizon_ablation"
sys.path.insert(0, str(HORIZON_EXP_DIR))

import run_horizon_ablation as core


RUNS_DIR = EXP_DIR / "runs"
TABLE_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"

DEFAULT_WINDOWS = [30, 60, 120, 180, 240]
FIXED_HORIZON = 10


def configure_core_paths() -> None:
    core.EXP_DIR = EXP_DIR
    core.RUNS_DIR = RUNS_DIR
    core.TABLE_DIR = TABLE_DIR
    core.FIG_DIR = FIG_DIR


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(windows: list[int]) -> None:
    configure_core_paths()
    reference_cols = core.read_reference_columns()
    normal_files = sorted(core.CLEAN_NORMAL_DIR.glob("*.csv"))
    infected_files = sorted(core.CLEAN_INFECTED_DIR.glob("*.csv"))
    mixed_files = sorted(core.CLEAN_MIXED_DIR.glob("*.csv"))

    print("Window ablation dry run")
    print("Project dir:", PROJECT_DIR)
    print("Experiment dir:", EXP_DIR)
    print("Device:", core.DEVICE)
    print("Windows:", windows)
    print("Fixed H:", FIXED_HORIZON, "S:", core.S)
    print("Feature count:", len(reference_cols))
    print("Normal files:", len(normal_files))
    print("Infected files:", len(infected_files))
    print("Mixed files:", len(mixed_files))
    print()

    for window in windows:
        total_windows = 0
        for path in normal_files:
            df = core.validate_columns(path, reference_cols)
            total_windows += max(0, (len(df) - (window + FIXED_HORIZON)) // core.S + 1)
        n_train = int(total_windows * 0.8)
        print(
            f"T={window}: normal windows={total_windows}, "
            f"train={n_train}, val={total_windows - n_train}"
        )


def save_aggregate_outputs(rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("T")
    csv_path = TABLE_DIR / "window_ablation_summary.csv"
    md_path = TABLE_DIR / "window_ablation_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)

    x = df["T"].to_numpy()

    plt.figure(figsize=(9, 5))
    plt.plot(x, df["mixed_mean_precision"], marker="o", label="precision")
    plt.plot(x, df["mixed_mean_recall"], marker="o", label="recall")
    plt.plot(x, df["mixed_mean_f1"], marker="o", label="F1")
    plt.xlabel("Input window T, seconds")
    plt.ylabel("metric")
    plt.ylim(0, 1.05)
    plt.title("Window ablation: mixed metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "window_ablation_metrics.png", dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, df["mixed_mean_FPR"], marker="o", color="tab:red", label="FPR")
    ax1.set_xlabel("Input window T, seconds")
    ax1.set_ylabel("FPR", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, df["mixed_mean_detection_delay_sec"], marker="o", color="tab:blue", label="delay")
    ax2.set_ylabel("Detection delay, seconds", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Window ablation: FPR and detection delay")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "window_ablation_fpr_delay.png", dpi=300)
    plt.close()

    print("Aggregate outputs:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", FIG_DIR / "window_ablation_metrics.png")
    print(" -", FIG_DIR / "window_ablation_fpr_delay.png")


def run(args) -> None:
    configure_core_paths()
    core.set_seed(args.seed)
    reference_cols = core.read_reference_columns()

    if args.dry_run:
        dry_run(args.windows)
        return

    aggregate_rows = []
    for window in args.windows:
        core.T = window
        run_name = f"T{window:03d}"
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 80)
        print(f"Input window T={window}, fixed H={FIXED_HORIZON}")
        print("=" * 80)

        dataset_summary = core.build_dataset(FIXED_HORIZON, run_dir, reference_cols)
        print(
            f"Dataset: total={dataset_summary['total_windows']} "
            f"train={dataset_summary['train_windows']} val={dataset_summary['val_windows']}"
        )

        train_summary = core.train_model(
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )
        val_stats = core.score_validation(run_dir)
        infected_results = core.evaluate_infected(run_dir, reference_cols, FIXED_HORIZON)
        mixed_results = core.evaluate_mixed(run_dir, reference_cols, FIXED_HORIZON)

        infected_positive_rate = float(
            np.mean([x["positive_window_rate"] for x in infected_results["files"]])
        )
        mixed_agg = mixed_results["aggregate_metrics"]

        row = {
            "T": window,
            "H": FIXED_HORIZON,
            "train_windows": dataset_summary["train_windows"],
            "val_windows": dataset_summary["val_windows"],
            "best_epoch": train_summary["best_epoch"],
            "best_val_loss": train_summary["best_val_loss"],
            "threshold": val_stats["recommended_threshold"],
            "infected_mean_positive_rate": infected_positive_rate,
            "mixed_mean_precision": mixed_agg["mean_precision"],
            "mixed_mean_recall": mixed_agg["mean_recall"],
            "mixed_mean_f1": mixed_agg["mean_f1_score"],
            "mixed_mean_FPR": mixed_agg["mean_false_positive_rate"],
            "mixed_mean_detection_delay_sec": mixed_agg["mean_detection_delay_sec"],
        }
        aggregate_rows.append(row)

        print("Window result:")
        print(
            f"  infected_positive_rate={infected_positive_rate:.4f} "
            f"mixed_precision={row['mixed_mean_precision']:.4f} "
            f"mixed_recall={row['mixed_mean_recall']:.4f} "
            f"mixed_F1={row['mixed_mean_f1']:.4f} "
            f"mixed_FPR={row['mixed_mean_FPR']:.4f} "
            f"delay={row['mixed_mean_detection_delay_sec']:.1f}s"
        )

    save_aggregate_outputs(aggregate_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated input-window ablation experiment.")
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
