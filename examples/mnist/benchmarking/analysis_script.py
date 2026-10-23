import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path(
    "./mnist_benchmarking.csv"
)

df = pd.read_csv(CSV_PATH)

# Keep only the thread counts included in the main analysis.
THREAD_COUNTS = [1, 2, 4, 6, 12]
df = df[df["threads"].isin(THREAD_COUNTS)].copy()

df["threads"] = df["threads"].astype(int)

print(f"Benchmark runs included: {len(df)}")
print(f"Dataset sizes: {sorted(df['samples'].unique())}")
print(f"Thread counts: {THREAD_COUNTS}")

print(df.head())


EPOCHS = 10

analysis_df = df[df["threads"].isin(THREAD_COUNTS)].copy()

summary = (
    analysis_df.groupby(["samples", "threads"])
    .agg(
        median_train=("train_seconds", "median"),
        mean_train=("train_seconds", "mean"),
        std_train=("train_seconds", "std"),
        median_workflow=("workflow_seconds", "median"),
        std_workflow=("workflow_seconds", "std"),
        runs=("run", "count"),
    )
    .reset_index()
)

# Runtime vs threads plot

for samples in sorted(summary["samples"].unique()):
    data = (
        summary[summary["samples"] == samples]
        .sort_values("threads")
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        data["threads"].astype(str),
        data["median_epoch_seconds"],
        marker="o",
    )

    plt.xlabel("Thread count", size=14)
    plt.ylabel("Median time per epoch (seconds)", size=14)
    plt.xticks(size=13)
    plt.yticks(size=13)
    #plt.title(f"MNIST Epoch Time — {samples:,} Samples")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



baseline = (
    summary[summary["threads"] == 1]
    [["samples", "median_train"]]
    .rename(columns={"median_train": "baseline_train"})
)

scaling = summary.merge(baseline, on="samples")

scaling["speedup"] = (
    scaling["baseline_train"] / scaling["median_train"]
)

scaling["efficiency_percent"] = (
    scaling["speedup"] / scaling["threads"] * 100
)

for samples in sorted(scaling["samples"].unique()):
    print(f"\nParallel scaling — {samples:,} samples")

    print(
        scaling[scaling["samples"] == samples][
            [
                "threads",
                "median_train",
                "speedup",
                "efficiency_percent",
            ]
        ]
        .sort_values("threads")
        .round(3)
        .reset_index(drop=True)
    )

# Speedup plot 


for samples in sorted(scaling["samples"].unique()):
    data = (
        scaling[scaling["samples"] == samples]
        .sort_values("threads")
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        data["threads"].astype(str),
        data["speedup"],
        marker="o",
        label="Measured speedup",
    )

    plt.plot(
        data["threads"].astype(str),
        data["threads"],
        linestyle="--",
        marker="",
        label="Ideal linear speedup",
    )

    plt.xlabel("Thread count", size=14)
    plt.ylabel("Speedup (×)", size=14)
    plt.xticks(size=13)
    plt.yticks(size=13)
    #plt.title(f"Parallel Speedup — {samples:,} Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Efficiency plot

for samples in sorted(scaling["samples"].unique()):
    data = (
        scaling[scaling["samples"] == samples]
        .sort_values("threads")
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        data["threads"].astype(str),
        data["efficiency_percent"],
        marker="o",
    )

    plt.axhline(
        y=100,
        linestyle="--",
        label="Ideal efficiency",
    )

    plt.xlabel("Thread count")
    plt.ylabel("Parallel efficiency (%)")
   # plt.title(f"Parallel Efficiency — {samples:,} Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
