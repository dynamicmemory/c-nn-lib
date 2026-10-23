import subprocess
import csv
import re
import sys 
import random 
from pathlib import Path 

RUNS = 10

SAMPLE_COUNTS = [5_000, 60_000]
EPOCHS = 10 
LEARNING_RATE = 0.2 

THREAD_COUNTS = [1, 2, 4, 6, 12, 24]

ROOT = Path(__file__).resolve().parents[3]
EXECUTABLE = ROOT / "build" / "mnist"
OUTPUT = ROOT/"examples" / "mnist" / "benchmarking"/ "mnsit_benchmarking.csv"

# Recognise the machine-readable timing lines from main.c
TRAIN_RE = re.compile(r"^TRAIN_SECONDS=([\d.eE+-]+)$", re.MULTILINE)
WORKFLOW_RE = re.compile(r"^WORKFLOW_SECONDS=([\d.eE+-]+)$", re.MULTILINE)


def run_benchmark(samples, threads, run, index, jobs):
    command = [
        str(EXECUTABLE),
        str(samples),
        str(EPOCHS),
        str(threads),
        str(LEARNING_RATE),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Benchmark failed: {command}")

    train_match = TRAIN_RE.search(result.stdout)
    workflow_match = WORKFLOW_RE.search(result.stdout)

    if not train_match or not workflow_match:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("Could not find both timing values in program output")

    train_seconds = float(train_match.group(1))
    workflow_seconds = float(workflow_match.group(1))

    print(
        f"Benchmark {index}/{jobs} (Rep {run}/{RUNS}): "
        f"samples={samples}, threads={threads}, "
        f"train={train_seconds:.3f}s, "
        f"workflow={workflow_seconds:.3f}s"
    )

    return train_seconds, workflow_seconds


def main():
    if not EXECUTABLE.is_file():
        sys.exit(
            f"Executable not found: {EXECUTABLE}\n"
            "Build the project first."
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Randomise configuration order to reduce ordering bias.
    # The fixed seed makes the run order reproducible.
    rng = random.Random(42)
    jobs = [
        (samples, threads, run)
        for run in range(1, RUNS + 1)
        for samples in SAMPLE_COUNTS
        for threads in THREAD_COUNTS
    ]
    rng.shuffle(jobs)

    columns = [
        "samples",
        "epochs",
        "learning_rate",
        "threads",
        "run",
        "train_seconds",
        "workflow_seconds",
    ]

    print(f"Running {len(jobs)} benchmarks")
    print(f"Writing results to {OUTPUT}")

    # Start a fresh results file for each benchmark campaign.
    # Flush after each run so completed results survive interruptions.
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        file.flush()

        for index, (samples, threads, run) in enumerate(jobs, start=1):
            train_seconds, workflow_seconds = run_benchmark(
                samples, threads, run, index, len(jobs)
            )

            writer.writerow({
                "samples": samples,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "threads": threads,
                "run": run,
                "train_seconds": train_seconds,
                "workflow_seconds": workflow_seconds,
            })

            file.flush()

    print(f"\nFinished. Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
