from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("results") / "ppo_per_analysis"
RAW_DIR = OUTPUT_DIR / "raw"

ENV_ID = "CartPole-v1"
ALGORITHMS = {
    "ppo": "PPO",
    "ppo_per": "PPO+PER",
}
SEEDS = range(1, 6)

EXPECTED_LAST_STEP = 499_712
MIN_COMPLETE_STEP = 499_000
RETURN_TAG = "charts/episodic_return"
LOSS_TAG = "losses/policy_loss"
SOLVED_RETURN = 475.0
SOLVED_WINDOW = 100
FINAL_WINDOW = 20
GRID_POINTS = 1001
SMOOTH_WINDOW = 21


@dataclass
class RunData:
    algorithm: str
    label: str
    seed: int
    run_dir: Path
    steps: np.ndarray
    returns: np.ndarray
    last_step: int


def load_accumulator(run_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    return accumulator


def latest_complete_run(algorithm: str, seed: int) -> tuple[Path, EventAccumulator]:
    pattern = f"{ENV_ID}__{algorithm}__{seed}__*"
    candidates = sorted(
        RUNS_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No run found for {algorithm}, seed {seed}: {pattern}")

    errors: list[str] = []
    for run_dir in candidates:
        try:
            accumulator = load_accumulator(run_dir)
            tags = set(accumulator.Tags().get("scalars", []))
            if RETURN_TAG not in tags or LOSS_TAG not in tags:
                errors.append(f"{run_dir}: missing required scalar tags")
                continue

            loss_events = accumulator.Scalars(LOSS_TAG)
            if not loss_events:
                errors.append(f"{run_dir}: empty policy-loss series")
                continue

            last_step = int(loss_events[-1].step)
            if last_step < MIN_COMPLETE_STEP:
                errors.append(f"{run_dir}: incomplete, last_step={last_step}")
                continue

            return run_dir, accumulator
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")

    details = "\n".join(errors)
    raise RuntimeError(
        f"No complete run found for {algorithm}, seed {seed}.\n{details}"
    )


def validate_replay_run(accumulator: EventAccumulator, run_dir: Path) -> None:
    required = [
        "replay/buffer_size",
        "replay/update_count",
        "replay/priority_mean",
    ]
    tags = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in required if tag not in tags]
    if missing:
        raise RuntimeError(f"{run_dir}: missing replay tags {missing}")

    buffers = accumulator.Scalars("replay/buffer_size")
    updates = accumulator.Scalars("replay/update_count")
    priorities = accumulator.Scalars("replay/priority_mean")

    if not buffers or not updates or not priorities:
        raise RuntimeError(f"{run_dir}: one or more replay series are empty")
    if buffers[-1].value != 32.0:
        raise RuntimeError(
            f"{run_dir}: final replay buffer is {buffers[-1].value}, expected 32"
        )
    if max(event.value for event in updates) <= 0:
        raise RuntimeError(f"{run_dir}: replay update_count never became positive")
    if not all(math.isfinite(event.value) for event in priorities):
        raise RuntimeError(f"{run_dir}: non-finite replay priority detected")


def read_run(algorithm: str, seed: int) -> RunData:
    run_dir, accumulator = latest_complete_run(algorithm, seed)

    if algorithm == "ppo_per":
        validate_replay_run(accumulator, run_dir)

    return_events = accumulator.Scalars(RETURN_TAG)
    loss_events = accumulator.Scalars(LOSS_TAG)

    steps = np.asarray([event.step for event in return_events], dtype=np.int64)
    returns = np.asarray([event.value for event in return_events], dtype=np.float64)
    last_step = int(loss_events[-1].step)

    if steps.size == 0:
        raise RuntimeError(f"{run_dir}: no episodic returns")
    if not np.all(np.isfinite(returns)):
        raise RuntimeError(f"{run_dir}: non-finite episodic return detected")

    return RunData(
        algorithm=algorithm,
        label=ALGORITHMS[algorithm],
        seed=seed,
        run_dir=run_dir,
        steps=steps,
        returns=returns,
        last_step=last_step,
    )


def collapse_duplicate_steps(
    steps: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"step": steps, "value": values})
    grouped = frame.groupby("step", sort=True, as_index=False)["value"].mean()
    return (
        grouped["step"].to_numpy(dtype=np.float64),
        grouped["value"].to_numpy(dtype=np.float64),
    )


def interpolate_returns(run: RunData, grid: np.ndarray) -> np.ndarray:
    unique_steps, mean_values = collapse_duplicate_steps(run.steps, run.returns)
    return np.interp(
        grid,
        unique_steps,
        mean_values,
        left=mean_values[0],
        right=mean_values[-1],
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def first_solved_step(run: RunData) -> float:
    rolling = (
        pd.Series(run.returns)
        .rolling(window=SOLVED_WINDOW, min_periods=SOLVED_WINDOW)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    indices = np.flatnonzero(rolling >= SOLVED_RETURN)
    if indices.size == 0:
        return float("nan")
    return float(run.steps[int(indices[0])])


def run_metrics(run: RunData, grid: np.ndarray) -> dict[str, float | int | str]:
    interpolated = interpolate_returns(run, grid)
    normalized_auc = float(np.trapezoid(interpolated, grid) / (grid[-1] - grid[0]))

    final_count = min(FINAL_WINDOW, run.returns.size)
    final_100_count = min(100, run.returns.size)

    return {
        "algorithm": run.algorithm,
        "label": run.label,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "last_step": run.last_step,
        "episodes": int(run.returns.size),
        "overall_mean_return": float(np.mean(run.returns)),
        "final20_mean_return": float(np.mean(run.returns[-final_count:])),
        "final100_mean_return": float(np.mean(run.returns[-final_100_count:])),
        "normalized_auc_return": normalized_auc,
        "solve_step_100ep_at_475": first_solved_step(run),
        "max_return": float(np.max(run.returns)),
    }


def save_raw_csv(run: RunData) -> None:
    frame = pd.DataFrame(
        {
            "global_step": run.steps,
            "episodic_return": run.returns,
            "algorithm": run.algorithm,
            "seed": run.seed,
            "run_dir": str(run.run_dir),
        }
    )
    path = RAW_DIR / f"{run.algorithm}_seed_{run.seed}.csv"
    frame.to_csv(path, index=False)


def summarize_groups(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "episodes",
        "overall_mean_return",
        "final20_mean_return",
        "final100_mean_return",
        "normalized_auc_return",
        "solve_step_100ep_at_475",
        "max_return",
    ]

    rows: list[dict[str, float | int | str]] = []
    for algorithm, group in seed_metrics.groupby("algorithm", sort=False):
        row: dict[str, float | int | str] = {
            "algorithm": algorithm,
            "label": ALGORITHMS[algorithm],
            "seeds": int(group["seed"].nunique()),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_count"] = int(values.notna().sum())
        rows.append(row)

    return pd.DataFrame(rows)


def paired_comparison(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_mean_return",
        "final20_mean_return",
        "final100_mean_return",
        "normalized_auc_return",
        "solve_step_100ep_at_475",
    ]
    rows: list[dict[str, float | int]] = []

    indexed = seed_metrics.set_index(["algorithm", "seed"])
    for seed in SEEDS:
        row: dict[str, float | int] = {"seed": seed}
        for metric in metrics:
            ppo = float(indexed.loc[("ppo", seed), metric])
            ppo_per = float(indexed.loc[("ppo_per", seed), metric])
            row[f"ppo_{metric}"] = ppo
            row[f"ppo_per_{metric}"] = ppo_per
            row[f"delta_per_minus_ppo_{metric}"] = ppo_per - ppo
        rows.append(row)

    return pd.DataFrame(rows)


def plot_learning_curve(runs: Iterable[RunData], grid: np.ndarray) -> Path:
    plt.figure(figsize=(10, 6))

    run_list = list(runs)
    for algorithm, label in ALGORITHMS.items():
        selected = [run for run in run_list if run.algorithm == algorithm]
        curves = np.vstack(
            [
                moving_average(interpolate_returns(run, grid), SMOOTH_WINDOW)
                for run in selected
            ]
        )
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0, ddof=1)

        line = plt.plot(grid, mean_curve, label=f"{label} mean")[0]
        plt.fill_between(
            grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=line.get_color(),
            label=f"{label} ±1 SD",
        )

    plt.axhline(
        SOLVED_RETURN,
        linestyle="--",
        linewidth=1,
        label=f"Solved threshold ({SOLVED_RETURN:.0f})",
    )
    plt.xlabel("Global environment steps")
    plt.ylabel("Episodic return")
    plt.title(f"{ENV_ID}: PPO vs PPO+PER ({len(list(SEEDS))} seeds)")
    plt.xlim(0, EXPECTED_LAST_STEP)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "ppo_vs_ppo_per_learning_curve.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_final20(seed_metrics: pd.DataFrame) -> Path:
    labels = [ALGORITHMS["ppo"], ALGORITHMS["ppo_per"]]
    groups = [
        seed_metrics.loc[
            seed_metrics["algorithm"] == "ppo",
            "final20_mean_return",
        ].to_numpy(dtype=np.float64),
        seed_metrics.loc[
            seed_metrics["algorithm"] == "ppo_per",
            "final20_mean_return",
        ].to_numpy(dtype=np.float64),
    ]
    means = [values.mean() for values in groups]
    stds = [values.std(ddof=1) for values in groups]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6)

    for index, values in enumerate(groups):
        offsets = np.linspace(-0.08, 0.08, num=len(values))
        plt.scatter(np.full_like(values, x[index], dtype=float) + offsets, values, zorder=3)

    plt.xticks(x, labels)
    plt.ylabel(f"Mean return over final {FINAL_WINDOW} episodes")
    plt.title(f"{ENV_ID}: final performance across seeds")
    plt.ylim(bottom=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    path = OUTPUT_DIR / "ppo_vs_ppo_per_final20.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[RunData] = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            run = read_run(algorithm, seed)
            runs.append(run)
            save_raw_csv(run)
            print(
                f"Loaded {run.label} seed {seed}: "
                f"episodes={run.returns.size}, last_step={run.last_step}, "
                f"run={run.run_dir}"
            )

    grid = np.linspace(0, EXPECTED_LAST_STEP, GRID_POINTS, dtype=np.float64)

    seed_metrics = pd.DataFrame([run_metrics(run, grid) for run in runs])
    seed_metrics_path = OUTPUT_DIR / "seed_metrics.csv"
    seed_metrics.to_csv(seed_metrics_path, index=False)

    group_summary = summarize_groups(seed_metrics)
    group_summary_path = OUTPUT_DIR / "group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    paired = paired_comparison(seed_metrics)
    paired_path = OUTPUT_DIR / "paired_seed_comparison.csv"
    paired.to_csv(paired_path, index=False)

    curve_path = plot_learning_curve(runs, grid)
    final20_path = plot_final20(seed_metrics)

    columns_to_print = [
        "label",
        "seeds",
        "final20_mean_return_mean",
        "final20_mean_return_std",
        "final100_mean_return_mean",
        "final100_mean_return_std",
        "normalized_auc_return_mean",
        "normalized_auc_return_std",
        "solve_step_100ep_at_475_mean",
        "solve_step_100ep_at_475_std",
    ]

    print("\nGroup summary:")
    print(group_summary[columns_to_print].to_string(index=False))
    print("\nCreated:")
    for path in [
        seed_metrics_path,
        group_summary_path,
        paired_path,
        curve_path,
        final20_path,
        RAW_DIR,
    ]:
        print(path)


if __name__ == "__main__":
    main()
