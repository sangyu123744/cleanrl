from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, wilcoxon
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import analyze_ppo_per as base


base.OUTPUT_DIR = Path("results") / "ppo_replay_ablation"
base.RAW_DIR = base.OUTPUT_DIR / "raw"
base.ALGORITHMS = {
    "ppo": "PPO",
    "ppo_uniform_replay_ablation": "PPO + uniform replay",
    "ppo_prioritized_replay_ablation": "PPO + prioritized replay",
}

REPLAY_ALGORITHMS = {
    "ppo_uniform_replay_ablation",
    "ppo_prioritized_replay_ablation",
}
PAIRWISE_COMPARISONS = [
    (
        "uniform_minus_ppo",
        "ppo_uniform_replay_ablation",
        "ppo",
        "Uniform replay - PPO",
    ),
    (
        "prioritized_minus_uniform",
        "ppo_prioritized_replay_ablation",
        "ppo_uniform_replay_ablation",
        "Prioritized replay - uniform replay",
    ),
    (
        "prioritized_minus_ppo",
        "ppo_prioritized_replay_ablation",
        "ppo",
        "Prioritized replay - PPO",
    ),
]
METRICS = [
    "overall_mean_return",
    "final20_mean_return",
    "final100_mean_return",
    "normalized_auc_return",
    "solve_step_100ep_at_475",
]


def validate_replay_run(
    accumulator: EventAccumulator,
    run_dir: Path,
    algorithm: str,
) -> None:
    required = [
        "replay/buffer_size",
        "replay/update_count",
        "replay/priority_mean",
        "replay/rollout_count",
        "replay/sample_weight_min",
        "replay/sample_weight_max",
    ]
    tags = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in required if tag not in tags]
    if missing:
        raise RuntimeError(f"{run_dir}: missing replay tags {missing}")

    buffers = accumulator.Scalars("replay/buffer_size")
    updates = accumulator.Scalars("replay/update_count")
    priorities = accumulator.Scalars("replay/priority_mean")
    rollout_counts = accumulator.Scalars("replay/rollout_count")
    weight_mins = accumulator.Scalars("replay/sample_weight_min")
    weight_maxs = accumulator.Scalars("replay/sample_weight_max")

    series = [buffers, updates, priorities, rollout_counts, weight_mins, weight_maxs]
    if any(not values for values in series):
        raise RuntimeError(f"{run_dir}: one or more replay series are empty")
    if buffers[-1].value != 32.0:
        raise RuntimeError(
            f"{run_dir}: final replay buffer is {buffers[-1].value}, expected 32"
        )
    if rollout_counts[-1].value != 4.0:
        raise RuntimeError(
            f"{run_dir}: final rollout_count is {rollout_counts[-1].value}, expected 4"
        )
    if max(event.value for event in updates) <= 0:
        raise RuntimeError(f"{run_dir}: replay update_count never became positive")
    if not all(
        math.isfinite(event.value)
        for values in [priorities, weight_mins, weight_maxs]
        for event in values
    ):
        raise RuntimeError(f"{run_dir}: non-finite replay statistic detected")

    if algorithm == "ppo_uniform_replay_ablation":
        if not all(
            abs(event.value - 1.0) < 1e-6
            for values in [weight_mins, weight_maxs]
            for event in values
        ):
            raise RuntimeError(f"{run_dir}: uniform replay weights are not all 1")

    if algorithm == "ppo_prioritized_replay_ablation":
        if not any(
            low.value < high.value - 1e-6
            for low, high in zip(weight_mins, weight_maxs)
        ):
            raise RuntimeError(f"{run_dir}: no non-uniform replay weights observed")


def read_run(algorithm: str, seed: int) -> base.RunData:
    run_dir, accumulator = base.latest_complete_run(algorithm, seed)
    if algorithm in REPLAY_ALGORITHMS:
        validate_replay_run(accumulator, run_dir, algorithm)

    return_events = accumulator.Scalars(base.RETURN_TAG)
    loss_events = accumulator.Scalars(base.LOSS_TAG)
    steps = np.asarray([event.step for event in return_events], dtype=np.int64)
    returns = np.asarray([event.value for event in return_events], dtype=np.float64)
    last_step = int(loss_events[-1].step)

    if steps.size == 0:
        raise RuntimeError(f"{run_dir}: no episodic returns")
    if not np.all(np.isfinite(returns)):
        raise RuntimeError(f"{run_dir}: non-finite episodic return detected")

    return base.RunData(
        algorithm=algorithm,
        label=base.ALGORITHMS[algorithm],
        seed=seed,
        run_dir=run_dir,
        steps=steps,
        returns=returns,
        last_step=last_step,
    )


def paired_seed_comparison(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = seed_metrics.set_index(["algorithm", "seed"])
    rows = []
    for comparison, left_algorithm, right_algorithm, label in PAIRWISE_COMPARISONS:
        for seed in base.SEEDS:
            row = {
                "comparison": comparison,
                "comparison_label": label,
                "left_algorithm": left_algorithm,
                "right_algorithm": right_algorithm,
                "seed": seed,
            }
            for metric in METRICS:
                left_value = float(indexed.loc[(left_algorithm, seed), metric])
                right_value = float(indexed.loc[(right_algorithm, seed), metric])
                row[f"left_{metric}"] = left_value
                row[f"right_{metric}"] = right_value
                row[f"delta_{metric}"] = left_value - right_value
            rows.append(row)
    return pd.DataFrame(rows)


def two_sided_sign_test(deltas: np.ndarray) -> tuple[float, int]:
    nonzero = deltas[np.abs(deltas) > 1e-12]
    if nonzero.size == 0:
        return 1.0, 0
    positives = int(np.sum(nonzero > 0))
    result = binomtest(positives, int(nonzero.size), p=0.5, alternative="two-sided")
    return float(result.pvalue), int(nonzero.size)


def paired_statistical_tests(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = seed_metrics.set_index(["algorithm", "seed"])
    rows = []

    for comparison, left_algorithm, right_algorithm, label in PAIRWISE_COMPARISONS:
        for metric in METRICS:
            left_values = np.asarray(
                [float(indexed.loc[(left_algorithm, seed), metric]) for seed in base.SEEDS],
                dtype=np.float64,
            )
            right_values = np.asarray(
                [float(indexed.loc[(right_algorithm, seed), metric]) for seed in base.SEEDS],
                dtype=np.float64,
            )
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            left_values = left_values[valid]
            right_values = right_values[valid]
            deltas = left_values - right_values

            if deltas.size == 0:
                rows.append(
                    {
                        "comparison": comparison,
                        "comparison_label": label,
                        "metric": metric,
                        "n_pairs": 0,
                    }
                )
                continue

            t_result = ttest_rel(left_values, right_values)
            sign_p, sign_n = two_sided_sign_test(deltas)

            if np.all(np.abs(deltas) <= 1e-12):
                wilcoxon_stat = 0.0
                wilcoxon_p = 1.0
            else:
                wilcoxon_result = wilcoxon(
                    deltas,
                    zero_method="wilcox",
                    alternative="two-sided",
                )
                wilcoxon_stat = float(wilcoxon_result.statistic)
                wilcoxon_p = float(wilcoxon_result.pvalue)

            rows.append(
                {
                    "comparison": comparison,
                    "comparison_label": label,
                    "metric": metric,
                    "n_pairs": int(deltas.size),
                    "left_mean": float(np.mean(left_values)),
                    "right_mean": float(np.mean(right_values)),
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                    "positive_deltas": int(np.sum(deltas > 0)),
                    "negative_deltas": int(np.sum(deltas < 0)),
                    "zero_deltas": int(np.sum(np.abs(deltas) <= 1e-12)),
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_pvalue_two_sided": float(t_result.pvalue),
                    "sign_test_n_nonzero": sign_n,
                    "sign_test_pvalue_two_sided": sign_p,
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_pvalue_two_sided": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curve(runs: Iterable[base.RunData], grid: np.ndarray) -> Path:
    plt.figure(figsize=(10, 6))
    run_list = list(runs)

    for algorithm, label in base.ALGORITHMS.items():
        selected = [run for run in run_list if run.algorithm == algorithm]
        curves = np.vstack(
            [
                base.moving_average(
                    base.interpolate_returns(run, grid),
                    base.SMOOTH_WINDOW,
                )
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
        base.SOLVED_RETURN,
        linestyle="--",
        linewidth=1,
        label=f"Solved threshold ({base.SOLVED_RETURN:.0f})",
    )
    plt.xlabel("Global environment steps")
    plt.ylabel("Episodic return")
    plt.title(f"{base.ENV_ID}: PPO replay ablation ({len(list(base.SEEDS))} seeds)")
    plt.xlim(0, base.EXPECTED_LAST_STEP)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    path = base.OUTPUT_DIR / "ppo_replay_ablation_learning_curve.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_final20(seed_metrics: pd.DataFrame) -> Path:
    labels = [base.ALGORITHMS[algorithm] for algorithm in base.ALGORITHMS]
    groups = [
        seed_metrics.loc[
            seed_metrics["algorithm"] == algorithm,
            "final20_mean_return",
        ].to_numpy(dtype=np.float64)
        for algorithm in base.ALGORITHMS
    ]
    means = [values.mean() for values in groups]
    stds = [values.std(ddof=1) for values in groups]

    plt.figure(figsize=(9, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=6)

    for index, values in enumerate(groups):
        offsets = np.linspace(-0.08, 0.08, num=len(values))
        plt.scatter(
            np.full_like(values, x[index], dtype=float) + offsets,
            values,
            zorder=3,
        )

    plt.xticks(x, labels, rotation=10)
    plt.ylabel(f"Mean return over final {base.FINAL_WINDOW} episodes")
    plt.title(f"{base.ENV_ID}: final performance across seeds")
    plt.ylim(bottom=0)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    path = base.OUTPUT_DIR / "ppo_replay_ablation_final20.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.RAW_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for algorithm in base.ALGORITHMS:
        for seed in base.SEEDS:
            run = read_run(algorithm, seed)
            runs.append(run)
            base.save_raw_csv(run)
            print(
                f"Loaded {run.label} seed {seed}: "
                f"episodes={run.returns.size}, last_step={run.last_step}, "
                f"run={run.run_dir}"
            )

    grid = np.linspace(
        0,
        base.EXPECTED_LAST_STEP,
        base.GRID_POINTS,
        dtype=np.float64,
    )

    seed_metrics = pd.DataFrame([base.run_metrics(run, grid) for run in runs])
    seed_metrics_path = base.OUTPUT_DIR / "seed_metrics.csv"
    seed_metrics.to_csv(seed_metrics_path, index=False)

    group_summary = base.summarize_groups(seed_metrics)
    group_summary_path = base.OUTPUT_DIR / "group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    paired = paired_seed_comparison(seed_metrics)
    paired_path = base.OUTPUT_DIR / "paired_seed_comparison.csv"
    paired.to_csv(paired_path, index=False)

    tests = paired_statistical_tests(seed_metrics)
    tests_path = base.OUTPUT_DIR / "paired_statistical_tests.csv"
    tests.to_csv(tests_path, index=False)

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
    print("\nPaired tests use only five seeds; interpret p-values cautiously.")
    print("\nCreated:")
    for path in [
        seed_metrics_path,
        group_summary_path,
        paired_path,
        tests_path,
        curve_path,
        final20_path,
        base.RAW_DIR,
    ]:
        print(path)


if __name__ == "__main__":
    main()
