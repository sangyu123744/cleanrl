from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "analysis"

EXPECTED_SEEDS = {1, 2, 3, 4, 5}
ROLLING_WINDOW = 20
STEP_INTERVAL = 1000

DQN_PATTERN = re.compile(
    r"^CartPole-v1__dqn__(\d+)__(\d+)\.csv$"
)
PER_PATTERN = re.compile(
    r"^CartPole-v1__dqn_per__(\d+)__(\d+)\.csv$"
)


def find_runs(pattern: re.Pattern[str]) -> dict[int, Path]:
    """Find one formal CSV file for each seed."""
    runs: dict[int, Path] = {}

    for path in RESULTS_DIR.glob("*.csv"):
        match = pattern.match(path.name)
        if match is None:
            continue

        seed = int(match.group(1))
        timestamp = int(match.group(2))

        # If duplicate files exist for one seed, keep the newest timestamp.
        if seed not in runs:
            runs[seed] = path
        else:
            old_match = pattern.match(runs[seed].name)
            assert old_match is not None
            old_timestamp = int(old_match.group(2))

            if timestamp > old_timestamp:
                runs[seed] = path

    missing = EXPECTED_SEEDS - set(runs)
    extra = set(runs) - EXPECTED_SEEDS

    if missing:
        raise FileNotFoundError(
            f"Missing CSV files for seeds: {sorted(missing)}"
        )

    if extra:
        print(f"Warning: ignoring extra seeds: {sorted(extra)}")

    return {
        seed: runs[seed]
        for seed in sorted(EXPECTED_SEEDS)
    }


def load_run(path: Path) -> pd.DataFrame:
    """Load and validate a TensorBoard scalar CSV."""
    frame = pd.read_csv(path)

    required_columns = {"Step", "Value"}
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame[["Step", "Value"]].copy()
    frame["Step"] = pd.to_numeric(frame["Step"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")

    frame = (
        frame.dropna()
        .sort_values("Step")
        .drop_duplicates(subset="Step", keep="last")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(f"{path.name} contains no valid data")

    frame["SmoothedValue"] = (
        frame["Value"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    return frame


def summarize_runs(
    algorithm: str,
    runs: dict[int, Path],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, float | int | str]] = []
    loaded: dict[int, pd.DataFrame] = {}

    for seed, path in runs.items():
        frame = load_run(path)
        loaded[seed] = frame

        final_20 = frame["Value"].tail(20)

        rows.append(
            {
                "Algorithm": algorithm,
                "Seed": seed,
                "EpisodesLogged": len(frame),
                "LastStep": int(frame["Step"].iloc[-1]),
                "FinalObservedReturn": float(
                    frame["Value"].iloc[-1]
                ),
                "Final20Mean": float(final_20.mean()),
                "Final20Std": float(
                    final_20.std(ddof=1)
                    if len(final_20) > 1
                    else 0.0
                ),
                "MaximumReturn": float(frame["Value"].max()),
                "CSVFile": path.name,
            }
        )

    return pd.DataFrame(rows), loaded


def interpolate_runs(
    runs: dict[int, pd.DataFrame],
    common_steps: np.ndarray,
) -> np.ndarray:
    interpolated: list[np.ndarray] = []

    for seed in sorted(runs):
        frame = runs[seed]

        values = np.interp(
            common_steps,
            frame["Step"].to_numpy(dtype=float),
            frame["SmoothedValue"].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )

        interpolated.append(values)

    return np.vstack(interpolated)


def calculate_group_summary(
    per_seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for algorithm, group in per_seed_summary.groupby(
        "Algorithm",
        sort=False,
    ):
        final_means = group["Final20Mean"].to_numpy(dtype=float)

        rows.append(
            {
                "Algorithm": algorithm,
                "NumberOfSeeds": len(group),
                "MeanFinal20Return": float(final_means.mean()),
                "StdFinal20Return": float(
                    final_means.std(ddof=1)
                ),
                "MinFinal20Return": float(final_means.min()),
                "MaxFinal20Return": float(final_means.max()),
                "MeanMaximumReturn": float(
                    group["MaximumReturn"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_mean_curves(
    common_steps: np.ndarray,
    dqn_values: np.ndarray,
    per_values: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))

    for label, values in (
        ("DQN", dqn_values),
        ("DQN+PER", per_values),
    ):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)

        line = axis.plot(
            common_steps,
            mean,
            label=label,
            linewidth=2,
        )[0]

        axis.fill_between(
            common_steps,
            mean - std,
            mean + std,
            alpha=0.2,
            color=line.get_color(),
        )

    axis.set_title(
        "CartPole-v1: DQN vs DQN+PER (5 Seeds)"
    )
    axis.set_xlabel("Environment Steps")
    axis.set_ylabel(
        f"Episodic Return ({ROLLING_WINDOW}-Episode Rolling Mean)"
    )
    axis.set_xlim(0, common_steps[-1])
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "dqn_vs_dqn_per_mean_std.png",
        dpi=200,
    )
    plt.close(figure)


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {RESULTS_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dqn_paths = find_runs(DQN_PATTERN)
    per_paths = find_runs(PER_PATTERN)

    dqn_summary, dqn_runs = summarize_runs(
        "DQN",
        dqn_paths,
    )
    per_summary, per_runs = summarize_runs(
        "DQN+PER",
        per_paths,
    )

    per_seed_summary = pd.concat(
        [dqn_summary, per_summary],
        ignore_index=True,
    )

    group_summary = calculate_group_summary(
        per_seed_summary
    )

    maximum_common_step = min(
        int(frame["Step"].iloc[-1])
        for frame in [*dqn_runs.values(), *per_runs.values()]
    )

    common_steps = np.arange(
        0,
        maximum_common_step + 1,
        STEP_INTERVAL,
        dtype=float,
    )

    dqn_interpolated = interpolate_runs(
        dqn_runs,
        common_steps,
    )
    per_interpolated = interpolate_runs(
        per_runs,
        common_steps,
    )

    per_seed_summary.to_csv(
        OUTPUT_DIR / "per_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary.to_csv(
        OUTPUT_DIR / "group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_mean_curves(
        common_steps,
        dqn_interpolated,
        per_interpolated,
    )

    print("\nPer-seed summary:")
    print(
        per_seed_summary[
            [
                "Algorithm",
                "Seed",
                "FinalObservedReturn",
                "Final20Mean",
                "MaximumReturn",
            ]
        ].to_string(index=False)
    )

    print("\nFive-seed aggregate summary:")
    print(group_summary.to_string(index=False))

    print("\nGenerated files:")
    print(OUTPUT_DIR / "per_seed_summary.csv")
    print(OUTPUT_DIR / "group_summary.csv")
    print(OUTPUT_DIR / "dqn_vs_dqn_per_mean_std.png")


if __name__ == "__main__":
    main()