#!/usr/bin/env python3
"""Prospective Monte Carlo power calculation for the frozen Tahoe panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate_power(
    rng: np.random.Generator,
    compounds: int,
    doses: int,
    effect: float,
    sigma_compound_slope: float,
    sigma_dose_slope: float,
    simulations: int,
    alpha: float,
) -> float:
    rejections = 0
    for _ in range(simulations):
        compound_slope = rng.normal(effect, sigma_compound_slope, compounds)
        dose_noise = rng.normal(0.0, sigma_dose_slope, (compounds, doses))
        estimates = compound_slope + dose_noise.mean(axis=1)
        statistic = estimates.mean() / (estimates.std(ddof=1) / np.sqrt(compounds))
        critical = t.ppf(1.0 - alpha, compounds - 1)
        rejections += statistic > critical
    return rejections / simulations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--doses", type=int, default=2)
    parser.add_argument("--effect", type=float, default=0.15)
    parser.add_argument("--sigma-compound-slope", type=float, default=0.35)
    parser.add_argument("--sigma-dose-slope", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    grid = []
    for compounds in [100, 120, 150, 180, 200, 220, 250]:
        grid.append(
            {
                "compound_clusters": compounds,
                "power": estimate_power(
                    rng,
                    compounds,
                    args.doses,
                    args.effect,
                    args.sigma_compound_slope,
                    args.sigma_dose_slope,
                    args.simulations,
                    args.alpha,
                ),
            }
        )
    passing = [row["compound_clusters"] for row in grid if row["power"] >= 0.80]
    result = {
        "seed": args.seed,
        "simulations_per_grid_point": args.simulations,
        "one_sided_alpha": args.alpha,
        "design_effect_sd_per_distance_sd": args.effect,
        "compound_random_slope_sd": args.sigma_compound_slope,
        "dose_random_slope_sd": args.sigma_dose_slope,
        "minimum_doses_per_compound": args.doses,
        "minimum_compound_clusters_for_80pct_power": min(passing) if passing else None,
        "grid": grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
