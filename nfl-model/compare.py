#!/usr/bin/env python3
"""
Judge experiments against the current model and the Elo benchmark, on the same replay games.

  python3 compare.py wprior_0.35 wprior_0.5          # names of experiments/<tag>_games.csv

For each experiment: Brier score overall and by part of the season, and a paired bootstrap of
"experiment minus current model" (negative = experiment is better). A change is worth keeping
only if the whole 95% interval sits below zero.
"""
import sys

import numpy as np
import pandas as pd

from nflmodel import data as D, elo

PARTS = {"Weeks 1-4": (1, 4), "Weeks 5-9": (5, 9), "Weeks 10+": (10, 99)}


def brier(p, result):
    y = np.where(result > 0, 1.0, np.where(result < 0, 0.0, 0.5))
    return (np.asarray(p) - y) ** 2


def load(path, name):
    d = pd.read_csv(path)[["game_id", "season", "week", "result", "cv_wp"]]
    return d.rename(columns={"cv_wp": name})


def main():
    base = load("outputs/replay_games.csv", "current")
    hist = {}
    elo.ratings_through(D.games("data"), int(base.season.max()) + 1, 1, history=hist)
    base["elo"] = base.game_id.map(hist)
    names = sys.argv[1:]
    for n in names:
        base = base.merge(load(f"experiments/{n}_games.csv", n)[["game_id", n]], on="game_id")

    cols = ["current", "elo"] + names
    print(f"{len(base)} games. Brier score (lower is better):")
    print(f"  {'model':<22}{'all':>7}" + "".join(f"{p:>11}" for p in PARTS))
    for c in cols:
        b = brier(base[c], base.result)
        parts = [b[base.week.between(lo, hi)].mean() for lo, hi in PARTS.values()]
        print(f"  {c:<22}{b.mean():>7.4f}" + "".join(f"{x:>11.4f}" for x in parts))

    rng = np.random.default_rng(0)
    cur = brier(base.current, base.result)
    print("\nExperiment minus current (negative = better), 95% bootstrap interval:")
    for n in names:
        diff = brier(base[n], base.result) - cur
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "KEEP" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
        print(f"  {n:<22}{diff.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]   {verdict}")


if __name__ == "__main__":
    main()
