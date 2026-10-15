#!/usr/bin/env python3
"""
Walk-forward backtest: for every week, fit ratings on plays known before kickoff, predict
that week's games, and compare the model, the market and blends of the two.

Used to pick W_PRIOR, ALPHA and MODEL_WEIGHT in model.py, and to check calibration
(when the model says 60%, does the home team win about 60% of the time?).

Run:  python3 -I backtest.py --data-dir data
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from model import Outcomes, Ratings, load_pbp, training_frame

SEASONS = {2025: range(2, 19), 2026: range(2, 7)}   # (season, weeks to predict); week 1 has no in-season data


def walk_forward(pbp, games, w_prior, alpha):
    rows = []
    for season, weeks in SEASONS.items():
        for week in weeks:
            wk = games[(games.season == season) & (games.week == week) & (games.game_type == "REG")
                       & games.result.notna()]
            if wk.empty:
                continue
            r = Ratings(training_frame(pbp, season, week, w_prior), alpha)
            for g in wk.itertuples():
                m, t, _, _ = r.margin_total(g.home_team, g.away_team, g.location == "Neutral")
                rows.append({"season": season, "week": week, "game_id": g.game_id, "model_m": m,
                             "model_t": t, "line": g.spread_line, "total_line": g.total_line,
                             "result": g.result, "total": g.total})
    return pd.DataFrame(rows)


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    data, outd = Path(a.data_dir), Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(data / "games.csv")
    pbp = {s: load_pbp(data / f"pbp_{s}.csv.gz") for s in (2024, 2025, 2026)}
    # Key-number weights learned only from seasons before the backtest, so no leakage.
    out = Outcomes(games[(games.season >= 2015) & (games.season < 2025) & (games.game_type == "REG")])

    # 1) hyperparameters of the ratings themselves: how well does the pure model predict margins?
    print("Model-only margin RMSE by prior weight x ridge alpha (lower is better):")
    runs = {}
    for w_prior, alpha in product([0.2, 0.35, 0.5, 0.75], [100, 250, 500]):
        bt = walk_forward(pbp, games, w_prior, alpha)
        runs[(w_prior, alpha)] = bt
        print(f"  w_prior {w_prior:<4} alpha {alpha:<4} RMSE {np.sqrt(((bt.model_m - bt.result) ** 2).mean()):.3f}")
    best = min(runs, key=lambda k: ((runs[k].model_m - runs[k].result) ** 2).mean())
    bt = runs[best]
    print(f"Best: w_prior {best[0]}, alpha {best[1]} on {len(bt)} games\n")

    # 2) blend weight on the model vs the market
    y = (bt.result > 0).astype(float).values
    nontie = bt.result != 0
    print("Blend weight on model   margin RMSE   win-prob log-loss   total RMSE (same weight)")
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        m = w * bt.model_m + (1 - w) * bt.line
        t = w * bt.model_t + (1 - w) * bt.total_line
        p = np.array([out.win_prob(x) for x in m])
        print(f"  {w:.1f}                   {np.sqrt(((m - bt.result) ** 2).mean()):6.3f}        "
              f"{log_loss(p[nontie], y[nontie]):.4f}            {np.sqrt(((t - bt.total) ** 2).mean()):6.3f}")

    # 3) uncertainty in the blend weight: bootstrap games and re-pick the best weight
    rng = np.random.default_rng(0)
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    picks = []
    for _ in range(500):
        s = bt.sample(len(bt), replace=True, random_state=int(rng.integers(1e9)))
        picks.append(ws[np.argmin([((w * s.model_m + (1 - w) * s.line - s.result) ** 2).mean() for w in ws])])
    lo, med, hi = np.percentile(picks, [10, 50, 90])
    print(f"\nBootstrap best blend weight: median {med:.2f}, 80% interval {lo:.2f}-{hi:.2f}")

    # 4) calibration of the market-only probabilities with the key-number distribution
    bt = bt.assign(p_mkt=[out.win_prob(x) for x in bt.line], home_win=y)
    bt["bin"] = pd.cut(bt.p_mkt, [0, 0.3, 0.45, 0.55, 0.7, 1.0])
    print("\nCalibration (market line -> our win prob vs actual home win rate):")
    print(bt[nontie].groupby("bin", observed=True).agg(games=("home_win", "size"), predicted=("p_mkt", "mean"),
                                                       actual=("home_win", "mean")).round(3).to_string())
    bt.to_csv(outd / "backtest_games.csv", index=False)


if __name__ == "__main__":
    main()
