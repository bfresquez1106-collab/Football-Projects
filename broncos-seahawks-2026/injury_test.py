#!/usr/bin/env python3
"""
Do missing starters actually move EPA? Tests the judgement-call injury adjustments in model.py.

For every game (2024-2026, week 3+):
  1. Starters from season-to-date snap counts (weeks before this one, >= 2 games played):
       DB  = top 5 CB/S by defensive snap share, each >= 50%
       LB  = top 2 LB by defensive snap share, each >= 50%
       RB1 = top RB by offensive snap share, >= 35%
  2. "Missing" = a starter with no snaps in this game (catches IR too, unlike the injury report).
  3. Residual = actual EPA/play in this game minus what walk-forward ratings predicted.
  4. Play-weighted regression of residual on number of starters missing.

Run:  python3 injury_test.py --data-dir data
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import Config, Ratings, load_pbp, training_frame

SEASONS = {2024: range(3, 19), 2025: range(3, 19), 2026: range(3, 6)}
GROUPS = {"DB": (["CB", "S", "FS", "SS", "DB"], "defense_pct", 5, 0.50),
          "LB": (["LB", "ILB", "MLB", "OLB"], "defense_pct", 2, 0.50),
          "RB": (["RB"], "offense_pct", 1, 0.35)}


def norm_name(s):
    s = re.sub(r"[^a-z ]", "", str(s).lower())
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def starters(snaps, season, week, team, group):
    pos, col, k, floor = GROUPS[group]
    d = snaps[(snaps.season == season) & (snaps.week < week) & (snaps.team == team) & snaps.position.isin(pos)]
    avg = d.groupby("player")[col].agg(["mean", "size"])
    avg = avg[(avg["size"] >= 2) & (avg["mean"] >= floor)].sort_values("mean", ascending=False)
    return list(avg.index[:k])


def missing(snaps, game_id, team, names, col):
    played = snaps[(snaps.game_id == game_id) & (snaps.team == team) & (snaps[col] > 0)].player
    return sum(n not in set(played) for n in names)


def wls(x, y, w):
    X = np.c_[np.ones(len(x)), x]
    W = np.asarray(w, float)
    beta = np.linalg.solve((X.T * W) @ X, (X.T * W) @ y)
    res = y - X @ beta
    # cluster-free sandwich (HC0) standard errors; games are close to independent
    bread = np.linalg.inv((X.T * W) @ X)
    meat = (X.T * (W * res) ** 2) @ X
    se = np.sqrt(np.diag(bread @ meat @ bread))
    return beta, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    data, outd = Path(a.data_dir), Path(a.out_dir)
    cfg = Config()

    pbp = {s: load_pbp(data / f"pbp_{s}.csv.gz") for s in (2023, 2024, 2025, 2026)}
    snaps = pd.concat([pd.read_csv(data / f"snap_counts_{s}.csv") for s in (2024, 2025, 2026)])
    snaps = snaps[snaps.game_type == "REG"]

    rows = []
    for season, weeks in SEASONS.items():
        for week in weeks:
            r = Ratings(training_frame(pbp, season, week, cfg.w_prior), cfg.alpha)
            wk = Ratings(pbp[season][pbp[season].week == week].assign(w=1.0), cfg.alpha).clean
            for (gid, off, dfn), d in wk.groupby(["game_id", "posteam", "defteam"]):
                if off not in r.tix or dfn not in r.tix:
                    continue
                h = d.h.iloc[0]
                ep, er = r.epa_pair(off, dfn, h)
                ps, rs = d[d["pass"] == 1], d[d["rush"] == 1]
                rows.append({
                    "season": season, "week": week, "game_id": gid, "off": off, "def": dfn,
                    "pass_res": ps.epa.mean() - ep, "n_pass": len(ps),
                    "rush_res": rs.epa.mean() - er, "n_rush": len(rs),
                    "def_db_out": missing(snaps, gid, dfn, starters(snaps, season, week, dfn, "DB"), "defense_pct"),
                    "def_lb_out": missing(snaps, gid, dfn, starters(snaps, season, week, dfn, "LB"), "defense_pct"),
                    "off_rb_out": missing(snaps, gid, off, starters(snaps, season, week, off, "RB"), "offense_pct"),
                })
    df = pd.DataFrame(rows).dropna(subset=["pass_res", "rush_res"])
    df.to_csv(outd / "injury_test_games.csv", index=False)
    print(f"{len(df)} team-games\n")

    tests = [("Starting DBs missing -> opponent pass EPA", "def_db_out", "pass_res", "n_pass"),
             ("Starting DBs missing -> opponent rush EPA", "def_db_out", "rush_res", "n_rush"),
             ("Starting LBs missing -> opponent rush EPA", "def_lb_out", "rush_res", "n_rush"),
             ("Starting LBs missing -> opponent pass EPA", "def_lb_out", "pass_res", "n_pass"),
             ("RB1 missing -> own rush EPA", "off_rb_out", "rush_res", "n_rush")]
    print(f"{'test':<44}{'EPA/play per starter':>22}{'95% CI':>20}   games by count")
    for name, x, y, w in tests:
        beta, se = wls(df[x].values, df[y].values, df[w].values)
        counts = df[x].clip(upper=3).value_counts().sort_index().to_dict()
        print(f"{name:<44}{beta[1]:>+22.3f}   [{beta[1] - 1.96 * se[1]:+.3f}, {beta[1] + 1.96 * se[1]:+.3f}]   {counts}")

    print("\nMean pass-EPA residual by starting DBs missing (positive = defense worse than rated):")
    print(df.assign(k=df.def_db_out.clip(upper=3)).groupby("k").apply(
        lambda g: pd.Series({"games": len(g), "pass_res": np.average(g.pass_res, weights=g.n_pass)}),
        include_groups=False).round(3).to_string())

    for team in ("DEN", "SEA"):
        print(f"\n{team} 2026 starters (season to date before week 6):")
        for grp in GROUPS:
            print(f"  {grp}: {starters(snaps, 2026, 6, team, grp)}")


if __name__ == "__main__":
    main()
