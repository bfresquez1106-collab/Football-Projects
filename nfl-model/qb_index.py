#!/usr/bin/env python3
"""
Build and test the QB index.

  1. Stability: split every QB-season into odd and even dropbacks. If a stat is skill, the two
     halves agree. From that agreement, k = how many dropbacks before the stat is half signal.
  2. Replacement level: what QBs with little playing time produce. Unknown QBs start here.
  3. Standardize, then regress: use each QB's stats BEFORE a game to predict his EPA per
     dropback IN that game. The regression decides how much each stat matters.
  4. Test on 2024-26 games the fit never saw, against simpler alternatives.

Writes qb_params.json (used by the model) and outputs/qb_ratings.csv.
Run:  python3 qb_index.py --data-dir data      (~2 minutes)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import qb

SEASONS = range(2016, 2027)
FIT, VALID, TEST = range(2019, 2023), [2023], range(2024, 2027)   # choose settings, then test once


def stability(db):
    """Split-half reliability per stat and the k it implies."""
    db = db.sort_values(["passer_id", "season", "game_id"]).copy()
    db["half"] = db.groupby(["passer_id", "season"]).cumcount() % 2
    s = qb.weekly_sums(db.assign(week=db.half))           # 'week' here = half 0 / half 1
    rows = []
    for (pid, season), g in s.groupby(["passer_id", "season"]):
        if g.n.sum() >= 250 and len(g) == 2:
            h = {}
            for _, r in g.iterrows():
                h[r.week] = {"epa": r.epa / r.n, "success": r.success / r.n, "sack_rate": r.sacks / r.n,
                             "int_rate": r.ints / max(r.att, 1), "cpoe": r.cp / max(r.cp_n, 1),
                             "n": r.n, "n_att": r.att, "n_cp": r.cp_n}
            rows.append(h)
    out = {}
    for st in qb.STATS:
        a = np.array([r[0][st] for r in rows]); b = np.array([r[1][st] for r in rows])
        unit = "n_att" if st == "int_rate" else "n_cp" if st == "cpoe" else "n"
        n_half = np.mean([(r[0][unit] + r[1][unit]) / 2 for r in rows])
        rel = float(np.corrcoef(a, b)[0, 1])
        out[st] = {"split_half_r": rel, "n_half": float(n_half),
                   "k": float(n_half * (1 - rel) / rel) if rel > 0.02 else 1e5}
    return out, len(rows)


def replacement_level(sums):
    """Per-dropback production of QB-seasons with under 200 dropbacks (backups, spot starters)."""
    qs = sums.groupby(["passer_id", "season"])[["n", "epa", "success", "sacks", "ints", "att", "cp", "cp_n"]].sum()
    b = qs[qs.n < 200].sum()
    return {"epa": b.epa / b.n, "success": b.success / b.n, "sack_rate": b.sacks / b.n,
            "int_rate": b.ints / b.att, "cpoe": b.cp / b.cp_n}


def snapshots(sums, db_games, p, seasons):
    """For every game: each QB's shrunk z-scores from BEFORE that week, and what he did IN it."""
    rows = []
    db_games = db_games[["passer_id", "season", "week", "name", "team", "n", "target"]]
    for season in seasons:
        for week in sorted(db_games[db_games.season == season].week.unique()):
            raw = qb.raw_as_of(sums, season, week, p["decay"])
            z = qb.zscores(qb.shrink(raw, p), p)
            g = db_games[(db_games.season == season) & (db_games.week == week)]
            g = g[g.passer_id.isin(z.index)]
            if len(g):
                rows.append(g.join(z, on="passer_id").join(raw[["n"]].rename(columns={"n": "prior_n"}), on="passer_id")
                            .join(raw[["epa"]].rename(columns={"epa": "raw_epa"}), on="passer_id"))
            # QBs with no history at all start exactly at replacement level (z of replacement)
            new = db_games[(db_games.season == season) & (db_games.week == week) & ~db_games.passer_id.isin(z.index)]
            if len(new):
                zr = {st: (p["replacement"][st] - p["mean"][st]) / p["sd"][st] for st in qb.STATS}
                rows.append(new.assign(**zr, prior_n=0.0, raw_epa=p["replacement"]["epa"]))
    return pd.concat(rows, ignore_index=True)


def wls(X, y, w):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.solve((X1.T * w) @ X1, (X1.T * w) @ y)


def wmse(pred, y, w):
    return float(np.sum(w * (y - pred) ** 2) / np.sum(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    data = Path(a.data_dir)

    db = pd.concat([qb.load_dropbacks(data / f"pbp_{s}.csv.gz") for s in SEASONS if (data / f"pbp_{s}.csv.gz").exists()])
    sums = qb.weekly_sums(db)
    games = sums[sums.n >= 10].copy()                     # QB-games with real playing time
    games["target"] = games.epa / games.n
    print(f"Data: {len(db):,} dropbacks, {db.passer_id.nunique()} QBs, seasons {db.season.min()}-{db.season.max()}\n")

    # 1) stability
    stab, n_qs = stability(db[db.season < 2024])
    print(f"Stability ({n_qs} QB-seasons with 250+ dropbacks, odd vs even dropbacks):")
    print(f"  {'stat':<10}{'split-half r':>13}{'per half':>10}{'k (half signal at)':>21}")
    for st, v in stab.items():
        unit = "attempts" if st in ("int_rate", "cpoe") else "dropbacks"
        print(f"  {st:<10}{v['split_half_r']:>13.2f}{v['n_half']:>10.0f}{v['k']:>13.0f} {unit}")

    # 2) replacement level and scale
    p = {"k": {st: v["k"] for st, v in stab.items()}, "replacement": replacement_level(sums[sums.season < 2024]),
         "decay": qb.DECAY}
    full = qb.shrink(pd.concat([qb.raw_as_of(sums, s, 99, 0.0001) for s in range(2017, 2024)]), p)
    full = full[full.n >= 200]
    p["mean"] = {st: float(full[st].mean()) for st in qb.STATS}
    p["sd"] = {st: float(full[st].std()) for st in qb.STATS}
    print("\nReplacement level vs a typical starter:")
    for st in qb.STATS:
        print(f"  {st:<10} replacement {p['replacement'][st]:+.3f}   starters {p['mean'][st]:+.3f} (SD {p['sd'][st]:.3f})")

    # 3) choose recency decay on a validation season, then fit weights
    print("\nRecency decay (fit 2019-22, check 2023):")
    best = None
    for d in (0.4, 0.6, 0.8):
        p["decay"] = d
        sn = snapshots(sums, games, p, list(FIT) + VALID)
        fit, val = sn[sn.season.isin(FIT)], sn[sn.season.isin(VALID)]
        beta = wls(fit[qb.STATS].values, fit.target.values, fit.n.values)
        err = wmse(np.c_[np.ones(len(val)), val[qb.STATS].values] @ beta, val.target.values, val.n.values)
        print(f"  decay {d}: validation error {err:.5f}")
        if best is None or err < best[1]:
            best = (d, err)
    p["decay"] = best[0]

    sn = snapshots(sums, games, p, list(FIT) + VALID + list(TEST))
    train, test = sn[sn.season.isin(list(FIT) + VALID)], sn[sn.season.isin(TEST)]
    beta = wls(train[qb.STATS].values, train.target.values, train.n.values)

    # 4) test against simpler versions
    def evaluate(cols):
        b = wls(train[cols].values, train.target.values, train.n.values)
        return wmse(np.c_[np.ones(len(test)), test[cols].values] @ b, test.target.values, test.n.values)
    raw_b = wls(train[["raw_epa"]].values, train.target.values, train.n.values)
    base = wmse(np.full(len(test), np.average(train.target, weights=train.n)), test.target.values, test.n.values)
    results = {"Everyone is average": base,
               "Raw EPA/dropback (no shrinking)": wmse(np.c_[np.ones(len(test)), test[["raw_epa"]].values] @ raw_b,
                                                       test.target.values, test.n.values),
               "Shrunk EPA/dropback only": evaluate(["epa"]),
               "EPA + success rate": evaluate(qb.COMPOSITE),
               "All five stats": evaluate(qb.STATS)}
    print(f"\nTest on {len(test):,} QB-games, 2024-26 (error predicting EPA/dropback; lower is better):")
    for k, v in results.items():
        print(f"  {k:<34}{v:.5f}   ({(1 - v / base) * 100:+.1f}% vs average)")

    # final fit on everything, weights in EPA per dropback per 1 SD
    beta = wls(sn[qb.COMPOSITE].values, sn.target.values, sn.n.values)
    p["intercept"], p["weights"] = float(beta[0]), dict(zip(qb.COMPOSITE, map(float, beta[1:])))
    print("\nComposite weights (EPA per dropback for +1 SD in each stat, the other held fixed):")
    for st, w in p["weights"].items():
        print(f"  {st:<10}{w:+.4f}")
    starters = sn[sn.n >= 20]
    pred = p["intercept"] + sum(w * starters[st] for st, w in p["weights"].items())
    p["starter_mean"], p["starter_sd"] = float(pred.mean()), float(pred.std())
    p["stability"], p["test"] = stab, results
    json.dump(p, open("qb_params.json", "w"), indent=1)

    # current ratings
    season, week = int(sums.season.max()), int(sums[sums.season == sums.season.max()].week.max()) + 1
    now = qb.ratings_as_of(sums, season, week, p)
    now = now[now.n >= 150].sort_values("rating", ascending=False)
    now.round(4).to_csv("outputs/qb_ratings.csv")
    print(f"\nQB index before {season} week {week} (100 = average starter, 10 = 1 SD), top 12 and Denver/Seattle:")
    show = pd.concat([now.head(12), now[now.team.isin(["DEN", "SEA"])]]).drop_duplicates()
    print(show[["name", "team", "n", "rating", "qb_index", "epa_index", "success_index", "cpoe_index",
                "sack_rate_index", "int_rate_index"]].round(3).to_string())


if __name__ == "__main__":
    main()
