#!/usr/bin/env python3
"""
QB lab: four questions about the QB index, each with one simple test.

  A. Scheme vs QB, and opponent strength. Each dropback's value is split into
       QB + team context (scheme, line, receivers) + defense faced
     with a ridge regression, the same idea as the team ratings. Switching parts on and off
     gives the variants. Only natural experiments can separate a QB from his team: QBs who
     change teams, and backups who play in the same scheme.
     Test (the transfer test): when a QB changes teams, which rating best predicts how he
     plays for the new team?
  B. Playmaking: explosive plays, scramble value, 3rd/4th-down value, deep passing. Is each one
     stable (skill), and does it predict next season beyond EPA + success rate?
  C. Pressure (PFR, 2018+): pressure faced and sacks per pressure, tested the same way.

Run:  python3 qb_lab.py --data-dir data
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = range(2016, 2027)
COLS = ["season", "season_type", "week", "game_id", "posteam", "defteam", "passer_id", "passer", "qb_dropback",
        "qb_epa", "success", "wp", "qb_spike", "qb_scramble", "yards_gained", "down", "air_yards", "complete_pass",
        "sack", "pass_attempt"]
DECAY = 0.6
OFF = 1e9                      # a penalty this large switches a part of the model off


def load(data):
    parts = []
    for s in SEASONS:
        d = pd.read_csv(data / f"pbp_{s}.csv.gz", usecols=lambda c: c in COLS, low_memory=False)
        parts.append(d[(d.season_type == "REG") & d.wp.between(0.05, 0.95) & (d.qb_dropback == 1)
                       & d.passer_id.notna() & (d.qb_spike != 1)])
    d = pd.concat(parts, ignore_index=True)
    d = d.fillna({"qb_epa": 0, "success": 0, "qb_scramble": 0, "yards_gained": 0, "complete_pass": 0, "sack": 0})
    d["ts"] = d.posteam + "_" + d.season.astype(str)          # team-season: scheme, line, receivers
    d["ds"] = d.defteam + "_" + d.season.astype(str)          # defense-season faced
    return d


# ---------------------------------------------------------------------------------------- A
def decompose(d, w, a_qb, a_ctx, a_def, y="qb_epa"):
    """Ridge fit of value = mean + QB + team context + defense, on (QB, team, defense) cells."""
    c = d.assign(w=w, wy=w * d[y]).groupby(["passer_id", "ts", "ds"]).agg(w=("w", "sum"), wy=("wy", "sum")).reset_index()
    c = c[c.w > 0]
    q, t, f = (pd.Categorical(c[k]) for k in ("passer_id", "ts", "ds"))
    nq, nt, nf = len(q.categories), len(t.categories), len(f.categories)
    P = nq + nt + nf + 1
    idx = [q.codes.astype(np.int64), nq + t.codes.astype(np.int64), nq + nt + f.codes.astype(np.int64),
           np.full(len(c), P - 1)]
    A, b = np.zeros((P, P)), np.zeros(P)
    for i in idx:
        np.add.at(b, i, c.wy.values)
        for j in idx:
            np.add.at(A, (i, j), c.w.values)
    pen = np.r_[np.full(nq, a_qb), np.full(nt, a_ctx), np.full(nf, a_def), 0.0]
    beta = np.linalg.solve(A + np.diag(pen), b)
    return {"mu": beta[-1], "qb": pd.Series(beta[:nq], q.categories),
            "ctx": pd.Series(beta[nq:nq + nt], t.categories), "def": pd.Series(beta[nq + nt:-1], f.categories)}


VARIANTS = {   # name: (QB shrink, context penalty, defense penalty)
    "raw (current)": (250, OFF, OFF),
    "+ opponent": (250, OFF, 600),
    "+ scheme only (very strong)": (250, 100, OFF),
    "+ opponent + scheme (very strong)": (250, 100, 600),
    "+ opponent + scheme (extreme)": (250, 30, 600),
    "+ opponent + scheme (strong)": (250, 300, 600),
    "+ opponent + scheme (medium)": (250, 1000, 600),
    "+ opponent + scheme (light)": (250, 3000, 600),
}


def transfer_test(d):
    """Predict each QB's EPA/dropback in season t from seasons t-3..t-1, split by moved vs stayed."""
    rows = []
    for t in range(2019, 2027):
        tr = d[d.season.between(t - 3, t - 1)]
        w = DECAY ** (t - 1 - tr.season)
        fits = {k: decompose(tr, w, *v) for k, v in VARIANTS.items()}
        cur = d[d.season == t].groupby(["passer_id", "posteam"]).agg(n=("qb_epa", "size"), y=("qb_epa", "mean")).reset_index()
        cur = cur[cur.n >= (100 if t == 2026 else 150)]
        prior = tr.groupby(["passer_id", "posteam"]).size().unstack(fill_value=0)
        for r in cur.itertuples():
            if r.passer_id not in prior.index or prior.loc[r.passer_id].sum() < 200:
                continue
            share = prior.loc[r.passer_id].get(r.posteam, 0) / prior.loc[r.passer_id].sum()
            kind = "moved" if share < 0.5 else "stayed" if share >= 0.8 else None
            if kind is None:
                continue
            row = {"season": t, "qb": r.passer_id, "team": r.posteam, "kind": kind, "n": r.n, "y": r.y}
            for k, f in fits.items():
                row[k] = f["mu"] + f["qb"].get(r.passer_id, 0.0)
                row[k + " | new team"] = row[k] + f["ctx"].get(f"{r.posteam}_{t - 1}", 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - mx) * (y - my)) / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))


# ---------------------------------------------------------------------------------------- B, C
def season_table(d, pressure):
    d = d.assign(expl=((d.complete_pass == 1) & (d.yards_gained >= 20)) | ((d.qb_scramble == 1) & (d.yards_gained >= 15)),
                 scr=np.where(d.qb_scramble == 1, d.qb_epa, 0.0), late=d.down.isin([3, 4]),
                 deep=np.where(d.air_yards >= 20, d.qb_epa, 0.0))
    d["late_epa"] = np.where(d.late, d.qb_epa, 0.0)
    g = d.groupby(["passer_id", "season"])
    s = pd.DataFrame({"name": g.passer.last(), "team": g.posteam.last(), "n": g.size(), "epa": g.qb_epa.mean(),
                      "success": g.success.mean(), "explosive": g.expl.mean(), "scramble": g.scr.mean(),
                      "late_down": g.late_epa.sum() / g.late.sum().clip(lower=1), "deep": g.deep.mean(),
                      "sacks": g.sack.sum()})
    # pressure: PFR team-game pressures, credited to that game's main QB
    main = d.groupby(["game_id", "posteam", "passer_id"]).size().reset_index(name="db")
    main = main.sort_values("db").groupby(["game_id", "posteam"]).tail(1)
    pm = main.merge(pressure, left_on=["game_id", "posteam"], right_on=["game_id", "team"])
    pm["season"] = pm.game_id.str[:4].astype(int)
    pg = pm.groupby(["passer_id", "season"]).agg(pressured=("pressured", "sum"), db=("db", "sum"), psacks=("sacked", "sum"))
    s = s.join(pg)
    s["pressure_rate"] = s.pressured / s.db
    s["sack_per_pressure"] = s.psacks / s.pressured
    return s


def split_half(d, metric_fn):
    d = d.sort_values(["passer_id", "season", "game_id"]).copy()
    d["half"] = d.groupby(["passer_id", "season"]).cumcount() % 2
    h = d.groupby(["passer_id", "season", "half"]).apply(metric_fn, include_groups=False).unstack("half")
    n = d.groupby(["passer_id", "season"]).size()
    h = h[n.reindex(h.index) >= 250].dropna()
    return float(np.corrcoef(h[0], h[1])[0, 1])


def next_season_cv(s, extra):
    """Leave-one-season-out: predict next season's EPA/dropback from this season's stats."""
    a = s[s.n >= 250].reset_index()
    b = a[["passer_id", "season", "epa", "n"]].assign(season=lambda x: x.season - 1).rename(columns={"epa": "y", "n": "n_next"})
    p = a.merge(b, on=["passer_id", "season"]).dropna(subset=["epa", "success"] + extra)
    errs = {"base": 0.0, "plus": 0.0}
    for t in p.season.unique():
        tr, te = p[p.season != t], p[p.season == t]
        for k, cols in (("base", ["epa", "success"]), ("plus", ["epa", "success"] + extra)):
            X = np.c_[np.ones(len(tr)), tr[cols]]
            beta = np.linalg.lstsq(X * np.sqrt(tr.n.values[:, None]), tr.y * np.sqrt(tr.n), rcond=None)[0]
            errs[k] += float(np.sum(te.n * (te.y - np.c_[np.ones(len(te)), te[cols]] @ beta) ** 2))
    return 100 * (1 - errs["plus"] / errs["base"]), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    data = Path(a.data_dir)
    d = load(data)
    print(f"{len(d):,} dropbacks, {d.passer_id.nunique()} QBs, {d.season.min()}-{d.season.max()}\n")

    # A. transfer test
    ev = transfer_test(d)
    print("A. TRANSFER TEST: predicting a QB's EPA/dropback next season (correlation, higher is better)")
    for kind in ("moved", "stayed"):
        e = ev[ev.kind == kind]
        print(f"  QBs who {kind} ({len(e)} QB-seasons):")
        for k in VARIANTS:
            print(f"    {k:<32} QB only {wcorr(e[k], e.y, e.n):.3f}   QB + new team's context {wcorr(e[k + ' | new team'], e.y, e.n):.3f}")
    ev.to_csv("outputs/qb_transfer_test.csv", index=False)

    # Mac Jones case study, using everything up to now
    now = d[d.season >= 2023]
    f = decompose(now, DECAY ** (2026 - now.season), 250, 1000, OFF)      # the adopted setting: scheme, medium
    raw = decompose(now, DECAY ** (2026 - now.season), *VARIANTS["raw (current)"])
    names = d.groupby("passer_id").passer.agg(lambda x: x.mode().iloc[0])
    pick = ["M.Jones", "B.Purdy", "P.Mahomes", "J.Allen", "L.Jackson", "J.Burrow", "J.Goff", "J.Herbert",
            "T.Tagovailoa", "S.Darnold", "B.Nix", "D.Maye"]
    ids = [i for i in names.index if names[i] in pick and i in f["qb"].index and d[d.passer_id == i].season.max() >= 2025]
    print("\nCase study (2023-26, recent seasons weigh more; scheme medium, no opponent): QB value per dropback")
    print(f"  {'QB':<14}{'raw':>8}{'QB only':>9}   teams (context per dropback)")
    for i in sorted(ids, key=lambda i: -f["qb"][i]):
        teams = d[(d.passer_id == i) & (d.season >= 2023)].groupby("ts").size()
        ctx = ", ".join(f"{t} {f['ctx'].get(t, 0):+.3f} ({n})" for t, n in teams.items())
        print(f"  {names[i]:<14}{raw['qb'][i]:>+8.3f}{f['qb'][i]:>+9.3f}   {ctx}")

    # B and C: season-level metrics
    pr = pd.concat([pd.read_csv(data / f"advstats_week_pass_{y}.csv") for y in range(2018, 2027)
                    if (data / f"advstats_week_pass_{y}.csv").exists()])
    pr = pr.groupby(["game_id", "team"]).agg(pressured=("times_pressured", "sum"), sacked=("times_sacked", "sum")).reset_index()
    s = season_table(d, pr)
    dd = d.assign(expl=((d.complete_pass == 1) & (d.yards_gained >= 20)) | ((d.qb_scramble == 1) & (d.yards_gained >= 15)))
    fns = {"epa": lambda g: g.qb_epa.mean(), "explosive": lambda g: g.expl.mean(),
           "scramble": lambda g: (g.qb_epa * g.qb_scramble).mean(),
           "late_down": lambda g: g.qb_epa[g.down.isin([3, 4])].mean(),
           "deep": lambda g: (g.qb_epa * (g.air_yards >= 20)).mean()}
    print("\nB/C. CANDIDATE STATS: stable? adds to next-season prediction beyond EPA + success?")
    print(f"  {'stat':<18}{'split-half r':>13}{'next-season gain':>18}")
    for m in ["explosive", "scramble", "late_down", "deep", "pressure_rate", "sack_per_pressure"]:
        r = split_half(dd[dd.season < 2026], fns[m]) if m in fns else float("nan")
        gain, n = next_season_cv(s[s.index.get_level_values("season") >= (2018 if "press" in m else 2016)], [m])
        print(f"  {m:<18}{r:>13.2f}{gain:>+17.2f}%   ({n} QB-season pairs)")
    print("  (gain = % less error predicting next season's EPA/dropback; split-half for EPA itself: "
          f"{split_half(dd[dd.season < 2026], fns['epa']):.2f})")

    elite = s[(s.index.get_level_values("season") >= 2023) & (s.index.get_level_values("season") <= 2025)]
    mcols = ["epa", "explosive", "scramble", "late_down", "deep", "pressure_rate"]
    elite = elite.groupby(level="passer_id").apply(
        lambda x: pd.Series(list(np.average(x[mcols].fillna(0), axis=0, weights=x.n)) + [x.n.sum()], index=mcols + ["n"]))
    elite = elite[elite.n >= 800]
    rk = elite.rank(ascending=False).astype(int)
    rk["pressure_rate"] = elite.pressure_rate.rank().astype(int)
    rk.index = [names.get(i, i) for i in rk.index]
    print(f"\nRanks among {len(elite)} QBs with 800+ dropbacks, 2023-25 (1 = best; pressure_rate 1 = least pressured):")
    print(rk.loc[[n for n in ["P.Mahomes", "J.Allen", "L.Jackson", "J.Burrow", "B.Purdy", "J.Goff", "J.Herbert", "B.Nix"]
                  if n in rk.index], ["epa", "explosive", "scramble", "late_down", "deep", "pressure_rate"]].to_string())


if __name__ == "__main__":
    main()
