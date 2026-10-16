#!/usr/bin/env python3
"""
Deep dive on the "Defense takes over" scenario: DEN 2+ takeaways, 0 giveaways, 4+ sacks made,
at most 1 sack taken.

  1. What those games look like on tonight's line: score and margin distribution.
  2. Which half does the work: the turnover half alone vs the sack half alone.
  3. How likely it is in THIS matchup, not the league: Poisson rates for each lever from the two
     teams' 2026 form (2025 at weight 0.2), corrected by how far the independence assumption
     misses the real league frequency.
  4. Every Broncos game since 2019 where the defense did it.

Run:  python3 defense_takes_over.py --data-dir data
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import scenarios as sc

HOME, AWAY, GAME = "DEN", "SEA", "2026_06_SEA_DEN"
SCEN = sc.SCENARIOS["Defense takes over"]
HALVES = {"Turnover half only": {"takeaways": 2, "giveaways": 0},
          "Sack half only": {"sacks_made": 4, "sacks_taken": 1},
          "Both (defense takes over)": SCEN}
W_PRIOR = 0.20


def meets(df, cond):
    hit = np.ones(len(df), bool)
    for k, v in cond.items():
        hit &= (df[k] >= v) if sc.GOOD_HIGH[k] else (df[k] <= v)
    return hit


def poisson_tail(lam, k, at_least):
    cdf = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + (0 if at_least else 1)))
    return 1 - cdf if at_least else cdf


def scenario_prob(lams):
    """P(all four lever conditions) assuming independent Poisson counts."""
    return math.prod(poisson_tail(lams[k], v, sc.GOOD_HIGH[k]) for k, v in SCEN.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    data, outd = Path(a.data_dir), Path(a.out_dir)

    games = pd.read_csv(data / "games.csv")
    sc.SEASONS = range(2019, 2027)
    tg = sc.team_games(data).merge(games[["game_id", "season", "week", "home_team", "away_team",
                                          "home_score", "away_score", "spread_line", "total_line"]], on="game_id")
    home = tg.team == tg.home_team
    tg["pts"] = np.where(home, tg.home_score, tg.away_score)
    tg["opp_pts"] = np.where(home, tg.away_score, tg.home_score)
    tg["exp"] = np.where(home, (tg.total_line + tg.spread_line) / 2, (tg.total_line - tg.spread_line) / 2)
    tg["opp_exp"] = tg.total_line - tg.exp
    hist = tg[(tg.season <= 2025) & tg.pts.notna() & tg.spread_line.notna() & tg.total_line.notna()].copy()

    gm = games[games.game_id == GAME].iloc[0]
    den_exp, sea_exp = (gm.total_line + gm.spread_line) / 2, (gm.total_line - gm.spread_line) / 2

    # 1 + 2) comparable games laid on tonight's line, for each half and the full scenario
    out = {"halves": {}}
    print(f"Tonight's line: {HOME} {den_exp:.1f}, {AWAY} {sea_exp:.1f}\n")
    print(f"{'condition':<28}{'games':>6}{'happens':>9}{'DEN pts':>9}{'SEA pts':>9}{'DEN wins':>10}{'by 7+':>7}{'by 14+':>8}")
    base = hist.assign(m=(hist.pts - hist.exp) - (hist.opp_pts - hist.opp_exp) + (den_exp - sea_exp))
    for name, cond in {"Any game": {}, **HALVES}.items():
        c = base[meets(base, cond)]
        den_pts, sea_pts = den_exp + (c.pts - c.exp), sea_exp + (c.opp_pts - c.opp_exp)
        row = {"games": len(c), "happens": len(c) / len(base),
               "den_pts": float(den_pts.median()), "sea_pts": float(sea_pts.median()),
               "win": float((c.m > 0).mean() + 0.5 * (c.m == 0).mean()),
               "by7": float((c.m >= 7).mean()), "by14": float((c.m >= 14).mean())}
        out["halves"][name] = row
        print(f"{name:<28}{row['games']:>6}{row['happens']:>9.1%}{row['den_pts']:>9.1f}{row['sea_pts']:>9.1f}"
              f"{row['win']:>10.0%}{row['by7']:>7.0%}{row['by14']:>8.0%}")

    c = base[meets(base, SCEN)]
    den_pts, sea_pts = den_exp + (c.pts - c.exp), sea_exp + (c.opp_pts - c.opp_exp)
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    print("\nDefense takes over, on tonight's line (percentiles 10/25/50/75/90):")
    print("  DEN points  " + "  ".join(f"{v:5.1f}" for v in den_pts.quantile(q)))
    print("  SEA points  " + "  ".join(f"{v:5.1f}" for v in sea_pts.quantile(q)))
    print("  DEN margin  " + "  ".join(f"{v:+5.1f}" for v in c.m.quantile(q)))
    out["margins"] = {"scenario": c.m.round(1).tolist(), "any": base.m.round(1).tolist()}
    out["percentiles"] = {"den": den_pts.quantile(q).round(1).tolist(), "sea": sea_pts.quantile(q).round(1).tolist(),
                          "margin": c.m.quantile(q).round(1).tolist()}

    # 3) matchup-specific likelihood
    lg = {k: hist[k].mean() for k in sc.LEVERS}
    recent = tg[tg.season >= 2025].assign(w=lambda d: np.where(d.season == 2026, 1.0, W_PRIOR))
    recent = recent[(recent.season == 2025) | (recent.week < 6)]

    def rate(team, col, by="team"):
        d = recent[recent[by] == team]
        return float((d[col] * d.w).sum() / d.w.sum())

    # DEN's lever counts tonight = DEN tendency x SEA tendency / league (log5-style)
    lam = {"takeaways": rate(HOME, "takeaways") * rate(AWAY, "giveaways") / lg["giveaways"],
           "giveaways": rate(HOME, "giveaways") * rate(AWAY, "takeaways") / lg["takeaways"],
           "sacks_made": rate(HOME, "sacks_made") * rate(AWAY, "sacks_taken") / lg["sacks_taken"],
           "sacks_taken": rate(HOME, "sacks_taken") * rate(AWAY, "sacks_made") / lg["sacks_made"]}
    league_indep = scenario_prob({k: lg[k] for k in SCEN})
    league_real = float(meets(hist, SCEN).mean())
    corr = league_real / league_indep       # levers move together (good defensive days cluster)
    matchup = scenario_prob(lam) * corr
    print(f"\nExpected counts tonight (DEN): " + ", ".join(f"{k} {v:.2f} (lg {lg[k]:.2f})" for k, v in lam.items()))
    print(f"League: independence says {league_indep:.2%}, reality {league_real:.2%} -> clustering factor {corr:.2f}")
    print(f"Tonight, DEN vs SEA: {matchup:.1%} (about 1 game in {1 / matchup:.0f})")
    for k in SCEN:
        print(f"  P({k} {'>=' if sc.GOOD_HIGH[k] else '<='} {SCEN[k]}): "
              f"tonight {poisson_tail(lam[k], SCEN[k], sc.GOOD_HIGH[k]):.0%}, league {poisson_tail(lg[k], SCEN[k], sc.GOOD_HIGH[k]):.0%}")
    out["likelihood"] = {"lambda": lam, "league_lambda": {k: lg[k] for k in SCEN}, "league_indep": league_indep,
                         "league_real": league_real, "clustering": corr, "matchup": matchup}

    # 4) Denver's own "defense takes over" games
    den = tg[(tg.team == HOME) & meets(tg, SCEN)].sort_values(["season", "week"])
    print(f"\nBroncos 'defense takes over' games since 2019 ({len(den)}):")
    rows = []
    for g in den.itertuples():
        res = "W" if g.pts > g.opp_pts else ("L" if g.pts < g.opp_pts else "T")
        rows.append({"season": g.season, "week": g.week, "opp": g.opp, "score": f"{g.pts:.0f}-{g.opp_pts:.0f}",
                     "result": res, "takeaways": g.takeaways, "sacks": g.sacks_made})
        print(f"  {g.season} wk {g.week:>2} vs {g.opp:<3} {res} {g.pts:.0f}-{g.opp_pts:.0f}   "
              f"takeaways {g.takeaways}, sacks {g.sacks_made:.0f}, sacks allowed {g.sacks_taken:.0f}")
    out["denver_games"] = rows
    with open(outd / "defense_takes_over.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
