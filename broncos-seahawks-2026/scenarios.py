#!/usr/bin/env python3
"""
Broncos upside scenarios: what does the final score look like when the levers break Denver's way?

  1. From every regular-season game 2019-2025, build team-game rows: points scored minus what the
     betting line implied ((total +/- spread) / 2), and the game's lever stats (takeaways,
     giveaways, sacks made/taken, explosive plays for/against; explosive = rush 10+, pass 20+).
  2. Regress points-over-expectation on the levers. Both teams' scores come from the same
     equation with mirrored stats, so a scenario gives a coherent final score.
  3. For each scenario: projected score, a range (residual SD), and how often a team has really
     produced a game at least that good on every lever ("how ideal is ideal?").
  4. Comparable games: real games that met the scenario, scored against their own lines.

Descriptive, not causal: sacks and turnovers partly follow game script (trailing teams pass more).

Run:  python3 scenarios.py --data-dir data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOME, AWAY, GAME = "DEN", "SEA", "2026_06_SEA_DEN"
SEASONS = range(2019, 2026)
COLS = ["game_id", "season_type", "posteam", "defteam", "pass", "rush", "sack",
        "interception", "fumble_lost", "yards_gained", "qb_kneel", "qb_spike"]
LEVERS = ["takeaways", "giveaways", "sacks_made", "sacks_taken", "expl_for", "expl_against"]

# Denver-favorable scenarios, from modest to dream. Unlisted levers stay at league average.
SCENARIOS = {
    "Average game (market)": {},
    "Clean offense": {"giveaways": 0, "sacks_taken": 1},
    "Win turnovers by +2": {"takeaways": 2, "giveaways": 0},
    "Defense takes over": {"takeaways": 2, "giveaways": 0, "sacks_made": 4, "sacks_taken": 1},
    "Explosive night": {"giveaways": 0, "expl_for": 10, "expl_against": 5},
    "Great night": {"takeaways": 2, "giveaways": 0, "sacks_made": 3, "sacks_taken": 2,
                    "expl_for": 7, "expl_against": 5},
    "Ideal night": {"takeaways": 3, "giveaways": 0, "sacks_made": 5, "sacks_taken": 1,
                    "expl_for": 10, "expl_against": 5},
}
# Which way is "good for Denver" for each lever (used to measure how often a scenario happens)
GOOD_HIGH = {"takeaways": True, "giveaways": False, "sacks_made": True, "sacks_taken": False,
             "expl_for": True, "expl_against": False}


def team_games(data):
    parts = []
    for s in SEASONS:
        p = pd.read_csv(data / f"pbp_{s}.csv.gz", usecols=COLS, low_memory=False)
        p = p[(p.season_type == "REG") & p.posteam.notna() & ((p["pass"] == 1) | (p["rush"] == 1))
              & (p.qb_kneel != 1) & (p.qb_spike != 1)]
        p = p.fillna({"sack": 0, "interception": 0, "fumble_lost": 0, "yards_gained": 0})
        p["give"] = p.interception + p.fumble_lost
        p["expl"] = (((p["pass"] == 1) & (p.sack != 1) & (p.yards_gained >= 20))
                     | ((p["rush"] == 1) & (p.yards_gained >= 10))).astype(int)
        parts.append(p.groupby(["game_id", "posteam"]).agg(give=("give", "sum"), sacked=("sack", "sum"),
                                                           expl=("expl", "sum")).reset_index())
    off = pd.concat(parts)
    opp = off.rename(columns={"posteam": "opp", "give": "takeaways", "sacked": "sacks_made",
                              "expl": "expl_against"})
    off = off.rename(columns={"posteam": "team", "give": "giveaways", "sacked": "sacks_taken", "expl": "expl_for"})
    tg = off.merge(opp, on="game_id")
    return tg[tg.team != tg.opp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    data, outd = Path(a.data_dir), Path(a.out_dir)

    games = pd.read_csv(data / "games.csv")
    g = games[games.season.isin(SEASONS) & (games.game_type == "REG")].dropna(
        subset=["home_score", "spread_line", "total_line"])
    side = pd.concat([
        pd.DataFrame({"game_id": g.game_id, "team": g.home_team, "pts": g.home_score,
                      "exp": (g.total_line + g.spread_line) / 2, "opp_pts": g.away_score,
                      "opp_exp": (g.total_line - g.spread_line) / 2}),
        pd.DataFrame({"game_id": g.game_id, "team": g.away_team, "pts": g.away_score,
                      "exp": (g.total_line - g.spread_line) / 2, "opp_pts": g.home_score,
                      "opp_exp": (g.total_line + g.spread_line) / 2})])
    df = side.merge(team_games(data), on=["game_id", "team"])
    df["resid"] = df.pts - df.exp
    df["opp_resid"] = df.opp_pts - df.opp_exp
    mean = df[LEVERS].mean()

    X = np.c_[np.ones(len(df)), (df[LEVERS] - mean).values]
    beta, *_ = np.linalg.lstsq(X, df.resid.values, rcond=None)
    coef = dict(zip(LEVERS, beta[1:]))
    sd = float(np.std(df.resid.values - X @ beta, ddof=X.shape[1]))
    print(f"{len(df)} team-games, {SEASONS.start}-{SEASONS.stop - 1}. League average per team-game: "
          + ", ".join(f"{k} {v:.1f}" for k, v in mean.items()))
    print("Points over expectation per one extra:  " + ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))
    print(f"Unexplained spread of a team's score: SD {sd:.1f} pts\n")

    gm = games[games.game_id == GAME].iloc[0]
    den_exp, sea_exp = (gm.total_line + gm.spread_line) / 2, (gm.total_line - gm.spread_line) / 2
    mirror = {"takeaways": "giveaways", "giveaways": "takeaways", "sacks_made": "sacks_taken",
              "sacks_taken": "sacks_made", "expl_for": "expl_against", "expl_against": "expl_for"}

    rows = []
    for name, sc in SCENARIOS.items():
        den_x = {k: sc.get(k, mean[k]) for k in LEVERS}
        sea_x = {k: den_x[mirror[k]] for k in LEVERS}
        den = den_exp + sum(coef[k] * (den_x[k] - mean[k]) for k in LEVERS)
        sea = sea_exp + sum(coef[k] * (sea_x[k] - mean[k]) for k in LEVERS)
        # how often does a team have a game at least this good on every specified lever?
        hit = np.ones(len(df), bool)
        for k, v in sc.items():
            hit &= (df[k] >= v) if GOOD_HIGH[k] else (df[k] <= v)
        comp = df[hit]
        # Comparable games, each scored against its own line, then laid on tonight's line
        den_c, sea_c = den_exp + comp.resid.mean(), sea_exp + comp.opp_resid.mean()
        margin_c = (comp.resid - comp.opp_resid) + (den_exp - sea_exp)
        rows.append({"scenario": name, "DEN": den, "SEA": sea, "margin": den - sea,
                     "happens": hit.mean(), "n_games": int(hit.sum()),
                     "comp_DEN": den_c, "comp_SEA": sea_c,
                     "comp_win": float((margin_c > 0).mean() + 0.5 * (margin_c == 0).mean()),
                     "comp_win_by_7": float((margin_c >= 7).mean()),
                     "comp_p90_margin": float(margin_c.quantile(0.9))})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(outd / "scenarios.csv", index=False)


if __name__ == "__main__":
    main()
