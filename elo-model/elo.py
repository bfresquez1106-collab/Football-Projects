#!/usr/bin/env python3
"""
Elo for the NFL: one number per team, nudged after every game.

The whole idea in three lines:
  1. Every team has a rating. Average is 1500.
  2. Before a game, the rating gap turns into a win probability.
  3. After the game, the winner takes rating points from the loser. Upsets move ratings more,
     because the model was more wrong.

Run:  python3 elo.py --games ../nfl-model/data/games.csv
"""
import argparse
import math
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}   # same franchise, new city
TRAIN, TEST = range(2002, 2022), range(2022, 2027)     # tune on the past, judge on the future
POINTS_PER_ELO = 1 / 25                                # 25 Elo points ~ 1 point on the scoreboard


def win_prob(elo_gap):
    """The Elo curve: a 0 gap is 50%, +100 is 64%, +200 is 76%, +400 is 91%."""
    return 1 / (1 + 10 ** (-elo_gap / 400))


def margin_multiplier(margin, winner_gap):
    """
    Optional upgrade: let a 28-point win count for more than a 1-point win.
      log(|margin| + 1)   grows with the margin but flattens out (a 50-point blowout isn't 2x a 25).
      2.2 / (...)         shrinks the bonus when a big favorite wins big, because that was expected
                          anyway. Without it, good teams' ratings slowly inflate.
    (This is the formula FiveThirtyEight used for its NFL Elo.)
    """
    return math.log(abs(margin) + 1) * 2.2 / (winner_gap * 0.001 + 2.2)


def run_elo(games, k=20, hfa=55, revert=1 / 3, use_margin=True):
    """
    Walk through every game in date order. For each one:
      predict first (using only ratings built from earlier games), THEN update.
    That order is what makes this an honest, walk-forward forecast.

      k        how fast ratings react. Too high: chases noise. Too low: slow to notice change.
      hfa      home-field advantage, in Elo points.
      revert   share of each team's distance from 1500 erased between seasons
               (rosters change, so last year's rating is only partly true this year).
    """
    ratings = defaultdict(lambda: 1500.0)
    season, rows = None, []
    for g in games.itertuples():
        if g.season != season:                           # new season: pull everyone toward 1500
            for t in ratings:
                ratings[t] = 1500 + (1 - revert) * (ratings[t] - 1500)
            season = g.season

        home, away = g.home_team, g.away_team
        gap = ratings[home] + (0 if g.location == "Neutral" else hfa) - ratings[away]
        p_home = win_prob(gap)
        rows.append({"game_id": g.game_id, "season": g.season, "week": g.week, "game_type": g.game_type,
                     "home": home, "away": away, "home_elo": ratings[home], "away_elo": ratings[away],
                     "p_home": p_home, "elo_margin": gap * POINTS_PER_ELO, "result": g.result})

        actual = 1.0 if g.result > 0 else 0.0 if g.result < 0 else 0.5
        mult = 1.0
        if use_margin and g.result != 0:
            winner_gap = gap if g.result > 0 else -gap
            mult = margin_multiplier(g.result, winner_gap)
        change = k * mult * (actual - p_home)            # surprise x speed
        ratings[home] += change
        ratings[away] -= change                          # zero-sum: points move, never appear
    return pd.DataFrame(rows), dict(ratings)


def scores(p, result):
    """Brier score (lower is better), log loss (lower is better), share of winners picked."""
    y = np.where(result > 0, 1.0, np.where(result < 0, 0.0, 0.5))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
            "winner": float(np.mean(np.where(result == 0, 0.5, (p > 0.5) == (result > 0))))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="../nfl-model/data/games.csv")
    ap.add_argument("--replay", default="../nfl-model/outputs/replay_games.csv")
    a = ap.parse_args()

    g = pd.read_csv(a.games)
    g = g[g.result.notna()].replace({"home_team": RELOCATED, "away_team": RELOCATED})
    g = g.sort_values(["gameday", "gametime", "game_id"])     # strict date order

    # 1) Tune on 2002-2021 only. The 2022+ games stay locked away until the final exam.
    print("Tuning on 2002-2021 regular seasons (log loss, lower is better):")
    results = []
    for k, hfa, revert, mov in product([10, 20, 30], [40, 55, 70], [0, 1 / 3, 1 / 2], [False, True]):
        preds, _ = run_elo(g, k, hfa, revert, mov)
        tr = preds[preds.season.isin(TRAIN) & (preds.game_type == "REG")]
        results.append(((k, hfa, revert, mov), scores(tr.p_home, tr.result)["log_loss"]))
    results.sort(key=lambda x: x[1])
    for (k, hfa, revert, mov), ll in results[:3] + results[-2:]:
        print(f"  k {k:>2}  home {hfa}  revert {revert:.2f}  margin {'on ' if mov else 'off'}   {ll:.4f}")
    (k, hfa, revert, mov), _ = results[0]
    plain = min((r for r in results if not r[0][3]), key=lambda x: x[1])
    print(f"Best: k={k}, home field={hfa} Elo (~{hfa * POINTS_PER_ELO:.1f} pts), revert={revert:.2f}, "
          f"margin of victory {'on' if mov else 'off'}  (best without margin: {plain[1]:.4f})\n")

    # 2) Final exam on 2022-2026, the same games the Gridiron Model's replay was scored on.
    preds, ratings = run_elo(g, k, hfa, revert, mov)
    test = preds[preds.season.isin(TEST) & (preds.game_type == "REG")]
    grid = pd.read_csv(a.replay)[["game_id", "cv_wp"]]
    both = test.merge(grid, on="game_id")
    home_rate = float((preds[preds.season.isin(TRAIN) & (preds.game_type == "REG")].result > 0).mean())
    contenders = {"Always home team": np.full(len(both), home_rate), "Elo": both.p_home.values,
                  "Gridiron Model": both.cv_wp.values, "Average of Elo + Gridiron": (both.p_home + both.cv_wp).values / 2}
    print(f"Test: {len(both)} games, 2022-2026 (same games for everyone)")
    print(f"  {'model':<28}{'Brier':>8}{'log loss':>10}{'winners':>9}")
    for name, p in contenders.items():
        s = scores(p, both.result.values)
        print(f"  {name:<28}{s['brier']:>8.3f}{s['log_loss']:>10.3f}{s['winner']:>9.1%}")
    print(f"  Elo margin miss: {np.mean(np.abs(both.result - both.elo_margin)):.1f} pts")
    print(f"  Elo and Gridiron picked the same winner in {np.mean((both.p_home > 0.5) == (both.cv_wp > 0.5)):.0%} of games\n")

    # 3) One game, step by step: last night.
    ex = preds[preds.game_id == "2026_06_SEA_DEN"].iloc[0]
    gap = ex.home_elo + hfa - ex.away_elo
    actual = 1.0 if ex.result > 0 else 0.0
    mult = margin_multiplier(ex.result, gap if ex.result > 0 else -gap) if mov else 1.0
    change = k * mult * (actual - ex.p_home)
    print("Worked example: SEA at DEN, Week 6 2026")
    print(f"  Ratings before:  DEN {ex.home_elo:.0f}, SEA {ex.away_elo:.0f}")
    print(f"  Gap = {ex.home_elo:.0f} + {hfa} home - {ex.away_elo:.0f} = {gap:+.0f}  ->  DEN win prob {ex.p_home:.1%}, "
          f"margin {gap * POINTS_PER_ELO:+.1f} pts")
    print(f"  Result: DEN by {ex.result:.0f}.  Surprise = {actual:.0f} - {ex.p_home:.3f} = {actual - ex.p_home:+.3f}")
    print(f"  Change = k {k} x margin multiplier {mult:.2f} x surprise = {change:+.1f}  ->  "
          f"DEN {ex.home_elo + change:.0f}, SEA {ex.away_elo - change:.0f}\n")

    top = sorted(ratings.items(), key=lambda x: -x[1])
    print("Current ratings: " + ", ".join(f"{i + 1}. {t} {r:.0f}" for i, (t, r) in enumerate(top[:8])))
    print(f"  ... DEN #{[t for t, _ in top].index('DEN') + 1} at {ratings['DEN']:.0f}")
    preds.to_csv("elo_predictions.csv", index=False)


if __name__ == "__main__":
    main()
