#!/usr/bin/env python3
"""
Grade logged pre-game forecasts against what happened.

  forecast_log.csv: one row per (game, forecaster), written BEFORE kickoff.
  Score level (needs games.csv): margin/total error, Brier score, log loss, and where the
    actual margin fell in each forecast's distribution (a 50% "percentile" = dead centre).
  Lever level (needs the game in play-by-play): turnovers, sacks and explosive plays vs what
    defense_takes_over.py expected, and which upside scenario the game actually matched.

Run:  python3 grade.py --data-dir data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import Outcomes
import scenarios as sc


def lever_check(data, game_id, outd):
    pbp = data / f"pbp_{game_id[:4]}.csv.gz"
    p = pd.read_csv(pbp, usecols=sc.COLS, low_memory=False)
    p = p[p.game_id == game_id]
    if p.empty:
        print("\nLever grading: play-by-play for this game isn't published yet. Re-download "
              f"{pbp.name} and run again.")
        return
    p = p[(p["pass"] == 1) | (p["rush"] == 1)].fillna({"sack": 0, "interception": 0, "fumble_lost": 0, "yards_gained": 0})
    p["give"] = p.interception + p.fumble_lost
    p["expl"] = (((p["pass"] == 1) & (p.sack != 1) & (p.yards_gained >= 20))
                 | ((p["rush"] == 1) & (p.yards_gained >= 10))).astype(int)
    t = p.groupby("posteam").agg(give=("give", "sum"), sacked=("sack", "sum"), expl=("expl", "sum"))
    home, away = game_id.split("_")[3], game_id.split("_")[2]
    actual = {"takeaways": t.loc[away, "give"], "giveaways": t.loc[home, "give"],
              "sacks_made": t.loc[away, "sacked"], "sacks_taken": t.loc[home, "sacked"],
              "expl_for": t.loc[home, "expl"], "expl_against": t.loc[away, "expl"]}
    exp = json.load(open(outd / "defense_takes_over.json"))["likelihood"]["lambda"]
    print(f"\nLever check ({home} perspective): actual vs expected")
    for k, v in actual.items():
        e = exp.get(k)
        print(f"  {k:<13} {v:>3.0f}" + (f"   expected {e:.2f}" if e is not None else ""))
    print("Scenarios matched:")
    for name, cond in sc.SCENARIOS.items():
        ok = all((actual[k] >= v) if sc.GOOD_HIGH[k] else (actual[k] <= v) for k, v in cond.items())
        print(f"  {'YES' if ok else ' - '}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    data, outd = Path(a.data_dir), Path(a.out_dir)

    games = pd.read_csv(data / "games.csv")
    log = pd.read_csv("forecast_log.csv").rename(columns={"total": "total_fc"}).merge(
        games[["game_id", "season", "result", "total", "overtime"]], on="game_id", how="left")
    done = log[log.result.notna()].copy()
    if done.empty:
        print("No logged games have finished yet.")
        return

    out = Outcomes(games[(games.season >= 2016) & (games.season < done.season.min()) & (games.game_type == "REG")])
    won = (done.result > 0).astype(float)
    p = done.home_wp.clip(1e-6, 1 - 1e-6)
    done["margin_err"] = done.result - done.home_margin
    done["total_err"] = done.total - done.total_fc
    done["brier"] = (p - won) ** 2
    done["log_loss"] = -(won * np.log(p) + (1 - won) * np.log(1 - p))
    done["right_side"] = (p > 0.5) == (won == 1)
    # where the actual margin landed in each forecast's distribution (50% = dead centre)
    done["margin_pctile"] = [float(out.pmf(m)[out.K < r].sum() + 0.5 * out.pmf(m)[out.K == r].sum())
                             for m, r in zip(done.home_margin, done.result)]
    done["total_pctile"] = [1 - out.over_prob(t, r) for t, r in zip(done.total_fc, done.total)]

    for gid, g in done.groupby("game_id"):
        r = g.iloc[0]
        print(f"{gid}: home margin {r.result:+.0f}, total {r.total:.0f}{' (overtime)' if r.overtime == 1 else ''}\n")
        cols = ["forecaster", "home_margin", "margin_err", "total_fc", "total_err", "home_wp", "brier",
                "log_loss", "right_side", "margin_pctile", "total_pctile"]
        print(g[cols].round(3).to_string(index=False))
        lever_check(data, gid, outd)

    print("\nAll graded games, by forecaster (lower Brier / log loss is better):")
    print(done.groupby("forecaster").agg(games=("game_id", "size"), brier=("brier", "mean"),
                                         log_loss=("log_loss", "mean"), mae_margin=("margin_err", lambda x: x.abs().mean()),
                                         mae_total=("total_err", lambda x: x.abs().mean())).round(3).to_string())
    done.to_csv(outd / "graded_forecasts.csv", index=False)


if __name__ == "__main__":
    main()
