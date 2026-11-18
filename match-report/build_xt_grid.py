"""Build the xT grid (xt16.npy) from the 64 matches of the 2018 World Cup.

Method: Soccermatics lesson 4 (Singh's Markov-chain xT) on StatsBomb open data.
Run once:  python build_xt_grid.py
"""
import warnings
import numpy as np
import pandas as pd
from statsbombpy import sb

from match_report import NX, NY, zone

warnings.filterwarnings("ignore")


def build_xt(ev):
    n = NX * NY
    mv = ev[ev["type"].isin(["Pass", "Carry"]) & (ev["period"] < 5)].copy()
    mv["end"] = mv["pass_end_location"].combine_first(mv["carry_end_location"])
    mv = mv.dropna(subset=["location", "end"])
    start = zone(mv["location"].apply(lambda l: l[0]), mv["location"].apply(lambda l: l[1]))
    end = zone(mv["end"].apply(lambda l: l[0]), mv["end"].apply(lambda l: l[1]))
    ok = mv["pass_outcome"].isna().to_numpy()

    sh = ev[(ev["type"] == "Shot") & (ev["shot_type"] != "Penalty") & (ev["period"] < 5)]
    sh = sh.dropna(subset=["location"])
    shot_zone = zone(sh["location"].apply(lambda l: l[0]), sh["location"].apply(lambda l: l[1]))
    goal = (sh["shot_outcome"] == "Goal").to_numpy()

    move_n = np.bincount(start, minlength=n)
    shot_n = np.bincount(shot_zone, minlength=n)
    goal_n = np.bincount(shot_zone[goal], minlength=n)
    total = move_n + shot_n
    p_move = np.divide(move_n, total, out=np.zeros(n), where=total > 0)
    p_shot = np.divide(shot_n, total, out=np.zeros(n), where=total > 0)
    p_goal = np.divide(goal_n, shot_n, out=np.zeros(n), where=shot_n > 0)

    T = np.zeros((n, n))
    np.add.at(T, (start[ok], end[ok]), 1)
    T = np.divide(T, move_n[:, None], out=np.zeros((n, n)), where=move_n[:, None] > 0)

    xt = np.zeros(n)
    for i in range(200):
        new = p_shot * p_goal + p_move * (T @ xt)
        if np.abs(new - xt).max() < 1e-6:
            break
        xt = new
    return xt.reshape(NY, NX)


if __name__ == "__main__":
    matches = sb.matches(competition_id=43, season_id=3)
    frames = [sb.events(match_id=m) for m in matches["match_id"]]
    ev = pd.concat(frames, ignore_index=True)
    for c in ["pass_end_location", "carry_end_location", "pass_outcome",
              "shot_type", "shot_outcome"]:
        if c not in ev.columns:
            ev[c] = np.nan
    grid = build_xt(ev)
    np.save("xt16.npy", grid)
    print("saved xt16.npy", grid.shape)
