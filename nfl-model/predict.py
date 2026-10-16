#!/usr/bin/env python3
"""
Predict any matchup from the command line.

  python3 predict.py DEN KC                     # KC at DEN, this week's injury report
  python3 predict.py DEN KC --neutral
  python3 predict.py DEN KC --week 12 --home-db-out 2 --away-lb-out 1   # override who's missing
  python3 predict.py DEN KC --home-qb Stidham                           # what if the backup starts?
"""
import argparse
import json
import os
from datetime import date

import pandas as pd

from nflmodel import data as D, qb
from nflmodel.engine import Model
from nflmodel.injuries import team_missing
from nflmodel.ratings import Ratings, training_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--season", type=int, default=date.today().year if date.today().month >= 8 else date.today().year - 1)
    ap.add_argument("--week", type=int, help="ratings use games before this week (default: next unplayed)")
    for side in ("home", "away"):
        for g in ("db", "lb"):
            ap.add_argument(f"--{side}-{g}-out", type=float, help=f"starting {g.upper()}s missing for {side} (overrides report)")
    ap.add_argument("--home-qb", help="starting QB for the home team (part of his name); default depth chart")
    ap.add_argument("--away-qb", help="starting QB for the away team")
    a = ap.parse_args()
    home, away = a.home.upper(), a.away.upper()

    games = D.games(a.data_dir)
    sched = games[(games.season == a.season) & (games.game_type == "REG")]
    week = a.week or int(sched[sched.result.isna()].week.min())
    pbp = {s: D.pbp(a.data_dir, s) for s in (a.season - 1, a.season)}
    tf = training_frame(pbp, a.season, week)
    model = Model(Ratings(tf), json.load(open("params.json")))
    snaps, prior, rep = D.snaps(a.data_dir, a.season), D.snaps(a.data_dir, a.season - 1), D.injuries(a.data_dir, a.season)
    miss = {}
    for side, team in (("home", home), ("away", away)):
        m = team_missing(snaps, prior, rep, week, team)
        for g in ("db", "lb"):
            v = getattr(a, f"{side}_{g}_out")
            if v is not None:
                m[g.upper()] = v
        miss[side] = m

    # QB adjustment: starter's rating minus the QB rating already inside the team's numbers
    qp = json.load(open("qb_params.json"))
    qsums = qb.weekly_sums(pd.concat([qb.load_dropbacks(f"{a.data_dir}/pbp_{s}.csv.gz") for s in range(a.season - 3, a.season + 1)
                                      if os.path.exists(f"{a.data_dir}/pbp_{s}.csv.gz")]))
    qtab, repl = qb.ratings_as_of(qsums, a.season, week, qp), qb.replacement_rating(qp)
    base = qb.team_baseline(tf[(tf.qb_dropback == 1) & tf.passer_id.notna()], qtab.rating, repl)
    depth = D.depth_qbs(a.data_dir, a.season)
    starters = qb.live_starters(depth, rep, week)
    qbs, adj = {}, {}
    for side, team, pick in (("home", home, a.home_qb), ("away", away, a.away_qb)):
        if pick:
            hit = qtab[qtab.name.str.contains(pick, case=False, na=False) | qtab.index.isin(
                depth[depth.name.str.contains(pick, case=False)].gsis_id)]
            pid, name = (hit.index[0], hit.name.iloc[0]) if len(hit) else (None, f"{pick} (no NFL history)")
        else:
            st = starters.get(team, {"id": None, "name": "unknown"})
            pid, name = st["id"], st["name"]
        rating = float(qtab.rating.get(pid, repl))
        adj[side] = rating - float(base.get(team, repl))
        qbs[side] = (name, 100 + 10 * (rating - qp["starter_mean"]) / qp["starter_sd"])

    p = model.predict(home, away, a.neutral, miss["home"], miss["away"], adj["home"], adj["away"])
    fav, wp = (home, p["home_wp"]) if p["home_wp"] >= 0.5 else (away, 1 - p["home_wp"])
    print(f"{away} at {home}{' (neutral)' if a.neutral else ''}, ratings through week {week - 1}")
    print(f"  QBs: {away} {qbs['away'][0]} (index {qbs['away'][1]:.0f}, {adj['away']:+.3f} EPA/dropback vs team baseline), "
          f"{home} {qbs['home'][0]} (index {qbs['home'][1]:.0f}, {adj['home']:+.3f})")
    print(f"  Projected: {home} {p['home_pts']:.1f}, {away} {p['away_pts']:.1f}  (total {p['total']:.1f})")
    print(f"  {fav} wins {wp:.0%}.  {home} by 7+: {p['home_by_7']:.0%}, {away} by 7+: {p['away_by_7']:.0%}")
    q = p["margin_q"]
    print(f"  {home} margin, 10th-90th percentile: {q['p10']:+d} / {q['p25']:+d} / {q['p50']:+d} / {q['p75']:+d} / {q['p90']:+d}")
    for side, team in (("home", home), ("away", away)):
        m = miss[side]
        who = ", ".join(f"{x['name']} ({x['status']})" for x in m["players"]) or "none listed"
        print(f"  {team} starters missing: DB {m['DB']:.2f}, LB {m['LB']:.2f}  [{who}]")


if __name__ == "__main__":
    main()
