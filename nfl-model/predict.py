#!/usr/bin/env python3
"""
Predict any matchup from the command line.

  python3 predict.py DEN KC                     # KC at DEN, this week's injury report
  python3 predict.py DEN KC --neutral
  python3 predict.py DEN KC --week 12 --home-db-out 2 --away-lb-out 1   # override who's missing
"""
import argparse
import json
from datetime import date

from nflmodel import data as D
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
    a = ap.parse_args()
    home, away = a.home.upper(), a.away.upper()

    games = D.games(a.data_dir)
    sched = games[(games.season == a.season) & (games.game_type == "REG")]
    week = a.week or int(sched[sched.result.isna()].week.min())
    pbp = {s: D.pbp(a.data_dir, s) for s in (a.season - 1, a.season)}
    model = Model(Ratings(training_frame(pbp, a.season, week)), json.load(open("params.json")))
    snaps, prior, rep = D.snaps(a.data_dir, a.season), D.snaps(a.data_dir, a.season - 1), D.injuries(a.data_dir, a.season)
    miss = {}
    for side, team in (("home", home), ("away", away)):
        m = team_missing(snaps, prior, rep, week, team)
        for g in ("db", "lb"):
            v = getattr(a, f"{side}_{g}_out")
            if v is not None:
                m[g.upper()] = v
        miss[side] = m

    p = model.predict(home, away, a.neutral, miss["home"], miss["away"])
    fav, wp = (home, p["home_wp"]) if p["home_wp"] >= 0.5 else (away, 1 - p["home_wp"])
    print(f"{away} at {home}{' (neutral)' if a.neutral else ''}, ratings through week {week - 1}")
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
