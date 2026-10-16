#!/usr/bin/env python3
"""
The weekly run. Scheduled Tue / Thu / Sat mornings, or run by hand:

  1. Refresh nflverse data (schedule, current-season play-by-play, snaps, injury reports).
  2. Fit ratings on everything known so far and read this week's injury report.
  3. Predict every game in the upcoming week and log each one that hasn't kicked off yet.
  4. Grade logged predictions against final scores (the last prediction logged before kickoff).
  5. Rebuild the dashboard page.

Run:  python3 weekly.py               (or --no-fetch to reuse downloaded data)
"""
import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nflmodel import data as D
from nflmodel import elo
from nflmodel.engine import K, SCENARIOS, Model
from nflmodel.injuries import EFFECTS, team_missing
from nflmodel.ratings import Ratings, training_frame

LOG = Path("forecast_log.csv")
LOG_COLS = ["game_id", "season", "week", "home", "away", "logged_at", "kickoff", "home_pts", "away_pts",
            "margin", "total", "home_wp", "elo_wp", "elo_margin", "home_db_out", "home_lb_out", "away_db_out", "away_lb_out", "params_built"]


def kickoff_utc(gameday, gametime):
    t = gametime if isinstance(gametime, str) else "13:00"
    et = datetime.strptime(f"{gameday} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
    return et.astimezone(timezone.utc)


def score(p, result):
    """Brier, log loss and winners picked for win probabilities `p` against home margins `result`."""
    won = (result > 0).astype(float) + 0.5 * (result == 0)
    p = p.clip(1e-6, 1 - 1e-6)
    return {"games": int(len(p)), "brier": float(((p - won) ** 2).mean()),
            "log_loss": float((-(won * np.log(p) + (1 - won) * np.log(1 - p))).mean()),
            "winner_pct": float(np.where(result == 0, 0.5, (p > 0.5) == (result > 0)).mean())}


def grade(log, games):
    """Latest prediction logged before kickoff for each finished game, scored against the result."""
    if log.empty:
        return pd.DataFrame(), {}
    pre = log[log.logged_at < log.kickoff].sort_values("logged_at").groupby("game_id").tail(1)
    g = pre.merge(games[["game_id", "home_score", "away_score"]], on="game_id")
    g = g[g.home_score.notna()].copy()
    if g.empty:
        return g, {}
    g["result"] = g.home_score - g.away_score
    g["actual_total"] = g.home_score + g.away_score
    won = (g.result > 0).astype(float) + 0.5 * (g.result == 0)
    p = g.home_wp.clip(1e-6, 1 - 1e-6)
    g["brier"] = (p - won) ** 2
    g["log_loss"] = -(won * np.log(p) + (1 - won) * np.log(1 - p))
    g["picked_winner"] = np.where(g.result == 0, 0.5, ((p > 0.5) == (g.result > 0)).astype(float))
    g["margin_err"] = g.result - g.margin
    g["total_err"] = g.actual_total - g.total
    summary = {"games": int(len(g)), "brier": float(g.brier.mean()), "log_loss": float(g.log_loss.mean()),
               "winner_pct": float(g.picked_winner.mean()), "mae_margin": float(g.margin_err.abs().mean()),
               "mae_total": float(g.total_err.abs().mean()),
               "home_rate_brier": float(((won.mean() - won) ** 2).mean())}
    # the benchmark, graded on the same games (games logged before Elo was added have no Elo row)
    e = g[g.elo_wp.notna()] if "elo_wp" in g else g.iloc[0:0]
    if len(e):
        summary["head_to_head"] = {"gridiron": score(e.home_wp, e.result), "elo": score(e.elo_wp, e.result)}
    bins = pd.cut(g.home_wp, [0, 0.35, 0.5, 0.65, 1.0])
    summary["calibration"] = [{"bin": str(b), "games": int(len(d)), "predicted": float(d.home_wp.mean()),
                               "actual": float((d.result > 0).mean() + 0.5 * (d.result == 0).mean())}
                              for b, d in g.groupby(bins, observed=True)]
    return g, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--season", type=int, default=date.today().year if date.today().month >= 8 else date.today().year - 1)
    ap.add_argument("--week", type=int, help="default: the next week with unplayed games")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    season, data = a.season, Path(a.data_dir)
    if not a.no_fetch:
        D.fetch(data, [season - 1, season], current=season)

    games = D.games(data)
    sched = games[(games.season == season) & (games.game_type == "REG")].copy()
    week = a.week or int(sched[sched.result.isna()].week.min())
    params = json.load(open("params.json"))
    pbp = {s: D.pbp(data, s) for s in (season - 1, season)}
    r = Ratings(training_frame(pbp, season, week))
    model = Model(r, params)
    snaps_cur, snaps_prior = D.snaps(data, season), D.snaps(data, season - 1)
    report = D.injuries(data, season)
    missing = {t: team_missing(snaps_cur, snaps_prior, report, week, t) for t in r.teams}

    elo_now = elo.ratings_through(games, season, week)
    elo_prev = elo.ratings_through(games, season, week - 1) if week > 1 else elo_now

    now = datetime.now(timezone.utc)
    slate, new_rows = [], []
    for g in sched[sched.week == week].sort_values(["gameday", "gametime"]).itertuples():
        pred = model.predict(g.home_team, g.away_team, g.location == "Neutral",
                             missing[g.home_team], missing[g.away_team])
        pred |= elo.predict(elo_now, g.home_team, g.away_team, g.location == "Neutral")
        ko = kickoff_utc(g.gameday, g.gametime)
        slate.append({**pred, "game_id": g.game_id, "gameday": g.gameday, "gametime": g.gametime,
                      "kickoff": ko.isoformat(), "played": bool(pd.notna(g.result)),
                      "home_score": None if pd.isna(g.home_score) else int(g.home_score),
                      "away_score": None if pd.isna(g.away_score) else int(g.away_score)})
        if ko > now:
            new_rows.append({"game_id": g.game_id, "season": season, "week": week, "home": g.home_team,
                             "away": g.away_team, "logged_at": now.isoformat(timespec="seconds"),
                             "kickoff": ko.isoformat(), "home_pts": round(pred["home_pts"], 2),
                             "away_pts": round(pred["away_pts"], 2), "margin": round(pred["margin"], 2),
                             "total": round(pred["total"], 2), "home_wp": round(pred["home_wp"], 4),
                             "elo_wp": round(pred["elo_wp"], 4), "elo_margin": round(pred["elo_margin"], 2),
                             "home_db_out": round(missing[g.home_team]["DB"], 2), "home_lb_out": round(missing[g.home_team]["LB"], 2),
                             "away_db_out": round(missing[g.away_team]["DB"], 2), "away_lb_out": round(missing[g.away_team]["LB"], 2),
                             "params_built": params["built"]})

    # reindex: older logs gain any newer columns (e.g. elo_wp) as blanks
    log = (pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()).reindex(columns=LOG_COLS)
    # only log a game again when its prediction changed (new injuries, new data, recalibration)
    last = log.sort_values("logged_at").groupby("game_id").tail(1).set_index("game_id") if len(log) else None
    key = ["home_pts", "away_pts", "home_wp", "elo_wp", "params_built"]
    new_rows = [x for x in new_rows if last is None or x["game_id"] not in last.index
                or any(str(last.loc[x["game_id"], k]) != str(x[k]) for k in key)]
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        log.to_csv(LOG, index=False)
    graded, summary = grade(log[log.season == season], games)

    print(f"Season {season}, week {week}: {len(slate)} games, {len(new_rows)} logged before kickoff")
    for s in slate:
        fav, wp = (s["home"], s["home_wp"]) if s["home_wp"] >= 0.5 else (s["away"], 1 - s["home_wp"])
        print(f"  {s['away']:>3} {s['away_pts']:4.1f} @ {s['home']:<3} {s['home_pts']:4.1f}   {fav} {wp:.0%}"
              f"   (Elo: {s['home'] if s['elo_wp'] >= 0.5 else s['away']} {max(s['elo_wp'], 1 - s['elo_wp']):.0%})"
              + (f"   final {s['away_score']}-{s['home_score']}" if s["played"] else ""))
    if summary:
        print(f"Season record: {summary['games']} games, picked winner {summary['winner_pct']:.0%}, "
              f"Brier {summary['brier']:.3f} (always-home-rate {summary['home_rate_brier']:.3f}), "
              f"margin miss {summary['mae_margin']:.1f}")
        if "head_to_head" in summary:
            h = summary["head_to_head"]
            print(f"Head to head on {h['elo']['games']} games: Gridiron Brier {h['gridiron']['brier']:.3f}, "
                  f"Elo {h['elo']['brier']:.3f}")

    # Elo's long-run record on the same replay games Gridiron was scored on
    hist = {}
    elo.ratings_through(games, season, week, history=hist)
    rp = pd.read_csv("outputs/replay_games.csv")[["game_id", "result", "cv_wp"]]
    rp["elo_wp"] = rp.game_id.map(hist)
    rp = rp[rp.elo_wp.notna()]
    elo_backtest = {"gridiron": score(rp.cv_wp, rp.result), "elo": score(rp.elo_wp, rp.result)}

    remaining = sched[sched.result.isna()]
    out = {
        "meta": {"season": season, "week": week, "generated": now.isoformat(timespec="minutes"),
                 "params_built": params["built"]},
        "ratings": r.export(),
        "params": {k: params[k] for k in ("calibration", "sd_margin", "sd_total", "key_weights", "lever_coef",
                                          "lever_means", "sd_margin_given_levers", "scenario_freq")}
                  | {"k_min": int(K[0]), "scenarios": SCENARIOS, "effects": EFFECTS},
        "missing": missing,
        "slate": slate,
        "schedule": [{"week": int(g.week), "home": g.home_team, "away": g.away_team, "gameday": g.gameday,
                      "neutral": g.location == "Neutral"} for g in remaining.itertuples()],
        "record": {"summary": summary,
                   "games": [] if graded.empty else graded[["game_id", "week", "home", "away", "home_pts", "away_pts",
                                                            "home_wp", "elo_wp", "home_score", "away_score", "brier"]]
                   .sort_values(["week", "game_id"], ascending=[False, True]).to_dict("records")},
        "backtest": params["scorecard"],
        "elo": {"ratings": {t: elo_now[t] for t in r.teams}, "prev": {t: elo_prev[t] for t in r.teams},
                "hfa": elo.HFA, "tiers": [{"name": n, "medal": m, "floor": f, "means": d} for n, m, f, d in elo.TIERS],
                "backtest": elo_backtest},
    }
    tpl = Path("dashboard/template.html").read_text()
    Path("dashboard/nfl_model.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(out, default=float)))
    print("Dashboard written to dashboard/nfl_model.html")


if __name__ == "__main__":
    main()
