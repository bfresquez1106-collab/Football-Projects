#!/usr/bin/env python3
"""
Walk-forward replay that calibrates the model against real results only (no betting lines).

For every week of 2022 onward, fit ratings on plays known before kickoff, apply that week's
injury report, and predict every game. Then learn from the replay:

  calibration   result ~ a + b * raw margin, total ~ a + b * raw total. A slope below 1 means
                the raw model is overconfident and its margins get shrunk.
  sd / key wts  spread of real margins around the calibrated forecast, and how much more often
                3, 7, ... happen than a smooth normal says.
  levers        points over the model's expectation per takeaway, sack, explosive play.
  scenarios     how often a team really has each upside game.
  scorecard     Brier score, log loss, errors and calibration, with the calibration for each
                season fit on the OTHER seasons, so the scorecard isn't graded on its own answers.

Writes params.json (used by weekly.py, predict.py and the dashboard).
Run:  python3 calibrate.py --data-dir data         (~3 minutes)
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel import data as D
from nflmodel import qb
from nflmodel.engine import GOOD_HIGH, K, LEVERS, SCENARIOS, key_weights, normal_pmf
from nflmodel.injuries import team_missing, offense_adj
from nflmodel.ratings import ALPHA, FADE_WEEKS, W_EARLY, W_PRIOR, Ratings, prior_weight, training_frame

FIRST = 2022


def replay(data_dir, last_season, w_prior=W_PRIOR, alpha=ALPHA, qb_beta=0.0, qb_params="qb_params.json"):
    seasons = range(FIRST - 1, last_season + 1)
    pbp = {s: D.pbp(data_dir, s) for s in seasons}
    snaps = {s: D.snaps(data_dir, s) for s in seasons}
    inj = {s: D.injuries(data_dir, s) for s in seasons}
    games = D.games(data_dir)
    if qb_beta:
        qp = json.load(open(qb_params))
        qsums, _ = qb.load_sums([Path(data_dir) / f"pbp_{s}.csv.gz" for s in range(FIRST - 4, last_season + 1)
                                 if (Path(data_dir) / f"pbp_{s}.csv.gz").exists()])
        repl = qb.replacement_rating(qp)
    rows = []
    for season in range(FIRST, last_season + 1):
        sg = games[(games.season == season) & (games.game_type == "REG") & games.result.notna()]
        for week in sorted(sg.week.unique()):
            tf = training_frame(pbp, season, week, w_prior)
            r = Ratings(tf, alpha)
            qdelta = {}
            if qb_beta:
                # starter's rating minus the QB rating already inside the team's numbers
                qr = qb.ratings_as_of(qsums, season, week, qp).rating
                base = qb.team_baseline(tf[(tf.qb_dropback == 1) & tf.passer_id.notna()], qr, repl)
                cur = pbp[season][(pbp[season].week == week) & (pbp[season].qb_dropback == 1) & pbp[season].passer_id.notna()]
                for (gid, team), pid in qb.team_starters(cur).items():
                    qdelta[(gid, team)] = qr.get(pid, repl) - base.get(team, repl)
            for g in sg[sg.week == week].itertuples():
                if g.home_team not in r.tix or g.away_team not in r.tix:
                    continue
                hm = team_missing(snaps[season], snaps[season - 1], inj[season], week, g.home_team)
                am = team_missing(snaps[season], snaps[season - 1], inj[season], week, g.away_team)
                hadj, aadj = offense_adj(am), offense_adj(hm)
                hq, aq = qdelta.get((g.game_id, g.home_team), 0.0), qdelta.get((g.game_id, g.away_team), 0.0)
                rh, ra = r.raw_scores(g.home_team, g.away_team, g.location == "Neutral",
                                      (hadj[0] + qb_beta * hq, hadj[1]), (aadj[0] + qb_beta * aq, aadj[1]))
                rows.append({"game_id": g.game_id, "season": season, "week": week, "home": g.home_team,
                             "away": g.away_team, "neutral": int(g.location == "Neutral"), "home_qb_delta": hq, "away_qb_delta": aq,
                             "raw_home": rh, "raw_away": ra,
                             "home_score": g.home_score, "away_score": g.away_score})
        print(f"  replayed {season}", flush=True)
    bt = pd.DataFrame(rows)
    bt["raw_m"], bt["raw_t"] = bt.raw_home - bt.raw_away, bt.raw_home + bt.raw_away
    bt["result"], bt["total"] = bt.home_score - bt.away_score, bt.home_score + bt.away_score
    lev = pd.concat([D.team_game_levers(pbp[s]) for s in range(FIRST, last_season + 1)])
    return bt, lev


def fit_calibration(d):
    # margin: extra home-field points (only when there IS a home team) + shrink of the raw margin
    Xm = np.c_[1 - d.neutral.values, d.raw_m.values]
    mh, ms = np.linalg.lstsq(Xm, d.result.values, rcond=None)[0]
    ti, ts = np.linalg.lstsq(np.c_[np.ones(len(d)), d.raw_t.values], d.total.values, rcond=None)[0]
    return {"m_home": float(mh), "m_slope": float(ms), "t_icpt": float(ti), "t_slope": float(ts)}


def apply(d, c):
    return c["m_home"] * (1 - d.neutral) + c["m_slope"] * d.raw_m, c["t_icpt"] + c["t_slope"] * d.raw_t


def win_prob(m, sd, w):
    p = normal_pmf(m, sd) * w
    p /= p.sum()
    return float(p[K > 0].sum() + 0.5 * p[K == 0].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--season", type=int, default=date.today().year if date.today().month >= 9 else date.today().year - 1)
    ap.add_argument("--fetch", action="store_true", help="download missing seasons first")
    ap.add_argument("--w-prior", type=float, default=W_PRIOR, help="weight on last season's plays")
    ap.add_argument("--w-early", type=float, default=W_EARLY, help="Week 1 weight on last season, fading to --w-prior")
    ap.add_argument("--fade-weeks", type=int, default=FADE_WEEKS, help="weeks for --w-early to fade out")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="ridge shrinkage")
    ap.add_argument("--qb-beta", type=float, default=1.0, help="weight on the QB adjustment (0 = off)")
    ap.add_argument("--qb-params", default="qb_params.json", help="QB index settings to use")
    ap.add_argument("--tag", help="experiment name: writes experiments/<tag>.json instead of params.json")
    a = ap.parse_args()
    if a.fetch:
        D.fetch(a.data_dir, range(FIRST - 1, a.season + 1), current=a.season)

    print("Replaying seasons week by week...")
    wp = a.w_prior if a.w_early is None else (lambda wk: prior_weight(wk, a.w_prior, a.w_early, a.fade_weeks))
    bt, lev = replay(a.data_dir, a.season, wp, a.alpha, a.qb_beta, a.qb_params)

    # calibration (all seasons) + leave-one-season-out version for an honest scorecard
    cal = fit_calibration(bt)
    bt["m"], bt["t"] = apply(bt, cal)
    sd_m = float((bt.result - bt.m).std())
    sd_t = float((bt.total - bt.t).std())
    w = key_weights(bt.m.values, bt.result.values, sd_m)
    cv = []
    for s in bt.season.unique():
        c = fit_calibration(bt[bt.season != s])
        m, t = apply(bt[bt.season == s], c)
        cv.append(pd.DataFrame({"cv_m": m, "cv_t": t}, index=bt.index[bt.season == s]))
    bt = bt.join(pd.concat(cv))
    bt["cv_wp"] = [win_prob(m, sd_m, w) for m in bt.cv_m]
    bt["raw_wp"] = [win_prob(m, sd_m, w) for m in bt.raw_m]

    # levers: team points over the calibrated expectation
    side = pd.concat([
        pd.DataFrame({"game_id": bt.game_id, "season": bt.season, "team": bt.home, "pts": bt.home_score,
                      "exp": (bt.t + bt.m) / 2, "opp_pts": bt.away_score, "opp_exp": (bt.t - bt.m) / 2}),
        pd.DataFrame({"game_id": bt.game_id, "season": bt.season, "team": bt.away, "pts": bt.away_score,
                      "exp": (bt.t - bt.m) / 2, "opp_pts": bt.home_score, "opp_exp": (bt.t + bt.m) / 2})])
    tg = side.merge(lev, on=["game_id", "team"])
    means = tg[LEVERS].mean()
    X = np.c_[np.ones(len(tg)), (tg[LEVERS] - means).values]
    beta = np.linalg.lstsq(X, (tg.pts - tg.exp).values, rcond=None)[0]
    coef = dict(zip(LEVERS, beta[1:]))
    tg["fit"] = X @ beta
    # margin spread once the levers are known (both teams' fitted shifts), for scenario win probs
    pair = tg.merge(tg[["game_id", "team", "fit"]].rename(columns={"team": "opp", "fit": "opp_fit"}), on=["game_id", "opp"])
    resid_m = (pair.pts - pair.opp_pts) - (pair.exp - pair.opp_exp) - (pair.fit - pair.opp_fit)
    sd_m_levers = float(resid_m.std())
    freq = {}
    for name, cond in SCENARIOS.items():
        hit = np.ones(len(tg), bool)
        for k, v in cond.items():
            hit &= (tg[k] >= v) if GOOD_HIGH[k] else (tg[k] <= v)
        freq[name] = float(hit.mean())

    # scorecard
    y = (bt.result > 0).astype(float)
    nt = bt.result != 0

    def score(pcol, mcol, tcol):
        p = bt[pcol].clip(1e-6, 1 - 1e-6)
        return {"brier": float(((p - y) ** 2)[nt].mean()),
                "log_loss": float((-(y * np.log(p) + (1 - y) * np.log(1 - p)))[nt].mean()),
                "mae_margin": float((bt.result - bt[mcol]).abs().mean()),
                "mae_total": float((bt.total - bt[tcol]).abs().mean()),
                "winner_pct": float(((bt[pcol] > 0.5) == (y == 1))[nt].mean())}
    home_rate = float(y[nt].mean())
    bt["base_wp"], bt["base_m"], bt["base_t"] = home_rate, 0.0, float(bt.total.mean())
    card = {"games": int(len(bt)), "model": score("cv_wp", "cv_m", "cv_t"),
            "raw_model": score("raw_wp", "raw_m", "raw_t"),
            "always_home": score("base_wp", "base_m", "base_t")}
    bins = pd.cut(bt.cv_wp, [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    card["calibration"] = [{"bin": str(b), "games": int(len(d)), "predicted": float(d.cv_wp.mean()),
                            "actual": float((d.result > 0).mean() + 0.5 * (d.result == 0).mean())}
                           for b, d in bt.groupby(bins, observed=True)]
    card["by_season"] = {int(s): {"games": int(len(d)), "brier": float(((d.cv_wp - (d.result > 0)) ** 2).mean()),
                                  "mae_margin": float((d.result - d.cv_m).abs().mean())}
                         for s, d in bt.groupby("season")}

    params = {"built": str(date.today()), "seasons": f"{FIRST}-{a.season}", "w_prior": a.w_prior, "w_early": a.w_early, "fade_weeks": a.fade_weeks, "alpha": a.alpha, "qb_beta": a.qb_beta,
              "calibration": cal, "sd_margin": sd_m, "sd_total": sd_t, "key_weights": w.round(4).tolist(),
              "lever_coef": coef, "lever_means": means.to_dict(), "sd_margin_given_levers": sd_m_levers,
              "scenario_freq": freq, "scorecard": card}
    out = Path(f"experiments/{a.tag}.json") if a.tag else Path("params.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(params, f, indent=1)
    bt.to_csv(f"experiments/{a.tag}_games.csv" if a.tag else "outputs/replay_games.csv", index=False)

    print(f"\n{len(bt)} games replayed. Calibration: margin = {cal['m_home']:+.2f} (if home) + {cal['m_slope']:.2f} x raw, "
          f"total = {cal['t_icpt']:+.1f} + {cal['t_slope']:.2f} x raw")
    print(f"Margin SD {sd_m:.1f} (given levers {sd_m_levers:.1f}), total SD {sd_t:.1f}")
    print("Scorecard (season-out calibration):")
    for k in ("model", "raw_model", "always_home"):
        print(f"  {k:<12} " + "  ".join(f"{m} {v:.3f}" for m, v in card[k].items()))
    print("Calibration:", [(c["bin"], c["games"], round(c["predicted"], 2), round(c["actual"], 2)) for c in card["calibration"]])
    print("By season:", {s: (v["games"], round(v["brier"], 3), round(v["mae_margin"], 1)) for s, v in card["by_season"].items()})
    print("Lever points:", {k: round(v, 2) for k, v in coef.items()})
    print("Scenario frequency:", {k: round(v, 3) for k, v in freq.items()})


if __name__ == "__main__":
    main()
