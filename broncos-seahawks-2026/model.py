#!/usr/bin/env python3
"""
Opponent-adjusted EPA matchup model, blended with the betting market.

  1. Ridge regression of EPA/play on (offense + defense + home field), fit separately
     for pass and rush plays, garbage time removed (0.05 < WP < 0.95). Prior-season
     plays are down-weighted (W_PRIOR) so early-season ratings aren't pure noise.
  2. Matchup EPA/play -> points: league PPG + plays * (matchup EPA - league EPA).
  3. Blend the model margin with the market spread (weight chosen by backtest.py).
  4. Outcome distribution: a normal margin, re-weighted so key numbers (3, 7, ...)
     land as often as they really do. Gives win, cover and over/under probabilities.
  5. Lever analysis: what each controllable Broncos lever is worth in points and WP.

Run:  python3 -I model.py --data-dir data
"""
import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PBP_COLS = ["season", "season_type", "week", "game_id", "home_team", "away_team", "location",
            "posteam", "defteam", "pass", "rush", "epa", "wp", "qb_kneel", "qb_spike",
            "sack", "interception", "fumble_lost", "success", "down", "home_score", "away_score"]
_erf = np.vectorize(math.erf)


@dataclass
class Config:
    w_prior: float = 0.20       # weight on prior-season plays (backtest optimum)
    alpha: float = 250.0        # ridge shrinkage, in "plays"
    model_weight: float = 0.0   # margin blend weight on the model; backtest (319 games) found no edge over the line
    total_weight: float = 0.0   # total blend weight on the model
    tilt_decay: float = 0.5     # play-mix gains shrink as the defense adjusts; 1.0 = no shrink
    # Judgement-call injury adjustments, EPA/play: {(offense, defense): (d_pass, d_rush)}
    injuries: dict = field(default_factory=dict)


def wavg(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    return float((x[m] * w[m]).sum() / w[m].sum())


def norm_cdf(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2.0)))


# ----------------------------------- data ---------------------------------------
def load_pbp(path):
    df = pd.read_csv(path, usecols=lambda c: c in PBP_COLS, low_memory=False)
    df = df[(df.season_type == "REG") & df.posteam.notna() & df.defteam.notna()].copy()
    for c in ["pass", "rush", "qb_kneel", "qb_spike", "sack", "interception", "fumble_lost", "success"]:
        df[c] = df[c].fillna(0)
    return df


def training_frame(pbp_by_season, season, before_week, w_prior):
    """Plays known before kickoff of `season` week `before_week`: that season (w=1) + prior (w_prior)."""
    cur = pbp_by_season[season]
    cur = cur[cur.week < before_week].assign(w=1.0)
    prior = pbp_by_season.get(season - 1)
    parts = [cur] if prior is None else [prior.assign(w=w_prior), cur]
    return pd.concat(parts, ignore_index=True)


# ------------------------------- team ratings -----------------------------------
class Ratings:
    def __init__(self, raw, alpha):
        plays = raw[((raw["pass"] == 1) | (raw["rush"] == 1)) & (raw.qb_kneel != 1)
                    & (raw.qb_spike != 1) & raw.epa.notna()].copy()
        # +0.5 offense at home, -0.5 away, 0 at a neutral site
        plays["h"] = np.where(plays.location == "Neutral", 0.0,
                              np.where(plays.posteam == plays.home_team, 0.5, -0.5))
        clean = plays[(plays.wp > 0.05) & (plays.wp < 0.95)]
        self.plays, self.clean = plays, clean
        self.teams = sorted(set(clean.posteam) | set(clean.defteam))
        self.tix = {t: i for i, t in enumerate(self.teams)}
        self.PASS, self.RUSH = self._fit(clean[clean["pass"] == 1], alpha), self._fit(clean[clean["rush"] == 1], alpha)

        g = raw.groupby("game_id").agg(hs=("home_score", "max"), as_=("away_score", "max"), w=("w", "first"))
        self.ppg = wavg((g.hs + g.as_) / 2, g.w)
        gp = plays.groupby("game_id").size() / 2
        tg = plays.groupby(["game_id", "posteam"]).w.first().reset_index()
        tg["gp"] = tg.game_id.map(gp)
        self.pace = {t: wavg(d.gp, d.w) for t, d in tg.groupby("posteam")}
        # Pass share on the same plays the ratings were fit on, so the league baseline matches.
        self.pr = {t: wavg(d["pass"], d.w) for t, d in clean.groupby("posteam")}
        self.lg_pr = wavg(clean["pass"], clean.w)
        self.lg_epa = self.lg_pr * self.PASS["icpt"] + (1 - self.lg_pr) * self.RUSH["icpt"]

    def _fit(self, d, alpha):
        T, n = len(self.teams), len(d)
        X = np.zeros((n, 2 * T + 2))
        X[np.arange(n), d.posteam.map(self.tix).values] = 1.0
        X[np.arange(n), T + d.defteam.map(self.tix).values] = 1.0
        X[:, 2 * T] = d.h.values
        X[:, 2 * T + 1] = 1.0
        y, w = d.epa.values, d.w.values
        pen = np.r_[np.ones(2 * T), 0.0, 0.0]          # don't shrink home field or intercept
        beta = np.linalg.solve((X.T * w) @ X + alpha * np.diag(pen), (X.T * w) @ y)
        return {"off": dict(zip(self.teams, beta[:T])), "dfn": dict(zip(self.teams, beta[T:2 * T])),
                "home": float(beta[2 * T]), "icpt": float(beta[2 * T + 1]), "n": n}

    def table(self):
        df = pd.DataFrame({"off_pass": self.PASS["off"], "off_rush": self.RUSH["off"],
                           "def_pass": {t: -v for t, v in self.PASS["dfn"].items()},   # + = good defense
                           "def_rush": {t: -v for t, v in self.RUSH["dfn"].items()}})
        for c in list(df.columns):
            df[c + "_rk"] = df[c].rank(ascending=False).astype(int)
        return df

    def epa_pair(self, off, dfn, h, injuries=None):
        dp, dr = (injuries or {}).get((off, dfn), (0.0, 0.0))
        ep = self.PASS["icpt"] + self.PASS["off"][off] + self.PASS["dfn"][dfn] + self.PASS["home"] * h + dp
        er = self.RUSH["icpt"] + self.RUSH["off"][off] + self.RUSH["dfn"][dfn] + self.RUSH["home"] * h + dr
        return ep, er

    def n_plays(self, a, b):
        return (self.pace[a] + self.pace[b]) / 2

    def points(self, off, dfn, h, injuries=None):
        ep, er = self.epa_pair(off, dfn, h, injuries)
        pr = self.pr[off]
        return self.ppg + self.n_plays(off, dfn) * (pr * ep + (1 - pr) * er - self.lg_epa)

    def margin_total(self, home, away, neutral=False, injuries=None):
        h = 0.0 if neutral else 0.5
        hp, ap = self.points(home, away, h, injuries), self.points(away, home, -h, injuries)
        return hp - ap, hp + ap, hp, ap


# --------------------------- outcome distribution -------------------------------
class Outcomes:
    """Margin pmf = normal around the expected margin x key-number weights learned from history."""
    K = np.arange(-70, 71)

    def __init__(self, games):
        g = games.dropna(subset=["result", "spread_line", "total", "total_line"])
        self.sd_m = float((g.result - g.spread_line).std())
        self.sd_t = float((g.total - g.total_line).std())
        expected = np.zeros(len(self.K))
        for s in g.spread_line.values:
            expected += self._normal_pmf(s)
        observed = np.array([(g.result == k).sum() for k in self.K], float)
        self.weight = (observed + 5) / (expected + 5)    # +5 shrinks thin tails toward 1
        self.n_games = len(g)

    def _normal_pmf(self, m):
        return norm_cdf((self.K + 0.5 - m) / self.sd_m) - norm_cdf((self.K - 0.5 - m) / self.sd_m)

    def pmf(self, m, key_numbers=True):
        p = self._normal_pmf(m) * (self.weight if key_numbers else 1.0)
        return p / p.sum()

    def win_prob(self, m, key_numbers=True):
        p = self.pmf(m, key_numbers)
        return float(p[self.K > 0].sum() + 0.5 * p[self.K == 0].sum())

    def cover_prob(self, m, home_line):
        """P(home covers) when the market expects home margin `home_line` (nflverse spread_line)."""
        p = self.pmf(m)
        win, push = p[self.K > home_line].sum(), p[self.K == home_line].sum()
        return float(win), float(push)

    def over_prob(self, t, line):
        return float(1 - norm_cdf((line - t) / self.sd_t))


# ---------------------------------- levers --------------------------------------
def levers(r, out, home, away, m_blend, cfg):
    db = r.plays[r.plays["pass"] == 1]
    ns, sk = db[db.sack != 1], db[db.sack == 1]
    sack_pts = wavg(ns.epa, ns.w) - wavg(sk.epa, sk.w)
    to = r.plays[(r.plays.interception == 1) | (r.plays.fumble_lost == 1)]
    to_pts = r.lg_epa - wavg(to.epa, to.w)

    n = r.n_plays(home, away)
    ep_h, er_h = r.epa_pair(home, away, 0.5, cfg.injuries)
    ep_a, er_a = r.epa_pair(away, home, -0.5, cfg.injuries)
    tilt_home = cfg.tilt_decay * n * 0.10 * abs(ep_h - er_h)
    tilt_away = cfg.tilt_decay * n * 0.10 * abs(ep_a - er_a)
    home_dir = "pass-heavier" if ep_h > er_h else "run-heavier"
    away_dir = "run-heavier" if ep_a > er_a else "pass-heavier"   # push them off their better mode

    base = out.win_prob(m_blend)
    rows = [("Win the turnover battle by +1", to_pts),
            ("Avoid 1 sack", sack_pts),
            ("Get 1 more sack", sack_pts),
            (f"{home} offense {home_dir} by 10pp", tilt_home),
            (f"Push {away} {away_dir} by 10pp", tilt_away),
            ("Stack all of the above", to_pts + 2 * sack_pts + tilt_home + tilt_away)]
    res = []
    for name, pts in rows:
        wp = out.win_prob(m_blend + pts)
        res.append({"lever": name, "margin_pts": round(pts, 2), "wp": round(wp, 4),
                    "wp_delta_pp": round((wp - base) * 100, 1)})
    return res, {"turnover_pts": to_pts, "sack_pts": sack_pts}


# ----------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--game-id", default="2026_06_SEA_DEN")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model-weight", type=float, default=Config.model_weight)
    a = ap.parse_args()

    # Per-starter effects from injury_test.py (2024-26, 1,054 team-games):
    #   starting DB out: +0.008 pass / +0.012 rush EPA allowed; starting LB out: +0.043 / +0.023
    #   RB1 out: no measurable effect.
    # DEN: Surtain II + Hufanga out, Moss questionable (questionable starters sit ~32%) -> 2.32 DBs
    # SEA: LB Drake Thomas out (full-time starter). Charbonnet hasn't played in 2026, so his
    #      absence is already in SEA's ratings; RB1 by snaps (Wilson) is active.
    den_db_out, sea_lb_out = 2.32, 1
    cfg = Config(model_weight=a.model_weight,
                 injuries={("SEA", "DEN"): (0.008 * den_db_out, 0.012 * den_db_out),
                           ("DEN", "SEA"): (0.043 * sea_lb_out, 0.023 * sea_lb_out)})

    data, outd = Path(a.data_dir), Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(data / "games.csv")
    gm = games[games.game_id == a.game_id].iloc[0]
    season, week, home, away = int(gm.season), int(gm.week), gm.home_team, gm.away_team
    neutral = gm.location == "Neutral"
    line, tot_line = float(gm.spread_line), float(gm.total_line)

    pbp = {s: load_pbp(p) for s in (season - 1, season)
           if (p := data / f"pbp_{s}.csv.gz").exists()}
    r = Ratings(training_frame(pbp, season, week, cfg.w_prior), cfg.alpha)
    out = Outcomes(games[(games.season < season) & (games.season >= season - 10) & (games.game_type == "REG")])

    m, t, hp, ap_ = r.margin_total(home, away, neutral, cfg.injuries)
    m0, t0, _, _ = r.margin_total(home, away, neutral)
    m_blend = cfg.model_weight * m + (1 - cfg.model_weight) * line
    t_blend = cfg.total_weight * t + (1 - cfg.total_weight) * tot_line
    wp = out.win_prob(m_blend)
    cover, push = out.cover_prob(m_blend, line)
    lev, unit = levers(r, out, home, away, m_blend, cfg)

    print(f"{away} @ {home}  {gm.gameday}  market: {home} {-line:+.1f} (expected {home} margin {line:+.1f}), total {tot_line}")
    print(f"Ratings from {r.PASS['n']} pass / {r.RUSH['n']} rush plays | league {r.ppg:.1f} PPG | "
          f"pace {home} {r.pace[home]:.1f} {away} {r.pace[away]:.1f}")
    print(f"Home-field (EPA/play, home minus away): pass {r.PASS['home']:+.3f} rush {r.RUSH['home']:+.3f}")
    print(f"Outcome model: margin SD {out.sd_m:.1f}, total SD {out.sd_t:.1f} ({out.n_games} past games)\n")
    print("Opponent-adjusted ratings (EPA/play vs avg, + = good; rank of 32):")
    print(r.table().loc[[home, away]].round(3).to_string())
    print(f"\nModel:   {home} {hp:.1f} - {away} {ap_:.1f}  margin {m:+.1f}  total {t:.1f}  "
          f"(no injury adj: margin {m0:+.1f})")
    print(f"Blend ({cfg.model_weight:.0%} model): margin {m_blend:+.1f}  total {t_blend:.1f}  "
          f"=> {home} {(t_blend + m_blend) / 2:.1f}, {away} {(t_blend - m_blend) / 2:.1f}")
    print(f"\n{home} win prob: blend {wp:.1%} | model only {out.win_prob(m):.1%} | market only {out.win_prob(line):.1%}"
          f" | blend w/o key numbers {out.win_prob(m_blend, key_numbers=False):.1%}")
    print(f"{home} {-line:+.1f} covers {cover:.1%} (push {push:.1%}) | Over {tot_line} {out.over_prob(t_blend, tot_line):.1%}")
    print(f"P(margin exactly {int(math.floor(line))}): {out.pmf(m_blend)[out.K == math.floor(line)][0]:.1%} with key numbers, "
          f"{out.pmf(m_blend, False)[out.K == math.floor(line)][0]:.1%} without")
    print(f"\nLever values: turnover {unit['turnover_pts']:.2f} pts, sack {unit['sack_pts']:.2f} pts "
          f"(play-mix tilts shrunk by {cfg.tilt_decay})")
    for L in lev:
        print(f"  {L['lever']:<36} {L['margin_pts']:+5.2f} pts  WP {L['wp']:.1%} ({L['wp_delta_pp']:+.1f}pp)")

    r.table().round(4).to_csv(outd / "team_ratings.csv")
    with open(outd / "prediction.json", "w") as f:
        json.dump({"game_id": a.game_id, "model_weight": cfg.model_weight, "margin_model": m,
                   "total_model": t, "margin_blend": m_blend, "total_blend": t_blend,
                   "home_wp": wp, "home_cover": cover, "push": push,
                   "over": out.over_prob(t_blend, tot_line), "levers": lev}, f, indent=2)


if __name__ == "__main__":
    main()
