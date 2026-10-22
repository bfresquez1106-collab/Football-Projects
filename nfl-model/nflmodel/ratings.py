"""Opponent-adjusted EPA ratings (ridge regression) and the matchup -> points conversion."""
import numpy as np
import pandas as pd

W_PRIOR = 0.20   # weight on prior-season plays late in the season
W_EARLY = 0.75   # ...and in Week 1, fading to W_PRIOR over FADE_WEEKS (replay: better early, same late)
FADE_WEEKS = 8
ALPHA = 250.0    # ridge shrinkage, in "plays"


def wavg(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    return float((x[m] * w[m]).sum() / w[m].sum())


def prior_weight(before_week, w_late=W_PRIOR, w_early=W_EARLY, fade_weeks=FADE_WEEKS):
    """
    Weight on last season's plays. With w_early set, it starts at w_early in Week 1 and fades
    linearly to w_late over `fade_weeks` weeks: last season matters most when this season has
    barely started.
    """
    if w_early is None:
        return w_late
    return w_late + (w_early - w_late) * max(0.0, 1 - (before_week - 1) / fade_weeks)


def training_frame(pbp_by_season, season, before_week, w_prior=prior_weight):
    """Plays known before `season` week `before_week`: that season (weight 1) + prior season (w_prior).
    w_prior may be a number or a function of before_week."""
    if callable(w_prior):
        w_prior = w_prior(before_week)
    parts = []
    if season - 1 in pbp_by_season:
        parts.append(pbp_by_season[season - 1].assign(w=w_prior))
    if season in pbp_by_season:
        cur = pbp_by_season[season]
        parts.append(cur[cur.week < before_week].assign(w=1.0))
    return pd.concat(parts, ignore_index=True)


def luck_adjust(plays):
    """
    Take fumble-recovery luck out of EPA. Who recovers a fumble is a coin flip (a team's recovery
    rate in odd weeks says nothing about even weeks), but fumbling is partly skill. So each fumble
    keeps its cost at the AVERAGE recovery outcome instead of the actual bounce:
        epa_adj = epa - (lost - league share lost) * (EPA gap between a lost and a kept fumble)
    Rates and gaps come from the plays passed in (pre-kickoff data only), by run / pass / sack.
    """
    p = plays.copy()
    f = p.fumble == 1
    kind = np.where(p.sack == 1, "sack", np.where(p["pass"] == 1, "pass", "run"))
    for k in ("run", "pass", "sack"):
        m = f & (kind == k)
        lost = p.fumble_lost[m] == 1
        if lost.all() or not lost.any():
            continue
        share, gap = lost.mean(), p.epa[m][lost].mean() - p.epa[m][~lost].mean()
        p.loc[m, "epa"] = p.epa[m] - (p.fumble_lost[m] - share) * gap
    return p


class Ratings:
    def __init__(self, raw, alpha=ALPHA, luck=False):
        plays = raw[((raw["pass"] == 1) | (raw["rush"] == 1)) & (raw.qb_kneel != 1)
                    & (raw.qb_spike != 1) & raw.epa.notna()].copy()
        if luck:
            plays = luck_adjust(plays)
        # +0.5 offense at home, -0.5 away, 0 at a neutral site
        plays["h"] = np.where(plays.location == "Neutral", 0.0,
                              np.where(plays.posteam == plays.home_team, 0.5, -0.5))
        clean = plays[(plays.wp > 0.05) & (plays.wp < 0.95)]
        self.plays, self.clean = plays, clean
        self.teams = sorted(set(clean.posteam) | set(clean.defteam))
        self.tix = {t: i for i, t in enumerate(self.teams)}
        self.PASS = self._fit(clean[clean["pass"] == 1], alpha)
        self.RUSH = self._fit(clean[clean["rush"] == 1], alpha)

        g = raw.groupby("game_id").agg(hs=("home_score", "max"), as_=("away_score", "max"), w=("w", "first"))
        self.ppg = wavg((g.hs + g.as_) / 2, g.w)
        gp = plays.groupby("game_id").size() / 2
        tg = plays.groupby(["game_id", "posteam"]).w.first().reset_index()
        tg["gp"] = tg.game_id.map(gp)
        self.pace = {t: wavg(d.gp, d.w) for t, d in tg.groupby("posteam")}
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

    def epa_pair(self, off, dfn, h, adj=(0.0, 0.0)):
        ep = self.PASS["icpt"] + self.PASS["off"][off] + self.PASS["dfn"][dfn] + self.PASS["home"] * h + adj[0]
        er = self.RUSH["icpt"] + self.RUSH["off"][off] + self.RUSH["dfn"][dfn] + self.RUSH["home"] * h + adj[1]
        return ep, er

    def points(self, off, dfn, h, adj=(0.0, 0.0)):
        ep, er = self.epa_pair(off, dfn, h, adj)
        pr = self.pr[off]
        n = (self.pace[off] + self.pace[dfn]) / 2
        return self.ppg + n * (pr * ep + (1 - pr) * er - self.lg_epa)

    def raw_scores(self, home, away, neutral=False, home_adj=(0.0, 0.0), away_adj=(0.0, 0.0)):
        """Uncalibrated (home points, away points). *_adj = EPA added to that team's OFFENSE."""
        h = 0.0 if neutral else 0.5
        return self.points(home, away, h, home_adj), self.points(away, home, -h, away_adj)

    def export(self):
        """Everything the dashboard needs to recompute a matchup in the browser."""
        return {"ppg": self.ppg, "lg_epa": self.lg_epa,
                "pass": {"icpt": self.PASS["icpt"], "home": self.PASS["home"]},
                "rush": {"icpt": self.RUSH["icpt"], "home": self.RUSH["home"]},
                "teams": {t: {"off_pass": self.PASS["off"][t], "def_pass": self.PASS["dfn"][t],
                              "off_rush": self.RUSH["off"][t], "def_rush": self.RUSH["dfn"][t],
                              "pace": self.pace[t], "pass_rate": self.pr[t]} for t in self.teams}}
