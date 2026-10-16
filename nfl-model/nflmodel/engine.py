"""Calibrated predictions: ratings -> raw scores -> calibrated margin/total -> outcome distribution."""
import math

import numpy as np

from .injuries import offense_adj

K = np.arange(-60, 61)
LEVERS = ["takeaways", "giveaways", "sacks_made", "sacks_taken", "expl_for", "expl_against"]
GOOD_HIGH = {"takeaways": True, "giveaways": False, "sacks_made": True, "sacks_taken": False,
             "expl_for": True, "expl_against": False}
MIRROR = {"takeaways": "giveaways", "giveaways": "takeaways", "sacks_made": "sacks_taken",
          "sacks_taken": "sacks_made", "expl_for": "expl_against", "expl_against": "expl_for"}
# Upside scenarios from one team's point of view; unlisted levers stay at league average.
SCENARIOS = {
    "Clean offense": {"giveaways": 0, "sacks_taken": 1},
    "Win turnovers by +2": {"takeaways": 2, "giveaways": 0},
    "Defense takes over": {"takeaways": 2, "giveaways": 0, "sacks_made": 4, "sacks_taken": 1},
    "Explosive night": {"giveaways": 0, "expl_for": 10, "expl_against": 5},
    "Great night": {"takeaways": 2, "giveaways": 0, "sacks_made": 3, "sacks_taken": 2, "expl_for": 7, "expl_against": 5},
}

_erf = np.vectorize(math.erf)


def norm_cdf(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2.0)))


def normal_pmf(m, sd):
    return norm_cdf((K + 0.5 - m) / sd) - norm_cdf((K - 0.5 - m) / sd)


def key_weights(centers, results, sd):
    """How much more (or less) often each final margin happens than a smooth normal says."""
    expected = np.zeros(len(K))
    for c in centers:
        expected += normal_pmf(c, sd)
    observed = np.array([(np.asarray(results) == k).sum() for k in K], float)
    return (observed + 5) / (expected + 5)      # +5 shrinks thin tails toward 1


class Model:
    def __init__(self, ratings, params):
        self.r, self.p = ratings, params
        self.w = np.array(params["key_weights"])

    def pmf(self, m, sd=None):
        p = normal_pmf(m, sd or self.p["sd_margin"]) * self.w
        return p / p.sum()

    def win_prob(self, m, sd=None):
        p = self.pmf(m, sd)
        return float(p[K > 0].sum() + 0.5 * p[K == 0].sum())

    def calibrate(self, raw_home, raw_away, neutral=False):
        c = self.p["calibration"]
        m = c["m_home"] * (0 if neutral else 1) + c["m_slope"] * (raw_home - raw_away)
        t = c["t_icpt"] + c["t_slope"] * (raw_home + raw_away)
        return m, t

    def predict(self, home, away, neutral=False, home_missing=None, away_missing=None):
        home_missing, away_missing = home_missing or {}, away_missing or {}
        rh, ra = self.r.raw_scores(home, away, neutral, offense_adj(away_missing), offense_adj(home_missing))
        m, t = self.calibrate(rh, ra, neutral)
        p = self.pmf(m)
        cdf = np.cumsum(p)
        q = {f"p{int(x * 100)}": int(K[np.searchsorted(cdf, x)]) for x in (0.1, 0.25, 0.5, 0.75, 0.9)}
        return {"home": home, "away": away, "neutral": bool(neutral),
                "home_pts": (t + m) / 2, "away_pts": (t - m) / 2, "margin": m, "total": t,
                "home_wp": self.win_prob(m), "margin_q": q,
                "home_by_7": float(p[K >= 7].sum()), "away_by_7": float(p[K <= -7].sum())}
