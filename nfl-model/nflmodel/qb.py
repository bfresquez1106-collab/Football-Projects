"""
QB ratings: a regressed composite of per-dropback stats, built only from games before the one
being predicted.

  1. Per-dropback stats for each QB: EPA, success rate, CPOE, sack rate, interception rate.
  2. Recency: this season counts fully, each earlier season by DECAY (one season back = 0.6...).
  3. Shrink: stat = (n * raw + k * replacement) / (n + k). k is how many dropbacks a stat needs
     before it's half signal, half noise (measured in qb_index.py). The pull is toward
     REPLACEMENT level, because a QB we know little about is usually a backup.
  4. Standardize each stat to a z-score; its index is 100 + 10z.
  5. Composite = regression weights x z-scores, in expected EPA per dropback going forward.
     Only EPA and success rate carry weight: CPOE, sack rate and interception rate overlap with
     them and added nothing out of sample, so they're shown as indexes but left out of the rating.
"""
import numpy as np
import pandas as pd

QB_COLS = ["season", "season_type", "week", "game_id", "posteam", "passer_id", "passer", "qb_dropback",
           "qb_epa", "success", "sack", "interception", "cpoe", "pass_attempt", "wp", "qb_spike", "qb_kneel"]
STATS = ["epa", "success", "cpoe", "sack_rate", "int_rate"]     # all get a 100-scale index
COMPOSITE = ["epa", "success"]   # the rating itself: out of sample, adding the other three didn't help
GOOD_HIGH = {"epa": True, "success": True, "cpoe": True, "sack_rate": False, "int_rate": False}
DECAY = 0.6        # weight per season back
SEASONS_BACK = 3


def load_dropbacks(pbp_path):
    d = pd.read_csv(pbp_path, usecols=lambda c: c in QB_COLS, low_memory=False)
    d = d[(d.season_type == "REG") & (d.qb_dropback == 1) & d.passer_id.notna()
          & (d.qb_spike != 1) & d.wp.between(0.05, 0.95)]
    return d.fillna({"sack": 0, "interception": 0, "pass_attempt": 0, "success": 0, "qb_epa": 0})


def weekly_sums(db):
    """One row per QB per week: the raw sums every stat is built from."""
    db = db.assign(att=db.pass_attempt * (1 - db.sack), cp_n=db.cpoe.notna().astype(int), cp=db.cpoe.fillna(0))
    return db.groupby(["passer_id", "season", "week"]).agg(
        name=("passer", "last"), team=("posteam", "last"), n=("qb_dropback", "size"), epa=("qb_epa", "sum"),
        success=("success", "sum"), sacks=("sack", "sum"), ints=("interception", "sum"), att=("att", "sum"),
        cp=("cp", "sum"), cp_n=("cp_n", "sum")).reset_index()


def raw_as_of(sums, season, before_week, decay=DECAY):
    """Recency-weighted per-QB stats using only games before `season` week `before_week`."""
    s = sums[((sums.season == season) & (sums.week < before_week))
             | ((sums.season < season) & (sums.season >= season - SEASONS_BACK))].copy()
    w = decay ** (season - s.season)
    agg = {c: (s[c] * w).groupby(s.passer_id).sum() for c in ["n", "epa", "success", "sacks", "ints", "att", "cp", "cp_n"]}
    a = pd.DataFrame(agg)
    a["name"] = s.sort_values(["season", "week"]).groupby("passer_id").name.last()
    a["team"] = s.sort_values(["season", "week"]).groupby("passer_id").team.last()
    out = pd.DataFrame({"name": a.name, "team": a.team, "n": a.n,
                        "epa": a.epa / a.n, "success": a.success / a.n, "sack_rate": a.sacks / a.n,
                        "int_rate": a.ints / a.att.clip(lower=1), "cpoe": a.cp / a.cp_n.clip(lower=1),
                        "n_att": a.att, "n_cp": a.cp_n})
    return out


def shrink(raw, p):
    """Pull each stat toward replacement level by its own noise constant k."""
    out = raw.copy()
    for st in STATS:
        n = raw.n_att if st == "int_rate" else raw.n_cp if st == "cpoe" else raw.n
        out[st] = (n * raw[st].fillna(0) + p["k"][st] * p["replacement"][st]) / (n + p["k"][st])
    return out


def zscores(shrunk, p):
    return pd.DataFrame({st: (shrunk[st] - p["mean"][st]) / p["sd"][st] for st in STATS}, index=shrunk.index)


def composite(z, p):
    """Expected EPA per dropback going forward."""
    return p["intercept"] + sum(w * z[st] for st, w in p["weights"].items())


def ratings_as_of(sums, season, before_week, p):
    """Every QB's rating before a given week, plus each stat's 100-scale index."""
    raw = raw_as_of(sums, season, before_week, p.get("decay", DECAY))
    sh = shrink(raw, p)
    z = zscores(sh, p)
    out = sh[["name", "team", "n"]].copy()
    out["rating"] = composite(z, p)
    for st in STATS:
        sign = 1 if GOOD_HIGH[st] else -1
        out[f"{st}_index"] = 100 + 10 * sign * z[st]
    out["qb_index"] = 100 + 10 * (out.rating - p["starter_mean"]) / p["starter_sd"]
    return out


def team_starters(db):
    """Who started each game: the QB with the most dropbacks for that team (known before kickoff in practice)."""
    c = db.groupby(["game_id", "posteam", "passer_id"]).size().reset_index(name="n")
    return c.sort_values("n").groupby(["game_id", "posteam"]).tail(1).set_index(["game_id", "posteam"]).passer_id


def replacement_rating(p):
    """Rating of a QB with no history: replacement level on every stat."""
    return p["intercept"] + sum(w * (p["replacement"][st] - p["mean"][st]) / p["sd"][st] for st, w in p["weights"].items())


def live_starters(depth, report, week):
    """Depth-chart QB1, unless he's Out or Doubtful on this week's injury report; then the next QB down."""
    out = set()
    if len(report):
        r = report[(report.week == week) & report.report_status.isin(["Out", "Doubtful"])]
        out = set(r.gsis_id)
    starters = {}
    for team, g in depth.groupby("team"):
        ok = g[~g.gsis_id.isin(out)]
        pick = (ok if len(ok) else g).iloc[0]
        first = g.iloc[0]
        starters[team] = {"id": pick.gsis_id, "name": pick["name"],
                          "replaces": first["name"] if pick.gsis_id != first.gsis_id else None}
    return starters


def team_baseline(db_window, ratings, fill):
    """
    The QB rating a team's current numbers already contain: each QB's rating weighted by how many
    (recency-weighted) dropbacks he contributed to the team's ratings window.
    """
    c = db_window.groupby(["posteam", "passer_id"]).w.sum().reset_index(name="wn")
    c["rating"] = c.passer_id.map(ratings).fillna(fill)
    return (c.rating * c.wn).groupby(c.posteam).sum() / c.wn.groupby(c.posteam).sum()
