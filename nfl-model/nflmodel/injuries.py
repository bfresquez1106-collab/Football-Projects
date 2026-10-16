"""
Automatic injury adjustments from the weekly injury report.

Effects come from broncos-seahawks-2026/injury_test.py (2024-26, 1,054 team-games): each starting
DB or LB a defense is missing adds EPA/play to the OPPONENT's offense. Missing RB1s showed no
measurable effect, so they're left out. Play rates by report status were measured on 2023-25
injury reports against snap counts.
"""
import re

import pandas as pd

EFFECTS = {"DB": (0.008, 0.012), "LB": (0.043, 0.023)}     # (pass, rush) EPA/play per missing starter
PLAY_RATE = {"Out": 0.0, "Doubtful": 0.09, "Questionable": 0.60}
GROUPS = {"DB": (["CB", "S", "FS", "SS", "DB"], 5), "LB": (["LB", "ILB", "MLB", "OLB"], 2)}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def last_key(name):
    words = [w for w in re.sub(r"[^a-z ]", "", str(name).lower()).split() if w not in SUFFIXES]
    return words[-1] if words else ""


def starters(snaps_cur, snaps_prior, week, team, group):
    """Top defenders by season-to-date snap share (>= 50%, >= 2 games); prior season early on."""
    pos, k = GROUPS[group]
    for s, wk in ((snaps_cur, week), (snaps_prior, 99)):
        if s is None or s.empty:
            continue
        d = s[(s.week < wk) & (s.team == team) & s.position.isin(pos)]
        avg = d.groupby("player").defense_pct.agg(["mean", "size"])
        avg = avg[(avg["size"] >= 2) & (avg["mean"] >= 0.5)].sort_values("mean", ascending=False)
        if len(avg):
            return list(avg.index[:k])
    return []


def team_missing(snaps_cur, snaps_prior, report, week, team):
    """Expected number of starting DBs / LBs out this week, and who."""
    rep = report[(report.week == week) & (report.team == team) & report.report_status.isin(PLAY_RATE)] \
        if not report.empty else report
    out = {"DB": 0.0, "LB": 0.0, "players": []}
    for group, (pos, _) in GROUPS.items():
        names = starters(snaps_cur, snaps_prior, week, team, group)
        if rep.empty:
            continue
        r = rep[rep.position.isin(pos)].assign(key=lambda d: d.full_name.map(last_key))
        for n in names:
            hit = r[r.key == last_key(n)]
            if len(hit):
                status = hit.report_status.iloc[0]
                out[group] += 1 - PLAY_RATE[status]
                out["players"].append({"name": n, "group": group, "status": status})
    return out


def offense_adj(opp_missing):
    """EPA/play added to an offense because the defense it faces is missing starters."""
    p = sum(EFFECTS[g][0] * opp_missing.get(g, 0) for g in EFFECTS)
    r = sum(EFFECTS[g][1] * opp_missing.get(g, 0) for g in EFFECTS)
    return p, r
