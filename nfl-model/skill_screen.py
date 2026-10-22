#!/usr/bin/env python3
"""
Skill screen: which team stats are skill (worth adding to the model) and which are luck?

Every stat goes through three gates, on 2016-2025 team-seasons:

  1. Repeatable   odd weeks vs even weeks. Luck doesn't repeat; skill does.
                  (half-season correlation, then stepped up to a full season)
  2. Predictive   Weeks 1-9 value vs points per game in Weeks 10+ (scored for offense stats,
                  allowed for defense stats; defense stats are what the defense allowed/made). A stat can repeat and still not matter for points.
  3. New info     the same, after taking out what Weeks 1-9 EPA/play already says (partial
                  correlation). The model already has EPA; a stat only helps if it adds to it.

Run:  python3 skill_screen.py            (writes outputs/skill_screen.csv)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["season", "season_type", "week", "game_id", "home_team", "away_team", "posteam", "defteam",
        "pass", "rush", "qb_kneel", "qb_spike", "qb_dropback", "epa", "wp", "success", "down",
        "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
        "yards_gained", "sack", "qb_hit", "interception", "fumble", "fumble_lost", "pass_attempt",
        "cpoe", "yards_after_catch", "xyac_mean_yardage", "complete_pass", "pass_oe", "penalty",
        "penalty_team", "penalty_yards", "fixed_drive", "fixed_drive_result", "drive_inside20",
        "yardline_100", "field_goal_attempt", "field_goal_result", "kick_distance",
        "home_score", "away_score"]
# Stat name -> (higher is better for the team?, description). Each is computed for offense (o_) and defense (d_).
STATS = {
    "epa_play": "EPA per play (the model's core input)",
    "success": "success rate",
    "pass_epa": "EPA per dropback",
    "rush_epa": "EPA per designed run",
    "early_epa": "EPA per play on 1st and 2nd down",
    "late_epa": "EPA per play on 3rd and 4th down",
    "third_conv": "3rd-down conversion rate",
    "rz_td": "touchdown rate on drives reaching the 20",
    "explosive": "explosive play rate (run 10+, pass 20+)",
    "stuff": "runs stopped at or behind the line",
    "sack_rate": "sacks per dropback",
    "hit_rate": "QB hits (incl. sacks) per dropback",
    "int_rate": "interceptions per pass attempt",
    "fumble_rate": "fumbles per play",
    "takeaway_rate": "turnovers (INT + fumble lost) per play",
    "cpoe": "completion % over expected",
    "yac_oe": "yards after catch over expected",
    "pass_oe": "pass rate over expected (play-calling style)",
    "pen_yds": "penalty yards per game (charged to this side)",
    "start_fp": "average drive start, yards from own goal",
    "pts_drive": "points per drive",
    "fg_pct": "field-goal make rate (offense only)",
}


def load(data_dir, seasons):
    out = []
    for s in seasons:
        d = pd.read_csv(Path(data_dir) / f"pbp_{s}.csv.gz", usecols=lambda c: c in COLS, low_memory=False)
        out.append(d[(d.season_type == "REG") & d.posteam.notna() & d.defteam.notna()])
    return pd.concat(out, ignore_index=True)


def side_stats(p, side):
    """Per game and team, the sums needed for each stat, from the offense's or the defense's view."""
    key = "posteam" if side == "o" else "defteam"
    scrim = p[((p["pass"] == 1) | (p.rush == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1) & p.epa.notna()].copy()
    clean = scrim[scrim.wp.between(0.05, 0.95)].copy()
    run = (clean.rush == 1) & (clean.qb_dropback != 1)
    drop = clean.qb_dropback == 1
    clean = clean.assign(
        n=1, run=run, drop=drop, rush_e=clean.epa.where(run, 0), pass_e=clean.epa.where(drop, 0),
        early=clean.down.isin([1, 2]), late=clean.down.isin([3, 4]),
        early_e=clean.epa.where(clean.down.isin([1, 2]), 0), late_e=clean.epa.where(clean.down.isin([3, 4]), 0),
        expl=(((clean["pass"] == 1) & (clean.sack != 1) & (clean.yards_gained >= 20))
              | ((clean.rush == 1) & (clean.yards_gained >= 10))),
        stuffed=run & (clean.yards_gained <= 0), hit=drop & ((clean.qb_hit == 1) | (clean.sack == 1)),
        sk=drop & (clean.sack == 1), att=clean.pass_attempt == 1,
        cp=clean.cpoe.fillna(0), cp_n=clean.cpoe.notna(),
        yac=(clean.yards_after_catch - clean.xyac_mean_yardage).fillna(0),
        yac_n=(clean.complete_pass == 1) & clean.xyac_mean_yardage.notna() & clean.yards_after_catch.notna(),
        poe=clean.pass_oe.fillna(0), poe_n=clean.pass_oe.notna())
    clean["int_"] = clean.interception == 1
    clean["fum"] = clean.fumble == 1
    clean["take"] = (clean.interception == 1) | (clean.fumble_lost == 1)
    clean["t3c"], clean["t3n"] = clean.third_down_converted, clean.third_down_converted + clean.third_down_failed
    g = clean.groupby(["season", "week", "game_id", key]).agg(
        n=("n", "sum"), epa=("epa", "sum"), succ=("success", "sum"), run=("run", "sum"), rush_e=("rush_e", "sum"),
        drop=("drop", "sum"), pass_e=("pass_e", "sum"), early=("early", "sum"), early_e=("early_e", "sum"),
        late=("late", "sum"), late_e=("late_e", "sum"), expl=("expl", "sum"), stuffed=("stuffed", "sum"),
        hit=("hit", "sum"), sk=("sk", "sum"), att=("att", "sum"), int_=("int_", "sum"), fum=("fum", "sum"),
        take=("take", "sum"), t3c=("t3c", "sum"), t3n=("t3n", "sum"), cp=("cp", "sum"), cp_n=("cp_n", "sum"),
        yac=("yac", "sum"), yac_n=("yac_n", "sum"), poe=("poe", "sum"), poe_n=("poe_n", "sum"))

    # drives: start field position, points, red-zone touchdowns (all plays, not just competitive ones)
    dr = p[p.fixed_drive.notna()].groupby(["season", "week", "game_id", "posteam", "fixed_drive"]).agg(
        res=("fixed_drive_result", "first"), start=("yardline_100", "first"), rz=("drive_inside20", "max"),
        defteam=("defteam", "first")).reset_index()
    dr = dr[dr.start.notna()]
    dr["pts"] = dr.res.map({"Touchdown": 7, "Field goal": 3, "Opp touchdown": -7, "Safety": -2}).fillna(0)
    dr["fp"] = 100 - dr.start
    dr["rz_td"] = (dr.rz == 1) & (dr.res == "Touchdown")
    dg = dr.groupby(["season", "week", "game_id", key]).agg(drives=("pts", "size"), pts=("pts", "sum"), fp=("fp", "sum"),
                                                          rz=("rz", "sum"), rz_td=("rz_td", "sum"))
    pen = p[(p.penalty == 1) & p.penalty_team.notna()]
    pen = pen[pen.penalty_team == pen[key]].groupby(["season", "week", "game_id", key]).penalty_yards.sum().rename("pen")
    fg = p[p.field_goal_attempt == 1].assign(made=lambda x: x.field_goal_result == "made")
    fg = fg.groupby(["season", "week", "game_id", key]).agg(fga=("made", "size"), fgm=("made", "sum"))
    out = g.join(dg).join(pen).join(fg).fillna(0).reset_index().rename(columns={key: "team"})
    out["games"] = 1
    return out


def rates(s):
    """Turn summed counts (any grouping) into the stats."""
    q = lambda a, b: a / b.where(b > 0)
    return pd.DataFrame({
        "epa_play": q(s.epa, s.n), "success": q(s.succ, s.n), "pass_epa": q(s.pass_e, s["drop"]),
        "rush_epa": q(s.rush_e, s.run), "early_epa": q(s.early_e, s.early), "late_epa": q(s.late_e, s.late),
        "third_conv": q(s.t3c, s.t3n), "rz_td": q(s.rz_td, s.rz), "explosive": q(s.expl, s.n),
        "stuff": q(s.stuffed, s.run), "sack_rate": q(s.sk, s["drop"]), "hit_rate": q(s.hit, s["drop"]),
        "int_rate": q(s.int_, s.att), "fumble_rate": q(s.fum, s.n), "takeaway_rate": q(s["take"], s.n),
        "cpoe": q(s.cp, s.cp_n), "yac_oe": q(s.yac, s.yac_n), "pass_oe": q(s.poe, s.poe_n),
        "pen_yds": q(s.pen, s.games), "start_fp": q(s.fp, s.drives), "pts_drive": q(s.pts, s.drives),
        "fg_pct": q(s.fgm, s.fga)})


def partial_corr(x, y, z):
    """Correlation of x and y after removing what z explains in each."""
    m = x.notna() & y.notna() & z.notna()
    x, y, z = x[m], y[m], z[m]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--last", type=int, default=2025)
    a = ap.parse_args()
    p = load(a.data_dir, range(a.first, a.last + 1))

    # points per game for each team, scored and allowed
    gm = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first"), home=("home_team", "first"),
                                  away=("away_team", "first"), hs=("home_score", "max"), as_=("away_score", "max")).reset_index()
    pts = pd.concat([gm.rename(columns={"home": "team", "hs": "pf", "as_": "pa"})[["season", "week", "team", "pf", "pa"]],
                     gm.rename(columns={"away": "team", "as_": "pf", "hs": "pa"})[["season", "week", "team", "pf", "pa"]]])
    late_pts = pts[pts.week >= 10].groupby(["season", "team"])[["pf", "pa"]].mean()

    rows = []
    for side, target in (("o", "pf"), ("d", "pa")):
        g = side_stats(p, side)
        sums = g.columns.difference(["season", "week", "game_id", "team"])
        odd = rates(g[g.week % 2 == 1].groupby(["season", "team"])[sums].sum())
        even = rates(g[g.week % 2 == 0].groupby(["season", "team"])[sums].sum())
        early = rates(g[g.week <= 9].groupby(["season", "team"])[sums].sum()).join(late_pts)
        future = early[target]
        for stat, desc in STATS.items():
            if side == "d" and stat == "fg_pct":
                continue
            r_half = float(odd[stat].corr(even[stat]))
            r_full = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan   # Spearman-Brown: full season
            x = early[stat]
            pred = float(x.corr(future))
            new = np.nan if stat == "epa_play" else partial_corr(x, future, early["epa_play"])
            rows.append({"side": "offense" if side == "o" else "defense", "stat": stat, "what": desc,
                         "repeat_half": r_half, "repeat_season": r_full, "predicts": pred, "beyond_epa": new})
    res = pd.DataFrame(rows)
    n = early.shape[0]
    se = 1 / np.sqrt(n - 3)
    Path("outputs").mkdir(exist_ok=True)
    res.to_csv("outputs/skill_screen.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"{n} team-seasons, {a.first}-{a.last}. "
          f"'predicts' and 'beyond_epa' correlate with points SCORED (offense) or ALLOWED (defense). "
          f"A correlation needs about ±{2 * se:.2f} to stand out from chance.\n")
    for side, d in res.groupby("side", sort=False):
        print(side.upper())
        print(d.drop(columns="side").set_index("stat").round(2).to_string(), "\n")


if __name__ == "__main__":
    main()
