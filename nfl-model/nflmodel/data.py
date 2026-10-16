"""Download and load nflverse data. Betting-line columns are dropped at load: this model never sees them."""
import subprocess
from pathlib import Path

import pandas as pd

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
LINE_COLS = ["spread_line", "total_line", "away_moneyline", "home_moneyline", "away_spread_odds",
             "home_spread_odds", "under_odds", "over_odds"]
PBP_COLS = ["season", "season_type", "week", "game_id", "home_team", "away_team", "location",
            "posteam", "defteam", "pass", "rush", "epa", "wp", "qb_kneel", "qb_spike", "sack",
            "interception", "fumble_lost", "yards_gained", "success", "down", "home_score", "away_score"]
FILES = {"pbp": "pbp/play_by_play_{s}.csv.gz", "snaps": "snap_counts/snap_counts_{s}.csv",
         "injuries": "injuries/injuries_{s}.csv"}
LOCAL = {"pbp": "pbp_{s}.csv.gz", "snaps": "snap_counts_{s}.csv", "injuries": "injuries_{s}.csv"}


def _get(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", str(tmp), url], check=True)
    tmp.replace(dest)


def fetch(data_dir, seasons, current):
    """Download what's missing; always refresh the schedule and the current season's files."""
    data_dir = Path(data_dir)
    if data_dir.is_symlink() and not data_dir.exists():     # link to a scratch dir that's gone
        data_dir.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    _get(f"{RELEASES}/schedules/games.csv", data_dir / "games.csv")
    for s in seasons:
        for kind, remote in FILES.items():
            dest = data_dir / LOCAL[kind].format(s=s)
            if s == current or not dest.exists():
                _get(f"{RELEASES}/{remote.format(s=s)}", dest)


def games(data_dir):
    g = pd.read_csv(Path(data_dir) / "games.csv")
    return g.drop(columns=[c for c in LINE_COLS if c in g.columns])


def pbp(data_dir, season):
    df = pd.read_csv(Path(data_dir) / LOCAL["pbp"].format(s=season), usecols=lambda c: c in PBP_COLS,
                     low_memory=False)
    df = df[(df.season_type == "REG") & df.posteam.notna() & df.defteam.notna()].copy()
    for c in ["pass", "rush", "qb_kneel", "qb_spike", "sack", "interception", "fumble_lost", "success", "yards_gained"]:
        df[c] = df[c].fillna(0)
    return df


def snaps(data_dir, season):
    s = pd.read_csv(Path(data_dir) / LOCAL["snaps"].format(s=season))
    return s[s.game_type == "REG"]


def injuries(data_dir, season):
    p = Path(data_dir) / LOCAL["injuries"].format(s=season)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def team_game_levers(plays):
    """Per game and team: giveaways, sacks taken, explosive plays (rush 10+, pass 20+), plus the mirror."""
    p = plays[((plays["pass"] == 1) | (plays["rush"] == 1)) & (plays.qb_kneel != 1) & (plays.qb_spike != 1)].copy()
    p["give"] = p.interception + p.fumble_lost
    p["expl"] = (((p["pass"] == 1) & (p.sack != 1) & (p.yards_gained >= 20))
                 | ((p["rush"] == 1) & (p.yards_gained >= 10))).astype(int)
    off = p.groupby(["game_id", "posteam"]).agg(giveaways=("give", "sum"), sacks_taken=("sack", "sum"),
                                                expl_for=("expl", "sum")).reset_index().rename(columns={"posteam": "team"})
    opp = off.rename(columns={"team": "opp", "giveaways": "takeaways", "sacks_taken": "sacks_made",
                              "expl_for": "expl_against"})
    tg = off.merge(opp, on="game_id")
    return tg[tg.team != tg.opp]
