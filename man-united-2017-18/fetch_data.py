"""
Download Manchester United's 2017-18 Premier League events from the public
Wyscout dataset (Pappalardo et al., 2019).

The original files live on figshare. This script uses koenvo's GitHub mirror,
which splits the same data into one JSON file per match, so we only download
United's 38 games instead of the whole 380-game league file.

Output (in data/Wyscout/):
    events_ManUnited.json   every event from United's 38 matches, both teams,
                            same format as the figshare events_England.json
    matches_ManUnited.json  matchId, label and date for each match
    teams_ManUnited.json    the 20 Premier League teams, keyed by teamId
    players_ManUnited.json  squads per match, keyed by matchId then teamId

Run:
    python fetch_data.py
"""
import json
import pathlib
import re

import requests

BASE_URL = "https://raw.githubusercontent.com/koenvo/wyscout-soccer-match-event-dataset/main/processed"
TEAM_NAME = "Manchester United"
TEAM_ID = 1611
OUT_DIR = pathlib.Path(__file__).resolve().parent / "data" / "Wyscout"

##############################################################################
# Finding United's matches
# ----------------------------
#
# The mirror's index is a markdown table with one row per match, e.g.
# |[2499724](files/2499724.json)|Manchester United - West Ham United, 4 - 0|August 13, 2017 at ...|matches_England.json|

ROW = re.compile(r"^\|\[(\d+)\]\(files/\d+\.json\)\|(.+?)\|(.+?)\|(.+?)\|$")


def find_matches(session):
    index = session.get(BASE_URL + "/README.md", timeout=60)
    index.raise_for_status()
    matches = []
    for line in index.text.splitlines():
        row = ROW.match(line.strip())
        if row is None:
            continue
        match_id, label, date, source = row.groups()
        if source == "matches_England.json" and TEAM_NAME in label.split(",")[0]:
            matches.append({"matchId": int(match_id), "label": label, "date": date})
    return matches


##############################################################################
# Downloading the events
# ----------------------------
#
# Each match file holds "events", "teams" and "players". The events are already
# in the figshare format, so existing tutorial code works on them unchanged.

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    matches = find_matches(session)
    print(f"Found {len(matches)} {TEAM_NAME} matches")

    events, teams, players = [], {}, {}
    for i, match in enumerate(matches, start=1):
        cache = OUT_DIR / "matches" / f"{match['matchId']}.json"
        if not cache.exists():
            cache.parent.mkdir(exist_ok=True)
            response = session.get(f"{BASE_URL}/files/{match['matchId']}.json", timeout=60)
            response.raise_for_status()
            cache.write_bytes(response.content)
        data = json.loads(cache.read_text())
        events.extend(data["events"])
        teams.update(data["teams"])
        players[str(match["matchId"])] = data["players"]
        print(f"[{i:2}/{len(matches)}] {match['label']}")

    united_events = sum(e["teamId"] == TEAM_ID for e in events)
    print(f"{len(events)} events in total, {united_events} by {TEAM_NAME}")

    for name, obj in [("events", events), ("matches", matches),
                      ("teams", teams), ("players", players)]:
        with open(OUT_DIR / f"{name}_ManUnited.json", "w") as f:
            json.dump(obj, f)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
