"""
Elo: the benchmark model. One rating per team, updated from final scores.

Settings are FROZEN at the values tuned on 2002-2021 in ../elo-model (the full lesson lives
there). A benchmark only works if it doesn't move, so these never get retuned mid-season.
"""
import math
from collections import defaultdict

K, HFA, REVERT = 20, 55, 0.5          # reaction speed, home field (Elo pts), offseason pull to 1500
POINTS_PER_ELO = 1 / 25
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Tiers: fixed lines on "chance to beat an average team on a neutral field".
# Football field position, from the opponent's goal line back to your own.
TIERS = [  # (name, medal, minimum Elo above average, what it means)
    ("Red Zone", "gold", 100, "beats an average team 64%+ of the time"),
    ("Field Goal Range", "silver", 35, "beats an average team 55–64%"),
    ("Midfield", "bronze", -35, "a coin flip against an average team (45–55%)"),
    ("Backed Up", None, -10_000, "loses to an average team more often than not"),
]


def win_prob(gap):
    return 1 / (1 + 10 ** (-gap / 400))


def tier(rating, avg=1500.0):
    for name, medal, floor, _ in TIERS:
        if rating - avg >= floor:
            return name
    return TIERS[-1][0]


def ratings_through(games, season, before_week, history=None):
    """
    Replay every finished game (1999 on) up to `season` week `before_week`; return ratings.
    Pass a dict as `history` to also collect each game's pre-game home win probability.
    """
    g = games[games.result.notna()].replace({"home_team": RELOCATED, "away_team": RELOCATED})
    g = g[(g.season < season) | ((g.season == season) & (g.week < before_week))]
    ratings, current = defaultdict(lambda: 1500.0), None
    for x in g.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        if x.season != current:                       # offseason: halfway back to average
            for t in ratings:
                ratings[t] = 1500 + (1 - REVERT) * (ratings[t] - 1500)
            current = x.season
        gap = ratings[x.home_team] + (0 if x.location == "Neutral" else HFA) - ratings[x.away_team]
        if history is not None:
            history[x.game_id] = win_prob(gap)
        actual = 1.0 if x.result > 0 else 0.0 if x.result < 0 else 0.5
        mult = 1.0
        if x.result != 0:                             # margin-of-victory multiplier
            mult = math.log(abs(x.result) + 1) * 2.2 / ((gap if x.result > 0 else -gap) * 0.001 + 2.2)
        change = K * mult * (actual - win_prob(gap))
        ratings[x.home_team] += change
        ratings[x.away_team] -= change
    if current != season:                             # week 1: apply this season's offseason pull
        for t in ratings:
            ratings[t] = 1500 + (1 - REVERT) * (ratings[t] - 1500)
    return dict(ratings)


def predict(ratings, home, away, neutral=False):
    gap = ratings[home] + (0 if neutral else HFA) - ratings[away]
    return {"elo_wp": win_prob(gap), "elo_margin": gap * POINTS_PER_ELO}
