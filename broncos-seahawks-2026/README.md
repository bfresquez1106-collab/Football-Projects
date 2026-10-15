# Broncos vs Seahawks — Week 6, 2026

**Question:** What are Denver's chances against Seattle (Thu Oct 15, 2026, Mile High), and
which controllable levers move those chances the most?

## Answer (as of kickoff)

| | |
|---|---|
| Market | DEN +2.5, total 40.5 |
| Forecast | DEN 19.0, SEA 21.5 |
| DEN win probability | **42%** |
| DEN +2.5 covers | 48% |
| Over 40.5 | 50% |

The forecast is the market line. The walk-forward backtest (319 games, 2025 wk 2 – 2026 wk 5)
found that every amount of weight on the EPA model made predictions *worse*, so the default
blend weight is 0. The model is still useful for two things:

1. **Explaining the line.** Without injury adjustments the model's margin is −2.5, matching the
   market exactly. With the *tested* injury adjustments (below) it's −1.7, DEN 45%: Seattle
   losing LB Drake Thomas slightly outweighs Denver's missing secondary.
2. **Pricing the levers.**

| Broncos lever | Points | DEN win prob |
|---|---|---|
| +1 in turnover margin | +4.5 | 56% (+13.6pp) |
| Avoid 1 sack | +2.0 | 48% (+5.8pp) |
| Get 1 more sack | +2.0 | 48% (+5.8pp) |
| Push SEA run-heavier by 10pp | +0.7 | 45% (+2.2pp) |
| DEN pass-heavier by 10pp | +0.4 | 44% (+1.3pp) |

Turnovers dominate. Play-mix tilts are small and shrunk by half, because defenses adjust.

## Injury adjustments, tested (`injury_test.py`)

The original adjustments (+0.07 EPA to DEN pass D, −0.03 to SEA run game) were judgement calls.
Testing them on 1,054 team-games (2024–26): starters come from season-to-date snap counts, and a
starter with no snaps in a game counts as missing (this catches IR, unlike the injury report).

| Per missing starter | EPA/play | 95% CI |
|---|---|---|
| DB → opponent pass | +0.008 | −0.012 to +0.029 |
| DB → opponent rush | +0.012 | −0.004 to +0.027 |
| LB → opponent pass | **+0.043** | +0.008 to +0.078 |
| LB → opponent rush | +0.023 | −0.005 to +0.052 |
| RB1 → own rush | −0.010 | −0.046 to +0.026 |

- DEN (Surtain II, Hufanga out; Moss questionable, and questionable starters sit 32% of the
  time) → +0.019 pass / +0.028 rush. The data puts the original +0.07 at about 4× too large.
  Caveat: this is the *typical* starter; an elite CB like Surtain is likely worth more.
- Charbonnet hasn't played in 2026, so his absence is already in SEA's ratings. SEA's RB1 by
  snaps (Wilson) is active, so no adjustment.
- SEA LB Drake Thomas (77–100% of snaps every week) is out → +0.043 pass / +0.023 rush for DEN.
  It's the only significant result of the five tests, so treat it as likely, not certain.

## Broncos upside scenarios (`scenarios.py`)

What the final score looks like when the levers break Denver's way, priced from 3,726
team-games (2019–25). "Projected" prices exactly that stat line; "Comparable games" takes real
games at least that good, scores them against their own lines and puts them on tonight's line.

| Scenario (DEN stats; rest league avg) | Projected | Comparable games | DEN wins | Happens |
|---|---|---|---|---|
| Average game | DEN 19, SEA 21 | 19–22 | 41% | — |
| Clean offense: 0 giveaways, ≤1 sack taken | 22–19 | 25–19 | 69% | 13% of games |
| Win turnovers: 2+ takeaways, 0 giveaways | 22–18 | 26–17 | 78% | 12% |
| Defense takes over: + 4 sacks, ≤1 allowed | 24–16 | 25–15 | 81% | 2% |
| Explosive night: 0 giveaways, 10+ explosives, ≤5 allowed | 27–19 | 31–18 | 90% | 1.5% |
| Great night: +2 TO, 3 sacks, 7 explosives | 24–17 | 30–14 | 88% | 1% |
| Ideal night: +3 TO, 5 sacks, 10 explosives | 31–14 | 31–12 | — | 1 game in 3,726 |

Per extra event: takeaway +1.7 pts for, giveaway −1.6, sack taken −0.9, explosive play +1.3.
In an average game DEN wins by 7+ 22% of the time; the 90th-percentile outcome is DEN by 13–14.
Descriptive, not causal: trailing teams pass more and so take more sacks and throw more picks.

## Method

1. **Ratings:** ridge regression of EPA/play on offense + defense + home field, fit separately for
   pass and rush plays. Garbage time is removed (0.05 < WP < 0.95), neutral-site games get no home
   term, and prior-season plays are weighted 0.20.
2. **Points:** league PPG + plays × (matchup EPA − league EPA).
3. **Blend:** model margin vs market spread, with the weight picked by `backtest.py`.
4. **Outcome distribution:** a normal margin (SD 12.7, from 2015–2025 spread errors), re-weighted
   so key numbers land as often as they really do. Losing by exactly 3 is 7.5% here vs 3.1%
   under a plain normal, which is why DEN +2.5 covers only 48% even though the line is centred.

## Backtest findings (`backtest.py`)

- Best ratings settings: prior weight 0.20, ridge alpha 250. Model-only margin RMSE is 13.38,
  vs 12.41 for the market.
- Where the model disagreed with the line, results went the *other* way in 2025
  (slope −0.39) and the model's way in 2026 (+0.40, only 63 games). Five weeks is too few games
  to pick a blend weight.
- Bootstrap best blend weight: 0.00 (80% interval 0.00–0.00).
- The key-number win probabilities are well calibrated (each bin within ~2pp of actual).

## Ideas to try next

- Add features the market may underweight early in a season: QB-specific ratings,
  rest days, travel and weather.
- Weight missing starters by quality (e.g. prior-season EPA or PFF grade) so an elite CB counts
  for more than a depth starter.
- Log every forecast before kickoff and score it after the game (Brier score).

## Run it

```bash
pip install -r requirements.txt
mkdir -p data && cd data
R=https://github.com/nflverse/nflverse-data/releases/download
for y in $(seq 2019 2026); do curl -sSL -o pbp_$y.csv.gz $R/pbp/play_by_play_$y.csv.gz; done
for y in 2024 2025 2026; do
  curl -sSLO $R/injuries/injuries_$y.csv; curl -sSLO $R/snap_counts/snap_counts_$y.csv
done
curl -sSLO https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
cd ..
python3 model.py                         # tonight's game; --game-id for any other
python3 model.py --model-weight 0.3      # see the forecast with model weight
python3 backtest.py                      # ~1 minute
python3 injury_test.py                   # ~1 minute
python3 scenarios.py
```

Data: [nflverse](https://github.com/nflverse/nflverse-data) play-by-play and schedules. Injury
adjustments are judgement calls set in `model.py` (`Config.injuries`), not fitted values.
