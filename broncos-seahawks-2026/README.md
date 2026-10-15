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
   market exactly. The model-only view (DEN 37%) comes entirely from the injury
   judgement call on Surtain II / Hufanga (DEN pass D) and Charbonnet (SEA run game).
2. **Pricing the levers.**

| Broncos lever | Points | DEN win prob |
|---|---|---|
| +1 in turnover margin | +4.5 | 56% (+13.6pp) |
| Avoid 1 sack | +2.0 | 48% (+5.8pp) |
| Get 1 more sack | +2.0 | 48% (+5.8pp) |
| Push SEA run-heavier by 10pp | +0.7 | 45% (+2.2pp) |
| DEN pass-heavier by 10pp | +0.4 | 44% (+1.3pp) |

Turnovers dominate. Play-mix tilts are small and shrunk by half, because defenses adjust.

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
- Test the injury adjustments: backtest games where a top CB was out.
- Log every forecast before kickoff and score it after the game (Brier score).

## Run it

```bash
pip install -r requirements.txt
mkdir -p data && cd data
for y in 2024 2025 2026; do
  curl -sSLO https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_$y.csv.gz
  mv play_by_play_$y.csv.gz pbp_$y.csv.gz
done
curl -sSLO https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
cd ..
python3 model.py                         # tonight's game; --game-id for any other
python3 model.py --model-weight 0.3      # see the forecast with model weight
python3 backtest.py                      # ~1 minute
```

Data: [nflverse](https://github.com/nflverse/nflverse-data) play-by-play and schedules. Injury
adjustments are judgement calls set in `model.py` (`Config.injuries`), not fitted values.
