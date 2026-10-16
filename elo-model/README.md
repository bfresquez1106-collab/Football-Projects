# Elo model (a learning project)

**Question:** How far can one number per team, updated from final scores alone, get you in
predicting NFL games, and how does it compare with the EPA-based Gridiron Model?

**Answer:** Further than our EPA model. On the same 1,167 games (2022–26), Elo scored a better
Brier (0.223 vs 0.228), better in all five seasons, and the gap is very unlikely to be luck.

## The idea

1. Every team has a rating; average is 1500.
2. Before a game, the gap (plus home field) becomes a win probability:
   `p = 1 / (1 + 10^(-gap/400))`. A 0 gap is 50%, +100 is 64%, +200 is 76%.
3. After the game: `change = k × (actual − p)`, where actual is 1 for a win and 0 for a loss.
   The winner gains what the loser loses.

The `(actual − p)` part is the heart of it: **ratings move by how surprised the model was.**
A 90% favorite that wins learns almost nothing (+0.1). An underdog that wins teaches a lot.

## The four knobs (tuned on 2002–2021, tested on 2022–26)

| Knob | Best value | What it controls |
|---|---|---|
| `k` | 20 | Reaction speed. Higher chases noise; lower misses real change. |
| Home field | 55 Elo (~2.2 pts) | Close to the +2.6 pts the Gridiron calibration found |
| Season reversion | 50% | Rosters turn over, so half of last year's edge carries into this year |
| Margin of victory | on | Biggest single gain: log loss 0.637 → 0.629 |

## Results on the test seasons (2022–26, same games for every model)

| Model | Brier | Log loss | Picks winner |
|---|---|---|---|
| Always home team | 0.247 | 0.690 | 54.6% |
| **Elo** | **0.223** | **0.638** | 63.0% |
| Gridiron Model | 0.228 | 0.650 | 63.1% |
| Average of the two | 0.224 | 0.642 | 63.1% |

- Elo is better in every season, 2022 through 2026 (paired bootstrap: 95% interval for the
  Brier gap −0.009 to −0.001).
- The two models pick the same winner 87% of the time, so averaging them adds little.

## Why does the simple model win?

- **Points include everything:** special teams, penalties and red-zone finishing, which
  play-by-play EPA under-weights or leaves out.
- **Better memory between seasons.** Elo keeps 50% of last year; Gridiron weights last year's
  plays at only 20%, so its early-season ratings are noisier.
- **Fewer knobs, less to get wrong.** Four settings tuned on 20 seasons, against dozens of
  estimates from a few hundred games.

## Run it

```bash
python3 elo.py --games ../nfl-model/data/games.csv
```

It tunes, runs the test, walks through the Seahawks–Broncos update step by step, and writes
`elo_predictions.csv`.
