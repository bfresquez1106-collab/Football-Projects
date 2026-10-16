# Experiment log

Every change is tested the same way: rerun the 2022–26 walk-forward replay (1,167 games), compare
Brier scores game by game against the current model with a paired bootstrap (`compare.py`), and
keep the change only if the whole 95% interval shows improvement. Failed ideas are recorded too.

Brier: lower is better. Elo benchmark on the same games: **0.2232**.

| # | Date | Experiment | Brier | vs model before | Verdict |
|---|---|---|---|---|---|
| 0 | 2026-10-16 | Baseline v1 (`baseline_v1`) | 0.2282 | — | — |
| 1a | 2026-10-16 | Last season's plays weighted 0.35 / 0.5 / 0.75 all season (`wprior_*`) | 0.2282–0.2306 | +0.0000 to +0.0025 | Rejected: helps Weeks 1–4, hurts Weeks 10+ |
| 1b | 2026-10-16 | Weight that fades from 0.75 in Week 1 to 0.20 by Week 9 (`fade_75_8`) | 0.2271 | −0.0011 [−0.0027, +0.0005] | Not proven alone; re-tested with QBs |
| 2 | 2026-10-16 | QB adjustment, β = 0.5 / 1.0 / 1.5 (`qb_*`) | 0.2244 at β = 1.0 | −0.0038 [−0.0068, −0.0007] | **Kept** (β = 1.0, the theory value) |
| 2+1b | 2026-10-16 | QB adjustment + fading weight (`qb1_fade75`) | **0.2226** | −0.0018 on top of QB [−0.0036, −0.0002] | **Kept**: now the model |
| 3 | 2026-10-16 | Weather on totals: roof, wind, cold | total miss 10.70 / 10.65 vs 10.67 | roof worse, wind −0.02 pts | Rejected: fails out of sample; no weather for upcoming games |

## What each one taught

- **Season memory.** One fixed weight can't be right all year: last season is most of what you
  know in September and stale by December. A weight that fades fixes both ends.
- **QB index.** EPA and success rate per dropback are the most stable QB stats (half signal after
  ~250 and ~185 dropbacks). CPOE needs ~355 attempts and interception rate ~4,300 (mostly noise).
  EPA + success rate predicted future play best; adding CPOE, sack rate and interception rate
  didn't help, because they overlap with the first two (EPA and success correlate at 0.94).
- **QB adjustment.** β = 1.0 worked best, which is what the units predict: the rating is already
  in EPA per dropback. It helps most late in the season and catches offseason QB changes in Week 1.
- **Weather.** Indoor games and windy games really do differ, but the pattern didn't hold on
  new seasons, and a feature has to be known before kickoff to be usable.

## Where that leaves Gridiron vs Elo

0.2226 vs 0.2232 on the same games: Gridiron is ahead in 59% of bootstrap resamples, which is a
tie, not a win. It wins 2022, 2023 and 2025, and loses 2024 and 2026.
