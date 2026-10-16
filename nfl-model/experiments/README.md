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
| 4 | 2026-10-16 | Leak check: QB weights refit on 2019–23 only (`leakcheck`) | 0.2226 | 0.0000 | Result holds; leak was harmless |
| 5a | 2026-10-16 | QB index: throw + YAC + success, YAC weighted by regression (`qbv_split`) | 0.2227 | +0.0001 [−0.0001, +0.0002] | Neutral; agrees better with QBR (0.797 vs 0.782) |
| 5b | 2026-10-16 | QB index: no YAC credit at all (`qbv_noyac`) | 0.2230 | +0.0003 [−0.0001, +0.0008] | Rejected: YAC is partly the QB's doing |
| 5c | 2026-10-16 | QB index: + designed runs at face value (`qbv_run`) | 0.2224 | −0.0002 [−0.0005, +0.0000] | Neutral, slightly better |
| 5d | 2026-10-16 | 5a + 5c together (`qbv_split_run`) | **0.2224** | −0.0002 [−0.0005, +0.0001] | **Kept** by the tiebreak rule below (QBR agreement 0.791) |

| 6a | 2026-10-16 | QB lab, transfer test (`qb_lab.py`): rating + new team's context, 71 QBs who changed teams | corr 0.172 raw → 0.255 medium / 0.293 strong | stayers 0.430 → 0.429 / 0.422 | Scheme adjustment works for movers |
| 6b | 2026-10-16 | Opponent adjustment (defense faced) | transfer corr 0.167 vs 0.172 | — | Rejected: no help |
| 6c | 2026-10-16 | Playmaking: explosive rate, scramble value, 3rd/4th-down value, deep passing | next-season error −0.9 / −1.4 / −0.4 / +0.2% | — | Rejected: already inside EPA |
| 6d | 2026-10-16 | Pressure (PFR): pressure rate, sacks per pressure | next-season error −2.3 / −2.4% | — | Rejected |
| 6e | 2026-10-16 | Scheme adjustment in the model, medium (`qbv_scheme1000`) / strong (`qbv_scheme300`) | 0.2223 / 0.2225 | −0.0001 / 0.0000 | **Kept: medium** (transfer test; gentler; no cost to stayers) |

**Tiebreak rule (written before applying it):** a change that doesn't move game predictions is kept
if it's grounded in how football works *and* agrees at least as well with an independent measure
(ESPN QBR). QB index changes mostly fall here, because the QB adjustment only matters when the
starter differs from the QBs already inside a team's numbers.

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
- **QB index vs the real world.** Our index and ESPN QBR rank QBs similarly (0.78). The gaps came
  from two places: EPA gives the QB full credit for yards after the catch, which inflates QBs in
  YAC-heavy schemes, and dropbacks miss designed runs (Allen, Hurts, Dart). Crediting YAC at the
  regression's ~75% and adding designed runs fixed both directions without hurting predictions.

- **Scheme vs QB.** A QB's raw numbers carry his team: when he changes teams, his rating predicts
  his new numbers poorly (0.17). Taking out his old team's context and adding the new one's nearly
  doubles that (0.29 at the strongest setting). QBR shares the same bias, so it can't judge this
  change; the transfer test can. Effects: Mahomes and Herbert up (their teams held them back),
  Mac Jones, Maye and Darnold down (their teams lifted them), Purdy still #1.
- **Playmaking and pressure.** What separates Mahomes shows up in scrambles (2nd) and 3rd/4th downs
  (5th), but those plays are already high-EPA plays: weighting them again double-counts. Pressure
  rate is partly the QB's own doing, so it isn't a clean line measure.
- **Opponent strength** washes out over a few hundred dropbacks; it didn't help.

## Where that leaves Gridiron vs Elo

0.2225 vs 0.2232 on the same games: Gridiron is ahead in 61% of bootstrap resamples, which is a
tie, not a win. It wins 2022, 2023, 2025.
