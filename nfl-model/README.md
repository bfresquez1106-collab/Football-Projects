# Gridiron Model

**Question:** Using only on-field data (no betting lines), how well can we predict every NFL
game, and what moves each one?

A personal season-long model. Every week it rates all 32 teams from play-by-play, reads the
injury report, predicts the full slate, logs each prediction before kickoff, and grades it after.
`dashboard/nfl_model.html` is the viewable version: this week's slate, a matchup tool for any
two teams, the season record, and team ratings.

Grew out of [broncos-seahawks-2026](../broncos-seahawks-2026/), where each piece was first
built and tested on one game.

## How a prediction is made

1. **Team ratings.** Ridge regression of EPA/play on offense + defense + home field, fit
   separately for passes and runs. Garbage time is dropped (win probability outside 5–95%), and
   last season's plays count 0.20 as a prior.
2. **Injuries.** Starters come from season-to-date snap counts (top 5 DBs, top 2 LBs). Each
   starter on the report counts as missing by how often players with that status sit: Out 100%,
   Doubtful 91%, Questionable 40%. A missing starting DB adds +0.008 pass / +0.012 rush EPA/play
   to the opposing offense, and a missing LB adds +0.043 / +0.023 (from `injury_test.py`).
3. **Raw score.** League points per game + plays × (matchup EPA − league EPA).
4. **Calibration.** A week-by-week replay of 2022–26 (1,167 games) checks the raw model against
   real results and corrects it:
   - margin = **+2.6 home-field points** (none at neutral sites) + **0.81** × raw margin.
     The raw model under-counts home field and is a bit overconfident.
   - total = 20.9 + **0.52** × raw total. Only about half of the raw total's swing holds up.
5. **Outcome distribution.** A normal margin (SD 13.0) re-weighted so 3, 7 and other key margins
   land as often as they really do. This gives the win probability and the score range.
6. **Levers and scenarios.** Points over the model's expectation per takeaway (+1.8), giveaway
   (−1.6), sack taken (−1.0) and explosive play (+1.5), learned from the same replay.

## How good is it (replay, 1,167 games)

| | Brier (0 = perfect, 0.25 = coin flip) | Picks winner | Margin miss |
|---|---|---|---|
| **Model** | **0.229** | **63%** | 10.0 pts |
| Raw model, no calibration | 0.234 | 61% | 10.3 pts |
| Always pick the home team | 0.248 | 55% | 10.8 pts |

Each season is scored with calibration fit on the other seasons, so the scorecard isn't grading
its own answers. The season-by-season Brier score holds steady at 0.220–0.235.

## The Elo benchmark

Every game is also predicted by Elo (`nflmodel/elo.py`): one rating per team, updated from
final scores, with settings frozen at the values tuned on 2002–2021 in
[elo-model](../elo-model/). It's the bar Gridiron has to clear, and right now it's ahead
(replay Brier 0.223 vs 0.228). Both models are logged before every kickoff and graded on the
same games, so the Record tab shows the live head-to-head.

The Ratings tab groups teams into five **Elo tiers**, using fixed lines on the chance of
beating an average team on a neutral field:

| Tier | Elo vs average | Beats an average team |
|---|---|---|
| Dynasty | +110 or more | 65%+ |
| Powerhouse | +50 to +110 | 57–65% |
| Contender | −50 to +50 | 43–57% |
| Challenger | −110 to −50 | 35–43% |
| Underdog | below −110 | under 35% |

Fixed lines mean tier sizes change with the league. Averaged over the 2010–2025 season ends,
the tiers held 6.1 / 4.3 / 11.1 / 5.1 / 5.3 teams. Ratings spread out as a season goes on, so
early-season tiers are more crowded in the middle.

## Earlier weeks, reconstructed

Games played before logging started (2026 Weeks 1–5, plus Denver–Seattle in Week 6) appear on
the Record tab in their own section, rebuilt from the walk-forward replay: each game predicted
only from games before its week. The data has no hindsight, but some settings were tuned while
those weeks were visible, so it's kept apart from the live record.

## The rules this model follows

- **No betting lines.** `nflmodel/data.py` drops every line and odds column at load.
- **Log before kickoff, grade after.** Each game is graded on the last prediction logged before
  kickoff, and a game is logged again only when its prediction changes.
- **Every adjustment must pass a backtest first.** Injuries passed; judgement-call versions
  didn't.
- **Don't retune mid-season** without rerunning `calibrate.py` and comparing scorecards.

## Weekly schedule

A scheduled task runs `weekly.py` Tuesday, Thursday and Saturday mornings (Mountain time):
Tuesday picks up last week's results and new ratings, Thursday catches the Thursday-night
injury report, and Saturday the final reports for Sunday and Monday. Each run republishes the
dashboard and commits the log.

## Run it

```bash
pip install -r requirements.txt
python3 weekly.py                      # refresh data, predict + log the week, grade, build dashboard
python3 predict.py DEN KC              # KC at DEN with this week's injury report
python3 predict.py DEN KC --neutral --home-db-out 2 --away-lb-out 1
python3 calibrate.py --fetch           # rebuild params.json from the 2022+ replay (~1 min + downloads)
```

| File | What it is |
|---|---|
| `nflmodel/` | data loading, ratings, injuries, calibrated predictions |
| `params.json` | calibration, outcome distribution, lever values, replay scorecard |
| `forecast_log.csv` | every logged prediction, with when it was logged and the injury inputs |
| `dashboard/template.html` | the dashboard; `weekly.py` fills in the data to make `nfl_model.html` |

## Next improvements to test

- **QB-level ratings.** Team ratings can't tell when a backup starts, and that's likely the
  biggest single gain.
- **Wind and temperature** for totals. Totals are the weakest part (slope 0.52).
- **Within-season recency.** Weight recent weeks more than September.
- **Weight missing starters by quality**, so an All-Pro corner counts for more than a depth player.

Data: [nflverse](https://github.com/nflverse/nflverse-data) play-by-play, schedules, injury
reports and snap counts.
