# Football Projects: context for Claude

Personal NFL analytics, built as a learning project. Read this first; details live in each
project's README and in `nfl-model/experiments/README.md`.

## How the user wants to work

- **Mentor style.** Explain what, why and how, plainly. Teach the concept, not just the result.
- **No betting lines.** The model is built only from on-field data (`nflmodel/data.py` drops
  every line and odds column). Don't add market data to the model.
- **Simple beats complex.** Prefer the simplest version that tests best.

## Projects

| Folder | What it is |
|---|---|
| `nfl-model/` | **Gridiron Model**, the main project: season-long predictions for every game, a dashboard, weekly runs |
| `elo-model/` | Elo learning project (scores only); its frozen settings are the benchmark inside `nfl-model/nflmodel/elo.py` |
| `broncos-seahawks-2026/` | The original one-game case study (Week 6, 2026) where each piece was first built |

## Gridiron Model at a glance

- **Team ratings:** ridge regression of EPA/play (pass and rush separately). Last season's plays
  count 0.75 in Week 1, fading to 0.20 by Week 9.
- **QB adjustment:** starter's QB rating minus the QB rating already inside the team's numbers
  (β = 1.0). Starters come from depth charts (QB1 unless Out/Doubtful).
- **QB index** (`qb_index.py`, `nflmodel/qb.py`): throw value + YAC at ~75% + success rate +
  designed runs, shrunk toward backup level, recency-weighted, with team context (scheme)
  removed (medium setting). 100 = average starter, 10 = one standard deviation.
- **Injuries:** missing starting DBs/LBs from snap counts + injury report.
- **Calibration** from a 2022–26 walk-forward replay (1,167 games): +2.6 home-field points,
  margin shrink 0.78, totals shrink ~0.5; key-number-aware score distribution.
- **Scores (replay Brier, lower is better):** Gridiron **0.2225**, Elo **0.2232**, a statistical
  tie (Gridiron better in ~61% of resamples). Gridiron picks ~64.7% of winners.
- **Elo tiers** on the dashboard: Dynasty / Powerhouse / Contender / Challenger / Underdog
  (fixed Elo lines).

## Rules for changing the model

1. One change at a time. Run it as an experiment: `python3 calibrate.py --tag NAME [options]`,
   then `python3 compare.py NAME`.
2. **Keep it only if the paired bootstrap 95% interval is entirely below zero** (better).
3. **Tiebreak rule** for changes that don't move game predictions (most QB-index changes): keep
   if grounded in how football works *and* agrees at least as well with an independent measure
   (ESPN QBR; or the transfer test in `qb_lab.py` when QBR shares the bias).
4. Record every experiment, including rejected ones, in `nfl-model/experiments/README.md`.
5. A feature must be known before kickoff to be usable live.
6. Never edit logged predictions after kickoff. `forecast_log.csv` rows carry a `model_version`
   fingerprint of `params.json` + `qb_params.json`.

Already tested and rejected: weather, opponent adjustment for QBs, playmaking stats (explosives,
scrambles, 3rd/4th down, deep), PFR pressure rates, fixed season-memory weights.

## Weekly operation

- Scheduled task **"Gridiron Model weekly run"** (`trig_01APYLngX5SvZqZW84Jk7FGK`) fires into the
  original Claude Code session Tue/Thu/Sat 8:47am Mountain: `git pull`, `python3 weekly.py`,
  republish the dashboard, commit `forecast_log.csv` + dashboard, push.
- Dashboard artifact: https://claude.ai/artifact/EmCoARc1npBefUyzZHDDX9 (built from
  `dashboard/template.html` by `weekly.py`; the page self-checks its math against Python).
- Broncos highlight page: https://claude.ai/artifact/NXPSkRM4bNAjwMFqVuyxdG

## Practical notes

- `nfl-model/data` and `broncos-seahawks-2026/data` are gitignored links to a scratch folder.
  In a fresh container, `python3 weekly.py` re-downloads what it needs; `calibrate.py --fetch`
  and `qb_index.py` need 2016–2026 play-by-play (nflverse releases).
- Python scripts in this repo import their own modules; run them normally (not with `-I`).
- Work happens on branch `broncos-seahawks-model`; it hasn't been merged into `main` yet.

## Open next steps

- Test a draft-position starting point for rookie QBs (all rookies start at backup level today).
- Merge `broncos-seahawks-model` into `main` (pull request).
- Watch the live season record (aim: ~100 graded games before judging it against the replay).
