# Match Report: a one-page soccer match report from open event data

One function call draws a full one-page report for any match in StatsBomb's free open data:

```python
from match_report import match_report
match_report(18235, "Inter Milan", "Bayern Munich")
```

The example above is the 2010 Champions League final (Inter 2-0 Bayern).

![Inter 2-0 Bayern](example_inter_bayern.png)

A second match, the 2011 final (Barcelona 3-1 Manchester United), to show it works on a different game:

![Barcelona 3-1 Manchester United](example_barcelona_manutd.png)

## What is on the page

| Panel | What it shows |
|---|---|
| Pass maps | Average position of each player, linked by how often they passed to each other. Substitutes are grey. |
| Stats table | Shots, xG, xG per shot, possession, field tilt, crosses, corners, fouls, PPDA |
| Pass panels | Passes in the final third, into the final third, and to the box (completed in color, incomplete in grey) |
| Shot map | Every shot sized by xG, goals labelled |
| Defensive actions | Recoveries, clearances, interceptions, blocks and tackles, with the average defensive line and PPDA |
| xT maps | Expected threat added by zone, using the same color scale for both teams |

## How to run it

1. Install the packages: `pip install -r requirements.txt`
2. Keep `xt16.npy` in the same folder as `match_report.py`.
3. Find a match: `sb.matches(competition_id=..., season_id=...)` lists match ids and the exact team names.
4. Call `match_report(match_id, "Team A", "Team B")`.

Team names must match StatsBomb exactly (for example "Manchester United"). Heatmap colors are available for "blue" and "red".

## Method notes

- **xT (expected threat):** a 16 by 12 grid of zone values built with the position-based Markov method from the Soccermatics course, using 64 matches from the 2018 World Cup. A completed pass or carry is worth the value of its end zone minus the value of its start zone. Only moves that add threat are counted.
- **PPDA:** opponent passes in their own 60% of the pitch, divided by our tackles, interceptions and fouls in the pressing zone. StatsBomb pressure events are not counted, so numbers differ from definitions that include them.
- **Field tilt:** a team's share of all final-third passes.
- **To the box:** a pass counts if it ends inside the penalty area, wherever it started.

## Limitations

- One match is a small sample. A single action can dominate a zone, so the xT maps use coarse zones for display.
- The xT grid comes from 64 matches and is noisier than grids trained on full seasons.
- xT totals mostly reflect who had the ball. A fair comparison across matches would use per-90 or per-touch rates.

## Credits

- Event data: [StatsBomb open data](https://github.com/statsbomb/open-data). Data provided by StatsBomb.
- Expected threat: Karun Singh, "Introducing Expected Threat (xT)", and the Soccermatics course by David Sumpter.
- Plotting: mplsoccer.

Built while learning soccer analytics.
