# Manchester United 2017-18

**Question:** how did Manchester United move the ball in the 2017-18 Premier
League season? (Starting point: clustering their progressive passes.)

## Data

The public Wyscout match event dataset — Pappalardo et al. (2019),
[*A public data set of spatio-temporal match events in soccer competitions*](https://www.nature.com/articles/s41597-019-0247-7),
licensed CC BY 4.0. `fetch_data.py` downloads United's 38 league matches from
[koenvo's per-match mirror](https://github.com/koenvo/wyscout-soccer-match-event-dataset)
of the figshare files. The events keep the original figshare format.

| File (in `data/Wyscout/`) | Contents |
|---|---|
| `events_ManUnited.json` | ~66k events from United's 38 matches (both teams) |
| `matches_ManUnited.json` | `matchId`, label (teams and score), date |
| `teams_ManUnited.json` | Team info keyed by `teamId` (United is `1611`) |
| `players_ManUnited.json` | Squads keyed by `matchId`, then `teamId` |

The data is not committed; run the fetch script to create it.

## How to run

```shell
pip install -r requirements.txt
python fetch_data.py
```

```python
import json, pandas as pd
df = pd.DataFrame(json.load(open("data/Wyscout/events_ManUnited.json")))
united = df[df["teamId"] == 1611]
```
