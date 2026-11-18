"""One-page match report from StatsBomb open data.

Usage:
    from match_report import match_report
    match_report(18235, "Inter Milan", "Bayern Munich",
                 subtitle="Champions League final, 22 May 2010")

Needs xt16.npy (the xT grid) in the same folder as this file.
Team names must match StatsBomb exactly (e.g. "Manchester United").
Every color and the font live in THEME below: change them there.
"""
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from statsbombpy import sb

warnings.filterwarnings("ignore")

# ------------------------------------------------------------- THEME
# Rams x Eames: warm paper, near-black ink, one pair of team colors,
# one highlight color (the average defensive line).
THEME = {
    "paper":    "#F3F8F2",     # background (soft mint white)
    "ink":      "#2B2D42",     # text, key marks (deep indigo)
    "graphite": "#5E6178",     # pitch lines, secondary text
    "stone":    "#B9C4C0",     # misses, clearances, quiet marks
    "team_a":   "#3581B8",     # left team (blue)
    "team_b":   "#C33149",     # right team (crimson)
    "accent":   "#499167",     # the one highlight (green)
    "font":     ["Roboto", "Franklin Gothic Medium", "Segoe UI", "Arial", "DejaVu Sans"],
}

# One-knob sizing: raise these to make text and lines bigger everywhere.
FS = 1.25   # font scale
LW = 1.6    # line-width scale


# hand-made label fixes that nicknames in the lineups do not cover
KNOWN_NAMES = {
    "Esteban Matías Cambiasso Delau": "Cambiasso",
    "Júlio César Soares de Espíndola": "Júlio César",
    "Lucimar da Silva Ferreira": "Lúcio",
    "Samuel Eto''o Fils": "Eto'o",
    "Walter Adrián Samuel Luján": "Samuel",
    "Mario Gómez García": "Gómez",
    "Daniel Van Buyten": "Van Buyten",
    "Mark van Bommel": "Van Bommel",
    "Maicon Douglas Sisenando": "Maicon",
}

NX, NY = 16, 12
XT16 = np.load(Path(__file__).parent / "xt16.npy")


# ----------------------------------------------------- small helpers

def _canvas(ax, **pitch_kwargs):
    """A themed pitch: draws on a new figure, or on the ax it is given."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=THEME["paper"],
                  line_color=THEME["graphite"], linewidth=1 * LW, **pitch_kwargs)
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
        fig.set_facecolor(THEME["paper"])
    else:
        pitch.draw(ax=ax)
    return pitch, ax


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=12 * FS, color=THEME["ink"], pad=8)


def _ramp(color):
    """Heatmap ramp: fades from the paper color into a team color."""
    return LinearSegmentedColormap.from_list("ramp", [THEME["paper"], color])


def _halo():
    """Paper-colored outline so labels stay readable over marks."""
    return [pe.withStroke(linewidth=3 * LW, foreground=THEME["paper"])]


# ---------------------------------------------------------------- xT

def zone(x, y):
    col = np.clip((np.asarray(x) / 120 * NX).astype(int), 0, NX - 1)
    row = np.clip((np.asarray(y) / 80 * NY).astype(int), 0, NY - 1)
    return row * NX + col


def zone_xt(x, y):
    return XT16.ravel()[zone(x, y)]


def xt_moves(events, team):
    # completed passes (no outcome) plus all carries
    m = events[(events["team"] == team) & events["type"].isin(["Pass", "Carry"])].copy()
    m = m[m["pass_outcome"].isna()]
    m["end"] = m["pass_end_location"].combine_first(m["carry_end_location"])
    m = m.dropna(subset=["location", "end"])

    m["x"] = m["location"].apply(lambda loc: loc[0])
    m["y"] = m["location"].apply(lambda loc: loc[1])
    end_x = m["end"].apply(lambda loc: loc[0])
    end_y = m["end"].apply(lambda loc: loc[1])

    # value of where it ended minus value of where it started
    m["xt"] = zone_xt(end_x, end_y) - zone_xt(m["x"], m["y"])

    # progressive moves only, as in Soccermatics
    return m[m["xt"] > 0]


def xt_pitch(events, team, color, ax=None, min_label=0.08, vmax=None):
    m = xt_moves(events, team)
    pitch, ax = _canvas(ax, line_zorder=2)

    stat = pitch.bin_statistic(m["x"], m["y"], values=m["xt"],
                               statistic="sum", bins=(8, 6))
    pitch.heatmap(stat, ax=ax, cmap=_ramp(color), edgecolors=THEME["paper"],
                  linewidth=1 * LW, vmin=0, vmax=vmax, zorder=1)

    top = vmax if vmax else stat["statistic"].max()
    rows, cols = stat["statistic"].shape
    for r in range(rows):
        for c in range(cols):
            v = stat["statistic"][r, c]
            if v >= min_label:
                dark = v / top > 0.55
                ax.text(stat["cx"][r, c], stat["cy"][r, c], f"{v:.2f}",
                        ha="center", va="center", fontsize=10 * FS, zorder=3,
                        color=THEME["paper"] if dark else THEME["ink"])

    _title(ax, f"{team}: xT added by start zone (total {m['xt'].sum():.2f})")
    return ax


# ------------------------------------------------------- pass panels

ZONES = {
    "in_final_third":   ("Passes in final third",
                         lambda p: p["x"] >= 80),
    "into_final_third": ("Passes into final third",
                         lambda p: (p["x"] < 80) & (p["end_x"] >= 80)),
    "to_box":           ("Passes to the box",
                         lambda p: (p["end_x"] >= 102) & p["end_y"].between(18, 62)),
}


def pass_panel(events, team, color, zone, ax=None):
    title, rule = ZONES[zone]

    p = events[(events["team"] == team) & (events["type"] == "Pass")]
    p = p.dropna(subset=["pass_end_location"]).copy()
    p["x"] = p["location"].apply(lambda loc: loc[0])
    p["y"] = p["location"].apply(lambda loc: loc[1])
    p["end_x"] = p["pass_end_location"].apply(lambda loc: loc[0])
    p["end_y"] = p["pass_end_location"].apply(lambda loc: loc[1])

    p = p[rule(p)]
    done = p["pass_outcome"].isna()

    pitch, ax = _canvas(ax)

    # misses in stone underneath, completed passes in the team color on top
    miss, hit = p[~done], p[done]
    pitch.arrows(miss["x"], miss["y"], miss["end_x"], miss["end_y"],
                 color=THEME["stone"], alpha=0.8, width=1.5 * LW, headwidth=3.5,
                 headlength=3.5, zorder=1, ax=ax)
    pitch.arrows(hit["x"], hit["y"], hit["end_x"], hit["end_y"],
                 color=color, alpha=0.85, width=1.5 * LW, headwidth=3.5,
                 headlength=3.5, zorder=2, ax=ax)

    _title(ax, f"{team}: {title} ({done.sum()}/{len(p)}, "
               f"{done.mean() * 100:.0f}% complete)")
    return ax


# ------------------------------------------------------- pass network

def pass_network(events, team, color, short_names=None, min_passes=3, ax=None):
    short_names = short_names or {}

    p = events[(events["team"] == team) & (events["type"] == "Pass")].copy()
    p = p[p["pass_outcome"].isna()]
    p = p.dropna(subset=["pass_recipient"])
    p["x"] = p["location"].apply(lambda loc: loc[0])
    p["y"] = p["location"].apply(lambda loc: loc[1])

    avg = p.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), n=("type", "count"))

    subs = events[(events["team"] == team) & (events["type"] == "Substitution")]
    came_on = set(subs["substitution_replacement"].dropna())

    pairs = p.groupby(["player", "pass_recipient"]).size().reset_index(name="count")
    pairs = pairs[pairs["count"] >= min_passes]

    pitch, ax = _canvas(ax)
    _title(ax, f"{team} pass map")

    for _, row in pairs.iterrows():
        a = row["player"]
        b = row["pass_recipient"]
        if a in avg.index and b in avg.index:
            pitch.lines(avg.loc[a, "x"], avg.loc[a, "y"],
                        avg.loc[b, "x"], avg.loc[b, "y"],
                        lw=row["count"] / pairs["count"].max() * 6,
                        color=THEME["ink"], alpha=0.3, zorder=1, ax=ax)

    colors = [THEME["stone"] if name in came_on else color for name in avg.index]
    pitch.scatter(avg["x"], avg["y"], s=avg["n"] * 10, c=colors,
                  edgecolors=THEME["paper"], linewidth=1.5 * LW, alpha=0.95,
                  zorder=2, ax=ax)

    # labels sit just above each dot, with a paper halo so they stay readable
    for name, row in avg.iterrows():
        label = short_names.get(name, name.split()[-1])
        ax.text(row["x"], row["y"] - 3.5, label, ha="center", va="bottom",
                fontsize=9 * FS, color=THEME["ink"], path_effects=_halo(), zorder=4)

    return ax


# ----------------------------------------------------------- shot map

def shot_map(events, left_team, right_team, left_color, right_color,
             short_names=None, ax=None):
    short_names = short_names or {}

    shots = events[events["type"] == "Shot"].copy()
    shots["x"] = shots["location"].apply(lambda loc: loc[0])
    shots["y"] = shots["location"].apply(lambda loc: loc[1])

    pitch, ax = _canvas(ax)

    # the left team is flipped so it attacks the left goal
    totals = {}
    for team, color, flip in [(right_team, right_color, False),
                              (left_team, left_color, True)]:
        s = shots[shots["team"] == team]
        x, y = s["x"], s["y"]
        if flip:
            x, y = 120 - x, 80 - y

        goal = s["shot_outcome"] == "Goal"
        size = s["shot_statsbomb_xg"] * 1000

        pitch.scatter(x[~goal], y[~goal], s=size[~goal],
                      color=color, alpha=0.35, zorder=2, ax=ax)
        pitch.scatter(x[goal], y[goal], s=size[goal], color=color,
                      edgecolors=THEME["ink"], linewidth=1 * LW, alpha=1, zorder=3, ax=ax)

        for i in s.index[goal]:
            name = short_names.get(s.loc[i, "player"], s.loc[i, "player"].split()[-1])
            xg = s.loc[i, "shot_statsbomb_xg"]
            ax.text(x[i] + 2, y[i] - 4, f"{name} {xg:.2f}", fontsize=9 * FS,
                    color=THEME["ink"], path_effects=_halo(), zorder=4)

        totals[team] = s["shot_statsbomb_xg"].sum()

    _title(ax, "Shot map")
    ax.text(0, 87, f"{left_team}  {totals[left_team]:.2f} xG", ha="left", va="top",
            fontsize=12 * FS, color=left_color)
    ax.text(120, 87, f"{right_team}  {totals[right_team]:.2f} xG", ha="right", va="top",
            fontsize=12 * FS, color=right_color)
    ax.text(60, 87, "Dot size = xG, filled = goal", ha="center", va="top",
            fontsize=11 * FS, color=THEME["graphite"])
    return ax


# ------------------------------------------------- PPDA and stats table

def ppda(events, team, zone_x=48):
    # passes the opponent makes in their own 60% of the pitch
    opp = events[(events["type"] == "Pass") & (events["team"] != team)]
    opp_x = opp["location"].apply(lambda loc: loc[0])
    passes_allowed = (opp_x <= 120 - zone_x).sum()

    # our defensive actions in the pressing zone
    mine = events[events["team"] == team]
    is_tackle = (mine["type"] == "Duel") & (mine["duel_type"] == "Tackle")
    is_action = is_tackle | mine["type"].isin(["Interception", "Foul Committed"])
    acts = mine[is_action].dropna(subset=["location"])
    actions = (acts["location"].apply(lambda loc: loc[0]) >= zone_x).sum()

    return passes_allowed / actions


def stats_panel(events, left_team, right_team, left_color, right_color, ax,
                step=0.085, x_left=0.68, show_names=True):
    teams = [left_team, right_team]
    passes = events[events["type"] == "Pass"].copy()
    passes["x"] = passes["location"].apply(lambda loc: loc[0])

    n_pass = {t: (passes["team"] == t).sum() for t in teams}
    n_ft = {t: ((passes["team"] == t) & (passes["x"] >= 80)).sum() for t in teams}

    cols = {}
    for t in teams:
        shots = events[(events["team"] == t) & (events["type"] == "Shot")]
        mine = passes[passes["team"] == t]
        fouls = ((events["team"] == t) & (events["type"] == "Foul Committed")).sum()
        cols[t] = [
            f"{len(shots)}",
            f"{shots['shot_statsbomb_xg'].sum():.2f}",
            f"{shots['shot_statsbomb_xg'].mean():.2f}",
            f"{n_pass[t] / sum(n_pass.values()) * 100:.1f}",
            f"{n_ft[t] / sum(n_ft.values()) * 100:.1f}",
            f"{(mine['pass_cross'] == True).sum()}",
            f"{(mine['pass_type'] == 'Corner').sum()}",
            f"{fouls}",
            f"{ppda(events, t):.1f}",
        ]

    labels = ["Shots", "xG", "xG per shot", "Possession %", "Field tilt %",
              "Crosses", "Corners", "Fouls", "PPDA"]

    # drawn by hand: a header, then one hairline-separated row per stat
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_right = 1.0
    top = 0.5 + (len(labels) + 1) * step / 2

    y = top - step / 2
    if show_names:
        ax.text(x_left, y, left_team, ha="right", va="center", fontsize=13 * FS,
                fontweight="bold", color=left_color)
        ax.text(x_right, y, right_team, ha="right", va="center", fontsize=13 * FS,
                fontweight="bold", color=right_color)
    ax.plot([0, 1], [y - step / 2] * 2, color=THEME["ink"], lw=0.9 * LW, clip_on=False)

    for i, label in enumerate(labels):
        y = top - step / 2 - (i + 1) * step
        ax.text(0, y, label, ha="left", va="center", fontsize=12 * FS, color=THEME["graphite"])
        ax.text(x_left, y, cols[left_team][i], ha="right", va="center",
                fontsize=14 * FS, color=THEME["ink"])
        ax.text(x_right, y, cols[right_team][i], ha="right", va="center",
                fontsize=14 * FS, color=THEME["ink"])
        ax.plot([0, 1], [y - step / 2] * 2, color=THEME["ink"], lw=0.4 * LW,
                alpha=0.4, clip_on=False)
    return ax


# ------------------------------------------------- defensive actions

STYLES = {                       # kind: (marker, face color, edge color)
    "Ball recovery": ("o", THEME["paper"], THEME["ink"]),
    "Clearance":     ("s", THEME["stone"], THEME["graphite"]),
    "Interception":  ("D", THEME["ink"], THEME["ink"]),
    "Block":         ("v", THEME["paper"], THEME["ink"]),
    "Won tackle":    ("o", THEME["ink"], THEME["ink"]),
    "Lost tackle":   ("X", THEME["paper"], THEME["ink"]),
}
WON = ["Won", "Success In Play", "Success Out"]
LOST = ["Lost In Play", "Lost Out"]


def defensive_actions(events, team, ax=None, legend=True):
    d = events[events["team"] == team].dropna(subset=["location"]).copy()
    d["x"] = d["location"].apply(lambda loc: loc[0])
    d["y"] = d["location"].apply(lambda loc: loc[1])

    is_tackle = (d["type"] == "Duel") & (d["duel_type"] == "Tackle")
    d["kind"] = np.select(
        [d["type"] == "Ball Recovery",
         d["type"] == "Clearance",
         d["type"] == "Interception",
         d["type"] == "Block",
         is_tackle & d["duel_outcome"].isin(WON),
         is_tackle & d["duel_outcome"].isin(LOST)],
        ["Ball recovery", "Clearance", "Interception", "Block",
         "Won tackle", "Lost tackle"],
        default="other")
    d = d[d["kind"] != "other"]

    pitch, ax = _canvas(ax)

    for kind, (marker, face, edge) in STYLES.items():
        k = d[d["kind"] == kind]
        pitch.scatter(k["x"], k["y"], marker=marker, s=90, facecolors=face,
                      edgecolors=edge, linewidths=1 * LW, label=kind, zorder=3, ax=ax)

    avg_x = d.loc[d["kind"] != "Clearance", "x"].mean()
    high = (d["x"] >= 80).sum()
    ax.axvline(avg_x, color=THEME["accent"], lw=2.5 * LW, zorder=2,
               label="Average height (excl. clearances)")
    ax.text(avg_x, 84, f"{avg_x / 120 * 100:.1f}% up the pitch",
            fontsize=11 * FS, ha="center", va="top", color=THEME["ink"])
    ax.text(120, 84, f"PPDA {ppda(events, team):.1f}", fontsize=11 * FS, ha="right",
            va="top", color=THEME["graphite"])
    ax.text(120, 89, f"{high} actions in the final third", fontsize=11 * FS, ha="right",
            va="top", color=THEME["graphite"])

    if legend:
        ax.legend(loc="lower right", fontsize=8 * FS, framealpha=0.9)
    _title(ax, f"{team}: defensive actions ({len(d)})")
    return ax


# --------------------------------------------------- the whole page

def goals(events, team):
    shot_goals = ((events["team"] == team) & (events["type"] == "Shot")
                  & (events["shot_outcome"] == "Goal") & (events["period"] < 5)).sum()
    own_goals = ((events["team"] == team) & (events["type"] == "Own Goal For")).sum()
    return shot_goals + own_goals


def auto_short_names(match_id):
    lineups = sb.lineups(match_id=match_id)
    frames = lineups.values() if isinstance(lineups, dict) else [lineups]
    names = {}
    for df in frames:
        for _, row in df.iterrows():
            nick = row["player_nickname"]
            label = nick if isinstance(nick, str) else row["player_name"]
            names[row["player_name"]] = label.split()[-1]
    return names


def match_report(match_id, left, right, left_color=None, right_color=None,
                 short_names=None, subtitle=None, save_as=None, xt_vmax=0.3):
    left_color = left_color or THEME["team_a"]
    right_color = right_color or THEME["team_b"]

    events = sb.events(match_id=match_id)
    short_names = {**auto_short_names(match_id), **KNOWN_NAMES, **(short_names or {})}

    # some matches lack a column entirely (e.g. no carries): add it empty
    for c in ["pass_cross", "pass_type", "carry_end_location", "duel_type",
              "duel_outcome", "shot_type", "substitution_replacement"]:
        if c not in events.columns:
            events[c] = np.nan

    # stop early, with a clear message, if a team name is wrong
    for t in (left, right):
        if t not in set(events["team"]):
            raise ValueError(f"'{t}' not found. Teams here: "
                             f"{sorted(events['team'].dropna().unique())}")

    ink, graphite = THEME["ink"], THEME["graphite"]
    rc = {"font.family": "sans-serif", "font.sans-serif": THEME["font"]}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(29, 29), facecolor=THEME["paper"])
        # column 0 is the left rail (names, score, stats table, legend);
        # columns 1-3 are the maps. Rows: shot-map band, then one block per team.
        gs = fig.add_gridspec(
            8, 4, width_ratios=[0.8, 1, 1, 1],
            height_ratios=[1.4, 0.16, 1, 1, 0.16, 1, 1, 0.1],
            hspace=0.30, wspace=0.07, left=0.025, right=0.985, top=0.985, bottom=0.015)

        # ---- left rail: names, score, stats table, legend
        rail = gs[0:7, 0].subgridspec(3, 1, height_ratios=[1.5, 3.2, 1.6], hspace=0.12)

        ax_head = fig.add_subplot(rail[0])
        ax_head.axis("off")
        ax_head.set_xlim(0, 1)
        ax_head.set_ylim(0, 1)
        ax_head.text(0.0, 0.93, left, ha="left", va="top", fontsize=30,
                     fontweight="bold", color=left_color)
        ax_head.text(0.0, 0.76, right, ha="left", va="top", fontsize=30,
                     fontweight="bold", color=right_color)
        ax_head.text(0.0, 0.50, f"{goals(events, left)}\u2013{goals(events, right)}",
                     ha="left", va="top", fontsize=84, color=ink)
        if subtitle:
            ax_head.text(0.0, 0.10, subtitle, ha="left", va="top",
                         fontsize=13 * FS, color=graphite)
        ax_head.plot([0, 1], [0.0, 0.0], color=ink, lw=1 * LW, clip_on=False)

        # stats table: team colors sit on the column headers, so write short
        # colored labels instead of the full names (they can be long)
        ax_stats = fig.add_subplot(rail[1])
        stats_panel(events, left, right, left_color, right_color, ax=ax_stats,
                    step=0.088, x_left=0.70, show_names=False)
        y_head = 0.5 + 10 * 0.088 / 2 - 0.088 / 2      # the header row of the table
        ax_stats.text(0.70, y_head, "\u25CF", ha="right", va="center", fontsize=20 * FS,
                      color=left_color)
        ax_stats.text(1.0, y_head, "\u25CF", ha="right", va="center", fontsize=20 * FS,
                      color=right_color)

        # shared defensive-map legend, built by hand so it stays neutral
        handles = [Line2D([0], [0], marker=m, linestyle="none", markersize=11,
                          markerfacecolor=face, markeredgecolor=edge, label=kind)
                   for kind, (m, face, edge) in STYLES.items()]
        handles.append(Line2D([0], [0], color=THEME["accent"], lw=2.5 * LW,
                              label="Average height (excl. clearances)"))
        ax_leg = fig.add_subplot(rail[2])
        ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="upper left", fontsize=12 * FS, frameon=False,
                      labelcolor=ink, title="Defensive actions", title_fontsize=12 * FS,
                      alignment="left")

        # ---- top band: shot map (two columns wide) + how-to-read notes
        shot_map(events, left, right, left_color, right_color, short_names,
                 ax=fig.add_subplot(gs[0, 1:3]))
        ax_note = fig.add_subplot(gs[0, 3])
        ax_note.axis("off")
        ax_note.text(0.0, 0.92,
                     "How to read\n\n"
                     "Pass and defensive maps:\nattacking left to right \u2192\n\n"
                     "xT added = value of end zone\nminus start zone; completed\n"
                     "passes and carries, positive\ngains only",
                     ha="left", va="top", fontsize=13 * FS, color=graphite,
                     linespacing=1.6)

        # ---- one block of six maps per team
        blocks = [(left, left_color, 1, 2), (right, right_color, 4, 5)]
        for team, color, label_row, first_row in blocks:
            ax_lab = fig.add_subplot(gs[label_row, 1:4])
            ax_lab.axis("off")
            ax_lab.set_xlim(0, 1)
            ax_lab.set_ylim(0, 1)
            ax_lab.text(0.0, 0.5, team, ha="left", va="center", fontsize=24,
                        fontweight="bold", color=color)
            ax_lab.plot([0, 1], [0.0, 0.0], color=ink, lw=0.8 * LW, clip_on=False)

            r1, r2 = first_row, first_row + 1
            pass_network(events, team, color, short_names, ax=fig.add_subplot(gs[r1, 1]))
            pass_panel(events, team, color, "in_final_third", ax=fig.add_subplot(gs[r1, 2]))
            pass_panel(events, team, color, "into_final_third", ax=fig.add_subplot(gs[r1, 3]))
            pass_panel(events, team, color, "to_box", ax=fig.add_subplot(gs[r2, 1]))
            defensive_actions(events, team, ax=fig.add_subplot(gs[r2, 2]), legend=False)
            xt_pitch(events, team, color, ax=fig.add_subplot(gs[r2, 3]), vmax=xt_vmax)

        # ---- footer
        ax_foot = fig.add_subplot(gs[7, :])
        ax_foot.axis("off")
        ax_foot.set_xlim(0, 1)
        ax_foot.set_ylim(0, 1)
        ax_foot.plot([0, 1], [0.95, 0.95], color=ink, lw=0.6 * LW, clip_on=False)
        ax_foot.text(0.0, 0.4, "Data: StatsBomb open data", ha="left", va="center",
                     fontsize=11 * FS, color=graphite)
        ax_foot.text(1.0, 0.4, "xT grid: Soccermatics method on 2018 World Cup events",
                     ha="right", va="center", fontsize=11 * FS, color=graphite)

        fig.savefig(save_as or f"match_report_{match_id}.png", dpi=130,
                    facecolor=THEME["paper"], bbox_inches="tight")
        plt.show()
