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
    "pitch":    "#B4BFBA",     # pitch lines: quieter than the data
    "team_a":   "#3581B8",     # left team (blue)
    "team_b":   "#C33149",     # right team (crimson)
    "accent":   "#499167",     # the one highlight (green)
    "font":     ["Roboto", "Franklin Gothic Medium", "Segoe UI", "Arial", "DejaVu Sans"],
}

# One-knob sizing: raise these to make text and lines bigger everywhere.
FS = 1.4    # font scale
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
try:
    XT16 = np.load(Path(__file__).parent / "xt16.npy")
except FileNotFoundError:
    XT16 = None          # only needed for the optional xT map


# ----------------------------------------------------- small helpers

def _canvas(ax, **pitch_kwargs):
    """A themed pitch: draws on a new figure, or on the ax it is given."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color=THEME["paper"],
                  line_color=THEME["pitch"], linewidth=1.2, **pitch_kwargs)
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 7))
        fig.set_facecolor(THEME["paper"])
    else:
        pitch.draw(ax=ax)
    return pitch, ax


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=12 * FS, color=THEME["graphite"], pad=10)


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

    _title(ax, f"xT added by start zone  \u00b7  total {m['xt'].sum():.2f}")
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


def pass_panel(events, team, color, zone, ax=None, max_arrows=20):
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

    # misses: a small stone dot where the pass ended, no line
    miss, hit = p[~done], p[done]
    pitch.scatter(miss["end_x"], miss["end_y"], s=28, color=THEME["stone"],
                  alpha=0.9, zorder=1, ax=ax)

    # completed passes: only the most forward ones when there are too many
    shown = hit.assign(gain=hit["end_x"] - hit["x"]).nlargest(max_arrows, "gain")
    pitch.arrows(shown["x"], shown["y"], shown["end_x"], shown["end_y"],
                 color=color, alpha=0.8, width=1.8 * LW, headwidth=4,
                 headlength=4, zorder=2, ax=ax)

    _title(ax, f"{title}  \u00b7  {done.sum()}/{len(p)} complete")
    return ax


# ------------------------------------------------------- pass network

def pass_network(events, team, color, short_names=None, min_passes=5, max_links=14, ax=None):
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
    pairs = pairs[pairs["count"] >= min_passes].nlargest(max_links, "count")

    pitch, ax = _canvas(ax)
    _title(ax, "Pass network")

    for _, row in pairs.iterrows():
        a = row["player"]
        b = row["pass_recipient"]
        if a in avg.index and b in avg.index:
            pitch.lines(avg.loc[a, "x"], avg.loc[a, "y"],
                        avg.loc[b, "x"], avg.loc[b, "y"],
                        lw=2 + row["count"] / pairs["count"].max() * 9,
                        color=color, alpha=0.35, zorder=1, ax=ax)

    colors = [THEME["stone"] if name in came_on else color for name in avg.index]
    pitch.scatter(avg["x"], avg["y"], s=avg["n"] * 16, c=colors,
                  edgecolors=THEME["paper"], linewidth=1.5 * LW, alpha=0.95,
                  zorder=2, ax=ax)

    # labels: try above the dot, then below, then further out, skipping any
    # spot that would overlap a label already placed
    placed = []
    for name, row in avg.sort_values("n", ascending=False).iterrows():
        label = short_names.get(name, name.split()[-1])
        for dy, va in [(-3.5, "bottom"), (3.5, "top"), (-11, "bottom"), (11, "top")]:
            ly = row["y"] + dy
            half = len(label) * 1.4
            if all(abs(row["x"] - px) > half + ph or abs(ly - py) > 7.5
                   for px, py, ph in placed):
                break
        placed.append((row["x"], ly, half))
        ax.text(row["x"], ly, label, ha="center", va=va, fontsize=11 * FS,
                fontweight="bold", color=THEME["ink"], path_effects=_halo(), zorder=4)

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
        size = s["shot_statsbomb_xg"] * 1800

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

    ink = THEME["ink"]
    ax.set_title(f"{totals[left_team]:.2f} xG", loc="left", fontsize=16 * FS,
                 fontweight="bold", color=left_color, pad=10)
    ax.set_title(f"{totals[right_team]:.2f} xG", loc="right", fontsize=16 * FS,
                 fontweight="bold", color=right_color, pad=10)
    ax.set_title("Shot map  \u00b7  size = xG", loc="center", fontsize=12 * FS,
                 color=THEME["graphite"], pad=10)
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
            f"{n_pass[t] / sum(n_pass.values()) * 100:.0f}%",
            f"{n_ft[t] / sum(n_ft.values()) * 100:.0f}%",
            f"{ppda(events, t):.1f}",
        ]

    labels = ["Shots", "xG", "xG per shot", "Possession", "Field tilt", "PPDA"]

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
    "Interception":  ("D", THEME["ink"], THEME["ink"]),
    "Won tackle":    ("o", THEME["ink"], THEME["ink"]),
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
    d = d[d["kind"].isin(list(STYLES))]

    pitch, ax = _canvas(ax)

    for kind, (marker, face, edge) in STYLES.items():
        k = d[d["kind"] == kind]
        pitch.scatter(k["x"], k["y"], marker=marker, s=90, facecolors=face,
                      edgecolors=edge, linewidths=1 * LW, label=kind, zorder=3, ax=ax)

    avg_x = d.loc[d["kind"] != "Clearance", "x"].mean()
    ax.axvline(avg_x, color=THEME["accent"], lw=2.5 * LW, zorder=2,
               label="Average height (excl. clearances)")
    ax.text(avg_x, 84, f"{avg_x / 120 * 100:.1f}% up the pitch",
            fontsize=11 * FS, ha="center", va="top", color=THEME["ink"])

    if legend:
        ax.legend(loc="lower right", fontsize=8 * FS, framealpha=0.9)
    _title(ax, f"Defensive actions  \u00b7  {len(d)}")
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
    rc = {"font.family": "sans-serif", "font.sans-serif": THEME["font"],
          "font.weight": "medium", "axes.titleweight": "medium"}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(24, 10.5), facecolor=THEME["paper"])
        # Left rail: score, stats, shot map. Right: three maps per team, each
        # block headed by a thin rule in the team's color (no names needed).
        gs = fig.add_gridspec(
            5, 4, width_ratios=[1.25, 1, 1, 1],
            height_ratios=[0.04, 1, 0.04, 1, 0.16],
            hspace=0.22, wspace=0.07, left=0.02, right=0.985, top=0.98, bottom=0.015)

        # ---- left rail
        rail = gs[0:4, 0].subgridspec(3, 1, height_ratios=[1.3, 2.2, 2.6], hspace=0.12)

        ax_head = fig.add_subplot(rail[0])
        ax_head.axis("off")
        ax_head.set_xlim(0, 1)
        ax_head.set_ylim(0, 1)
        ax_head.text(0.0, 0.95, left, ha="left", va="top", fontsize=26,
                     fontweight="bold", color=left_color)
        ax_head.text(0.0, 0.66, right, ha="left", va="top", fontsize=26,
                     fontweight="bold", color=right_color)
        ax_head.text(1.0, 0.72, f"{goals(events, left)}\u2013{goals(events, right)}",
                     ha="right", va="center", fontsize=64, fontweight="bold", color=ink)
        if subtitle:
            ax_head.text(0.0, 0.22, subtitle, ha="left", va="top",
                         fontsize=13 * FS, color=graphite)

        ax_stats = fig.add_subplot(rail[1])
        step = 0.13
        stats_panel(events, left, right, left_color, right_color, ax=ax_stats,
                    step=step, x_left=0.72, show_names=False)
        y_head = 0.5 + 7 * step / 2 - step / 2          # header row of the table
        ax_stats.scatter([0.69], [y_head], s=420, color=left_color, clip_on=False)
        ax_stats.scatter([0.97], [y_head], s=420, color=right_color, clip_on=False)

        shot_map(events, left, right, left_color, right_color, short_names,
                 ax=fig.add_subplot(rail[2]))

        # ---- three maps per team
        for team, color, rule_row, map_row in [(left, left_color, 0, 1),
                                               (right, right_color, 2, 3)]:
            ax_rule = fig.add_subplot(gs[rule_row, 1:4])
            ax_rule.axis("off")
            ax_rule.set_xlim(0, 1)
            ax_rule.set_ylim(0, 1)
            ax_rule.plot([0, 1], [0.5, 0.5], color=color, lw=6, solid_capstyle="butt",
                         clip_on=False)
            pass_network(events, team, color, short_names, ax=fig.add_subplot(gs[map_row, 1]))
            pass_panel(events, team, color, "into_final_third", ax=fig.add_subplot(gs[map_row, 2]))
            defensive_actions(events, team, ax=fig.add_subplot(gs[map_row, 3]), legend=False)

        # ---- footer: one quiet line
        ax_foot = fig.add_subplot(gs[4, :])
        ax_foot.axis("off")
        ax_foot.set_xlim(0, 1)
        ax_foot.set_ylim(0, 1)
        handles = [Line2D([0], [0], marker=m, linestyle="none", markersize=10,
                          markerfacecolor=face, markeredgecolor=edge, label=kind)
                   for kind, (m, face, edge) in STYLES.items()]
        handles.append(Line2D([0], [0], color=THEME["accent"], lw=2.5 * LW,
                              label="Average height"))
        ax_foot.legend(handles=handles, loc="center left", ncol=4, frameon=False,
                       fontsize=11 * FS, labelcolor=ink, columnspacing=1.6,
                       handletextpad=0.4)
        ax_foot.text(1.0, 0.5, "Attacking left to right   \u00b7   StatsBomb open data",
                     ha="right", va="center", fontsize=11 * FS, color=graphite)

        fig.savefig(save_as or f"match_report_{match_id}.png", dpi=130,
                    facecolor=THEME["paper"], bbox_inches="tight")
        plt.show()
