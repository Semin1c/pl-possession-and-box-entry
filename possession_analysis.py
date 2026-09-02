"""
What possession actually buys you in the Premier League
======================================================
Eight seasons of FBref squad and player data, 2017-18 to 2024-25, asking which
possession metrics carry information about how many chances a side creates.

    python3 possession_analysis.py

Reads data/merged_squad_data.csv (160 squad-seasons) and
      data/merged_player_data.csv (4,343 player-seasons)

  figures/what_predicts_chances.png   families of possession metrics, ranked by
                                      how much of xG+xAG they explain
  figures/one_variable_vs_all.png     touches in the box against the full metric
                                      suite, under three cross-validation schemes
  figures/sterile_possession.png      possession share against box entry, and the
                                      clubs that convert one into the other worst
  figures/minutes_filter.png          why the player-level R2 depends almost
                                      entirely on the minutes cut-off
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PURPLE, GREEN, GREY, LGREY = "#3D195B", "#00A87E", "#B9BEC4", "#ECEEF0"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlecolor": PURPLE, "axes.labelcolor": PURPLE, "text.color": PURPLE,
    "xtick.color": PURPLE, "ytick.color": PURPLE, "axes.titlepad": 12,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": GREY,
})
os.makedirs("figures", exist_ok=True)


def strip(ax, keep_left=True):
    for s in ["top", "right"] + ([] if keep_left else ["left"]):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def pipe():
    return make_pipeline(StandardScaler(), LinearRegression())


squad = pd.read_csv("data/merged_squad_data.csv", low_memory=False)
TARGET = "xG+xAG"
y = squad[TARGET]

# Every possession column, minus the identifiers and the xG terms that make up
# the target itself.
ALL = [c for c in squad.columns
       if c not in ["Squad", "Season", "xG.1", "xAG.1", TARGET, "Age"]]

# Grouped by what the metric is actually measuring.
BLOCKS = {
    "Attacking territory\nAtt 3rd, Att Pen, carries and passes into it":
        ["Att 3rd", "Att Pen", "CPA", "1/3", "PrgR", "PrgC", "PrgDist"],
    "Possession volume\npossession %, touches, carries, distance":
        ["Poss", "Touches", "Live", "Carries", "TotDist", "Rec"],
    "Own-half touches\ndefensive and middle third":
        ["Def Pen", "Def 3rd", "Mid 3rd"],
    "Take-ons\nattempted, completed, success rate":
        ["Att", "Succ", "Succ%", "Tkld", "Tkld%"],
    "Losing the ball, counted\nmiscontrols and dispossessions":
        ["Mis", "Dis"],
    "Losing the ball, as a rate\nthe same losses, per touch":
        ["loss_rate"],
}

# Losing the ball is only meaningful relative to how often you have it. A side
# with 65% possession touches the ball roughly twice as often as one with 35%,
# so a raw count of miscontrols compares two different things.
squad["losses"] = squad["Mis"] + squad["Dis"]
squad["loss_rate"] = squad["losses"] / squad["Touches"]

kf = KFold(5, shuffle=True, random_state=42)
print(f"{len(squad)} squad-seasons, {squad.Season.nunique()} seasons, "
      f"{squad.Squad.nunique()} clubs")
print(f"target: {TARGET} per 90, mean {y.mean():.2f}, sd {y.std():.2f}\n")

print("cross-validated R2 by family of metric")
block_scores = {}
for name, cols in BLOCKS.items():
    block_scores[name] = cross_val_score(pipe(), squad[cols], y, cv=kf, scoring="r2").mean()
    print(f"  {name.splitlines()[0]:22s} ({len(cols):2d}) R2 = {block_scores[name]:.3f}")
solo = cross_val_score(pipe(), squad[["Att Pen"]], y, cv=kf, scoring="r2").mean()
full = cross_val_score(pipe(), squad[ALL], y, cv=kf, scoring="r2").mean()
poss = cross_val_score(pipe(), squad[["Poss"]], y, cv=kf, scoring="r2").mean()
print(f"\n  touches in the box, alone  R2 = {solo:.3f}")
print(f"  possession %, alone        R2 = {poss:.3f}")
print(f"  all {len(ALL)} metrics together   R2 = {full:.3f}\n")

# How much of the rate version is genuinely new, and how much is another way of
# measuring the same thing? Very little is new.
lr_alone = cross_val_score(pipe(), squad[["loss_rate"]], y, cv=kf, scoring="r2").mean()
lr_plus = cross_val_score(pipe(), squad[["Att Pen", "loss_rate"]], y, cv=kf, scoring="r2").mean()
best = squad.nsmallest(1, "loss_rate").iloc[0]
worst = squad.nlargest(1, "loss_rate").iloc[0]
print("losing the ball, counted versus as a rate")
print(f"  raw miscontrols + dispossessions   R2 = {block_scores[[k for k in BLOCKS if k.startswith('Losing the ball, counted')][0]]:.3f}")
print(f"  the same losses per touch          R2 = {lr_alone:.3f} "
      f"(correlation with chances created {np.corrcoef(squad.loss_rate, y)[0, 1]:+.3f})")
print(f"  on top of box touches              R2 = {lr_plus:.3f}, so it adds "
      f"{lr_plus - solo:+.3f} once box entry is known")
print(f"  best retention  {best.Squad} {best.Season[:4]}: "
      f"{best.loss_rate * 100:.2f} losses per 100 touches")
print(f"  worst retention {worst.Squad} {worst.Season[:4]}: "
      f"{worst.loss_rate * 100:.2f} losses per 100 touches\n")

# If the rate is right for losses, is it right for everything? No, and the split
# is intuitive: things you want LESS of only mean something as a rate, because the
# count just tracks how much you had the ball. Things you want MORE of are about
# volume, and dividing by touches throws that away.
EVENT_COUNTS = ["Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
                "Live", "Att", "Succ", "Tkld", "Carries", "TotDist", "PrgDist",
                "PrgC", "1/3", "CPA", "Mis", "Dis", "Rec", "PrgR"]
norm = []
for c in EVENT_COUNTS:
    squad["_rate"] = squad[c] / squad["Touches"]
    raw_r2 = cross_val_score(pipe(), squad[[c]], y, cv=kf, scoring="r2").mean()
    rate_r2 = cross_val_score(pipe(), squad[["_rate"]], y, cv=kf, scoring="r2").mean()
    norm.append((c, raw_r2, rate_r2, rate_r2 - raw_r2))
squad.drop(columns="_rate", inplace=True)
helps = [n for n in norm if n[3] > 0.05]
hurts = [n for n in norm if n[3] < -0.05]
print("should every count be divided by touches, the way losses were?")
print(f"  dividing helps {len(helps)}, hurts {len(hurts)}, changes little "
      f"{len(norm) - len(helps) - len(hurts)}")
print("  helps (things you want LESS of):")
for c, a, b, d in sorted(helps, key=lambda t: -t[3]):
    print(f"    {c:9s} raw {a:+.3f} -> per touch {b:+.3f}")
print("  hurts most (things you want MORE of, where volume is the point):")
for c, a, b, d in sorted(hurts, key=lambda t: t[3])[:5]:
    print(f"    {c:9s} raw {a:+.3f} -> per touch {b:+.3f}")
print()

# ---------------------------------------------------------------------------
# 1. Which families carry information
# ---------------------------------------------------------------------------
order = sorted(block_scores, key=block_scores.get)
fig, ax = plt.subplots(figsize=(10.6, 5.2))
vals = [block_scores[k] for k in order]
colors = [GREEN if k.startswith("Attacking territory") else GREY for k in order]
ax.barh(range(len(order)), vals, 0.64, color=colors, zorder=3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=10, linespacing=1.45)
ax.set_xlim(0, 1.02)
ax.set_xlabel("Cross-validated R², explaining chances created (xG+xAG per 90)", fontsize=11)
ax.set_title("How much each family of possession metrics explains", loc="left")
ax.xaxis.grid(True, color=LGREY, zorder=0)
strip(ax, keep_left=False)
for i, v in enumerate(vals):
    ax.text(v + 0.014, i, f"{v:.3f}", va="center", fontsize=12, fontweight="bold",
            color=colors[i] if colors[i] == GREEN else PURPLE)
fig.tight_layout()
fig.savefig("figures/what_predicts_chances.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. One variable against the whole suite, under three CV schemes
# ---------------------------------------------------------------------------
schemes = [
    ("Random folds", KFold(5, shuffle=True, random_state=42), None),
    ("Held-out clubs", GroupKFold(5), squad.Squad),
    ("Held-out seasons", GroupKFold(5), squad.Season),
]
rows = []
print("does it survive being tested on clubs and seasons the model never saw?")
for label, cvobj, groups in schemes:
    a = cross_val_score(pipe(), squad[["Poss"]], y, cv=cvobj, groups=groups, scoring="r2").mean()
    b = cross_val_score(pipe(), squad[["Att Pen"]], y, cv=cvobj, groups=groups, scoring="r2").mean()
    c = cross_val_score(pipe(), squad[ALL], y, cv=cvobj, groups=groups, scoring="r2").mean()
    rows.append((label, a, b, c))
    print(f"  {label:17s} possession % {a:.3f} | box touches {b:.3f} | all {len(ALL)} {c:.3f}")
print()

fig, ax = plt.subplots(figsize=(10.8, 4.8))
x = np.arange(len(rows))
w = 0.26
series = [("Possession %", [r[1] for r in rows], GREY),
          ("Touches in the box", [r[2] for r in rows], GREEN),
          (f"All {len(ALL)} metrics", [r[3] for r in rows], PURPLE)]
for i, (name, vals, col) in enumerate(series):
    ax.bar(x + (i - 1) * w, vals, w, color=col, zorder=3, label=name)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10.5,
                fontweight="bold", color=col)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=11.5)
ax.set_ylim(0, 1.02)
ax.set_ylabel("Cross-validated R²", fontsize=11)
ax.set_title("One variable against the full metric suite", loc="left")
ax.yaxis.grid(True, color=LGREY, zorder=0)
strip(ax)
ax.legend(frameon=False, fontsize=10.5, loc="upper right", ncol=3)
fig.tight_layout()
fig.savefig("figures/one_variable_vs_all.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Sterile possession: holding the ball without reaching the box
# ---------------------------------------------------------------------------
fit = sm.OLS(squad["Att Pen"], sm.add_constant(squad[["Poss"]])).fit()
squad["box_gap"] = fit.resid
squad["label"] = squad.Squad + " " + squad.Season.str[:4]
plain = sm.OLS(y, sm.add_constant(squad[["Poss"]])).fit()
plus = sm.OLS(y, sm.add_constant(squad[["Poss", "box_gap"]])).fit()
print("sterile possession")
print(f"  box touches ~ possession %      R2 = {fit.rsquared:.3f}")
print(f"  xG+xAG ~ possession %           R2 = {plain.rsquared:.3f}")
print(f"  xG+xAG ~ possession % + box gap R2 = {plus.rsquared:.3f} "
      f"(gap p = {plus.pvalues.box_gap:.1e})")
# Being straight about what that is: the gap is the part of box entry possession
# does not explain, so possession + gap spans the same space as possession +
# box touches. It is a decomposition of one variable, not a second one.
same = sm.OLS(y, sm.add_constant(squad[["Poss", "Att Pen"]])).fit()
print(f"  xG+xAG ~ possession % + box touches R2 = {same.rsquared:.3f} "
      f"<- identical, as it must be: this is a decomposition, not a new predictor")

# Is the gap a stable club trait or does it bounce around season to season?
between = 1 - (squad.groupby("Squad").box_gap.transform(lambda g: g - g.mean()).var(ddof=0)
               / squad.box_gap.var(ddof=0))
lagged = squad.sort_values(["Squad", "Season"]).copy()
lagged["next"] = lagged.groupby("Squad").box_gap.shift(-1)
lagged = lagged.dropna(subset=["next"])
persist = np.corrcoef(lagged.box_gap, lagged["next"])[0, 1]
print(f"\n  is the gap persistent? {between:.1%} of its variance is a club trait; "
      f"a club's gap correlates r = {persist:.2f} with its own next season (n = {len(lagged)})")
spurs = squad[squad.Squad == "Tottenham"].sort_values("Season")
print("  Tottenham, season by season:")
print("   ", {r.Season[:5] + r.Season[7:]: round(r.box_gap, 1) for r in spurs.itertuples()})
worst = squad.nsmallest(6, "box_gap")
best = squad.nlargest(6, "box_gap")
print("\n  least box entry for the ball they had:")
print(worst[["label", "Poss", "Att Pen", TARGET]].round(2).to_string(index=False))
print("\n  most:")
print(best[["label", "Poss", "Att Pen", TARGET]].round(2).to_string(index=False), "\n")

fig, ax = plt.subplots(figsize=(10.4, 6.6))
ax.scatter(squad.Poss, squad["Att Pen"], s=42, color=GREY, alpha=0.75,
           edgecolor="white", linewidth=0.6, zorder=3)
xs = np.linspace(squad.Poss.min(), squad.Poss.max(), 50)
ax.plot(xs, fit.params.const + fit.params.Poss * xs, color=PURPLE, lw=1.6, ls="--",
        zorder=4, label="what possession share alone predicts")
for df, col, base in [(worst, "#C2185B", -1), (best, GREEN, 1)]:
    ax.scatter(df.Poss, df["Att Pen"], s=62, color=col, zorder=5,
               edgecolor="white", linewidth=0.8)
    # Several of these sit almost on top of each other, so fan the labels out
    # left and right rather than stacking them at the same offset.
    placed = df.sort_values("Poss").reset_index(drop=True)
    for i, r in placed.iterrows():
        side = -1 if i % 2 == 0 else 1
        ax.annotate(r.label, (r.Poss, r["Att Pen"]), fontsize=8.8, color=col,
                    fontweight="bold",
                    xytext=(side * 16, base * (11 + 9 * (i % 3))),
                    textcoords="offset points",
                    ha="right" if side < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=0, shrinkB=3, alpha=0.6))
ax.set_xlabel("Share of possession (%)", fontsize=11)
ax.set_ylabel("Touches in the opposition penalty area, per 90", fontsize=11)
ax.set_title("Possession share against box entry, 2017-18 to 2024-25", loc="left")
ax.grid(True, color=LGREY, zorder=0)
strip(ax)
ax.legend(frameon=False, fontsize=10.5, loc="upper left")
ax.text(0.985, 0.05, "green: most box entry for their possession\n"
                     "pink: least", transform=ax.transAxes, ha="right", fontsize=10,
        color=PURPLE, linespacing=1.5)
fig.tight_layout()
fig.savefig("figures/sterile_possession.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. The player-level model is a story about the minutes filter
# ---------------------------------------------------------------------------
players = pd.read_csv("data/merged_player_data.csv", low_memory=False)
PFEATS = [c for c in players.columns
          if c not in ["Player", "Nation", "Pos", "Season", "Squad", "xG.1", "xAG.1",
                       TARGET, "Starts", "Min", "90s", "Age"]]
thresholds = [0, 1, 2, 3, 5, 8, 10, 15]
curve = []
print("player-level model, by minimum minutes played")
for thr in thresholds:
    q = players[players["90s"] >= thr].dropna(subset=PFEATS + [TARGET])
    xtr, xte, ytr, yte = train_test_split(q[PFEATS], q[TARGET], test_size=0.25,
                                          random_state=42)
    r2 = r2_score(yte, pipe().fit(xtr, ytr).predict(xte))
    curve.append((thr, len(q), r2))
    print(f"  90s >= {thr:<3d} n = {len(q):4d}   R2 = {r2:.3f}")
noisy = players[players["90s"] < 1]
print(f"\n  the {len(noisy)} players under one full match have per-90 values up to "
      f"{noisy[TARGET].max():.1f}, against a league average of "
      f"{players[players['90s'] >= 5][TARGET].mean():.2f}\n")

fig, ax = plt.subplots(figsize=(10.2, 4.9))
xs = [c[0] for c in curve]
r2s = [c[2] for c in curve]
ax.plot(xs, r2s, marker="o", color=GREEN, lw=2.4, markersize=8, zorder=4)
ax.fill_between(xs, 0, r2s, color=GREEN, alpha=0.10, zorder=2)
ax.set_xlabel("Minimum full matches played to be included", fontsize=11)
ax.set_ylabel("R² on held-out players", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("The player-level result against the minutes cut-off", loc="left")
ax.grid(True, color=LGREY, zorder=0)
strip(ax)
for xi, r2 in zip(xs, r2s):
    ax.annotate(f"{r2:.2f}", (xi, r2), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=10.5, fontweight="bold", color=GREEN)
ax.annotate("keeping every cameo appearance,\nper-90 rates are mostly noise",
            xy=(0.05, r2s[0]), xytext=(2.1, 0.38), fontsize=10.5, color=PURPLE,
            linespacing=1.5, arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
fig.tight_layout()
fig.savefig("figures/minutes_filter.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. What survives a transfer -- the recruitment question
# ---------------------------------------------------------------------------
# A club signing a player is betting that a number follows him. Test that
# directly: for players who changed Premier League club between consecutive
# seasons, how well does each metric carry across the move?
pl = players[players["90s"] >= 5].copy()
pl["pos"] = pl["Pos"].astype(str).str.split(",").str[0].str.strip()
pl = pl[pl["pos"].isin(["DF", "MF", "FW"])].copy()
pl["yr"] = pl["Season"].str[:4].astype(int)

LABELS = {
    "Att 3rd": "Touches in the final third", "PrgR": "Progressive passes received",
    "PrgC": "Progressive carries", "Succ": "Take-ons completed",
    "CPA": "Carries into the box", "Att Pen": "Touches in the box",
    "xAG.1": "Expected assists", "xG+xAG": "Expected goals + assists",
    "Succ%": "Take-on success rate",
}
BEHAVIOUR = {"Att 3rd", "PrgR", "PrgC", "Succ", "CPA", "Att Pen"}
boot_rng = np.random.default_rng(5)


def transfer_pairs(metric, adjust=True):
    """One row per player who moved club, with the metric before and after."""
    dd = pl.dropna(subset=[metric]).copy()
    # Judge a player against same-position team-mates, so a move between a strong
    # and a weak squad does not masquerade as the player changing.
    dd["v"] = (dd[metric] - dd.groupby(["Squad", "Season", "pos"])[metric].transform("mean")
               if adjust else dd[metric])
    cur = dd[["Player", "yr", "Squad", "v"]]
    nxt = cur.assign(yr=lambda t: t.yr - 1)
    both = cur.merge(nxt, on=["Player", "yr"], suffixes=("", "_next"))
    return both[both.Squad != both.Squad_next]


def r_with_ci(pairs):
    r = np.corrcoef(pairs.v, pairs.v_next)[0, 1]
    draws = [np.corrcoef(s.v, s.v_next)[0, 1]
             for s in (pairs.iloc[boot_rng.integers(0, len(pairs), len(pairs))]
                       for _ in range(3000))]
    return r, *np.percentile(draws, [2.5, 97.5])


transfer = {m: r_with_ci(transfer_pairs(m)) for m in LABELS}
n_moves = len(transfer_pairs("Att 3rd"))
print(f"what survives a transfer ({n_moves} players who changed Premier League club)")
for m, (r, lo, hi) in sorted(transfer.items(), key=lambda kv: -kv[1][0]):
    print(f"  {LABELS[m]:30s} r = {r:.3f}  [{lo:.2f}, {hi:.2f}]")

order5 = sorted(transfer, key=lambda m: transfer[m][0])
fig, ax = plt.subplots(figsize=(11.0, 5.6))
ys = np.arange(len(order5))
vals = [transfer[m][0] for m in order5]
errs = np.array([[transfer[m][0] - transfer[m][1] for m in order5],
                 [transfer[m][2] - transfer[m][0] for m in order5]])
cols5 = [GREEN if m in BEHAVIOUR else PURPLE for m in order5]
ax.barh(ys, vals, 0.62, color=cols5, zorder=3)
ax.errorbar(vals, ys, xerr=errs, fmt="none", ecolor="#55606B", elinewidth=1.3,
            capsize=3.5, zorder=5)
ax.set_yticks(ys)
ax.set_yticklabels([LABELS[m] for m in order5], fontsize=10.5)
ax.set_xlim(0, 0.92)
ax.set_xlabel("Correlation between a player's metric before and after changing club",
              fontsize=11)
ax.set_title("What follows a player to a new club", loc="left")
ax.xaxis.grid(True, color=LGREY, zorder=0)
strip(ax, keep_left=False)
for yy, m in zip(ys, order5):
    ax.text(transfer[m][2] + 0.018, yy, f"{transfer[m][0]:.2f}", va="center",
            fontsize=11.5, fontweight="bold",
            color=GREEN if m in BEHAVIOUR else PURPLE)
ax.text(0.985, 0.11, "green: what a player does\npurple: what it produced",
        transform=ax.transAxes, ha="right", fontsize=10.5, color=PURPLE, linespacing=1.6)
fig.tight_layout()
fig.savefig("figures/what_survives_a_transfer.png", dpi=200)
plt.close(fig)

# The same metric looks far more repeatable before position and squad are removed.
steps = ["as measured", "vs same position", "vs position team-mates", "across a transfer"]
decay = {}
for m in ["Att 3rd", "xG+xAG"]:
    dd = pl.dropna(subset=[m]).copy()
    series = []
    for mode in ["raw", "pos", "team"]:
        if mode == "raw":
            dd["v"] = dd[m]
        elif mode == "pos":
            dd["v"] = dd[m] - dd.groupby("pos")[m].transform("mean")
        else:
            dd["v"] = dd[m] - dd.groupby(["Squad", "Season", "pos"])[m].transform("mean")
        cur = dd[["Player", "yr", "v"]]
        both = cur.merge(cur.assign(yr=lambda t: t.yr - 1), on=["Player", "yr"],
                         suffixes=("", "_next"))
        series.append(np.corrcoef(both.v, both.v_next)[0, 1])
    series.append(transfer[m][0])
    decay[m] = series
print("\n  how repeatability falls as context is stripped out:")
for m, v in decay.items():
    print(f"    {LABELS[m]:28s} " + "  ".join(f"{s}={r:.2f}" for s, r in zip(steps, v)))

fig, ax = plt.subplots(figsize=(10.4, 4.9))
for m, col in [("Att 3rd", GREEN), ("xG+xAG", PURPLE)]:
    ax.plot(range(4), decay[m], marker="o", markersize=8, lw=2.4, color=col,
            label=LABELS[m], zorder=4)
    for i, v in enumerate(decay[m]):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 11 if col == GREEN else -20),
                    textcoords="offset points", ha="center", fontsize=10.5,
                    fontweight="bold", color=col)
ax.set_xticks(range(4))
ax.set_xticklabels(steps, fontsize=10.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Correlation with the player's next season", fontsize=11)
ax.set_title("Repeatability, as position and squad are taken out", loc="left")
ax.grid(True, color=LGREY, zorder=0)
strip(ax)
ax.legend(frameon=False, fontsize=10.5, loc="lower left")
fig.tight_layout()
fig.savefig("figures/repeatability_decay.png", dpi=200)
plt.close(fig)

print("\nwrote figures/what_predicts_chances.png, one_variable_vs_all.png,")
print("      sterile_possession.png, minutes_filter.png,")
print("      what_survives_a_transfer.png, repeatability_decay.png")
