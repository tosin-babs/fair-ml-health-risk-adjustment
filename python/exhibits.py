"""
Figures, in a plain journal style.

Everything is drawn from the CSVs the analysis wrote, so a figure cannot
disagree with its table. Figures carry no "Figure N" label of their own; the
manuscript numbers them in reading order.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 110,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})
P = config.PALETTE
T = config.TABLES

FAMILY = {"WLS": "linear", "Elastic net": "linear", "Constrained WLS": "fair",
          "Tweedie GLM": "glm", "Two-part": "glm",
          "LightGBM (Tweedie)": "gbm", "LightGBM (squared error)": "gbm",
          "Random forest": "gbm", "Neural network": "nn", "CANN": "nn"}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"{name}.{ext}", dpi=config.FIG_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def figure_accuracy():
    t = pd.read_csv(T / "table2_accuracy.csv").sort_values("r2")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    y = np.arange(len(t))
    colours = [P[FAMILY.get(m, "linear")] for m in t["model"]]
    axes[0].hlines(y, t["r2_lo"], t["r2_hi"], color=colours, linewidth=2.2)
    axes[0].scatter(t["r2"], y, color=colours, s=34, zorder=3)
    axes[0].set_yticks(y, t["model"])
    axes[0].set_xlabel("Out-of-sample R-squared (95% interval)")
    axes[0].set_title("Accuracy")
    axes[1].hlines(y, t["cpm_lo"], t["cpm_hi"], color=colours, linewidth=2.2)
    axes[1].scatter(t["cpm"], y, color=colours, s=34, zorder=3)
    axes[1].set_xlabel("Cumming's prediction measure (95% interval)")
    axes[1].set_title("Absolute-error accuracy")
    _save(fig, "fig_accuracy")


def figure_calibration():
    t = pd.read_csv(T / "table2c_calibration.csv")
    show = ["WLS", "Tweedie GLM", "LightGBM (Tweedie)", "CANN"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for m in show:
        s = t[t["model"] == m]
        if s.empty:
            continue
        ax.plot(s["decile"], s["predictive_ratio"], marker="o", markersize=4,
                linewidth=1.8, color=P[FAMILY[m]], label=m,
                linestyle="--" if m in ("Tweedie GLM", "CANN") else "-")
    ax.axhline(1.0, color=P["ink"], linewidth=0.9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of predicted spending")
    ax.set_ylabel("Predictive ratio (predicted / observed)")
    ax.set_title("Calibration across the distribution of predicted spending")
    ax.legend(loc="best")
    _save(fig, "fig_calibration")


def figure_group_compensation():
    t = pd.read_csv(T / "table3_group_fairness.csv")
    show = ["WLS", "Tweedie GLM", "LightGBM (Tweedie)", "CANN"]
    groups = list(dict.fromkeys(t["group"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    offs = np.linspace(-0.27, 0.27, len(show))
    for off, m in zip(offs, show):
        s = t[t["model"] == m].set_index("group").reindex(groups)
        yy = np.arange(len(groups)) + off
        ax.hlines(yy, s["nc_lo"], s["nc_hi"], color=P[FAMILY[m]], linewidth=1.6)
        ax.scatter(s["nc"], yy, color=P[FAMILY[m]], s=22, zorder=3, label=m)
    ax.axvline(0, color=P["ink"], linewidth=0.9)
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlabel("Net compensation, $ per person-year (predicted minus observed; 95% interval)")
    ax.set_title("Which groups each model under- or over-pays")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_group_compensation")


def figure_frontier():
    t = pd.read_csv(T / "table3b_frontier.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, colour in (("Penalized WLS", P["linear"]), ("Stacked LightGBM", P["gbm"])):
        s = t[t["method"] == method].sort_values("lambda")
        ax.plot(s["max_abs_nc_target"], s["r2"], marker="o", markersize=4,
                color=colour, linewidth=1.8, label=f"{method}, increasing lambda")
    for method, colour, marker in (("Constrained WLS", P["linear"], "s"),
                                   ("Stacked LightGBM, constrained", P["gbm"], "s"),
                                   ("LightGBM, decile recalibration", P["fair"], "D")):
        s = t[t["method"] == method]
        if len(s):
            ax.scatter(s["max_abs_nc_target"], s["r2"], marker=marker, s=55,
                       color=colour, edgecolor=P["ink"], zorder=4, label=method)
    ax.set_xlabel("Largest absolute net compensation among target groups ($)")
    ax.set_ylabel("Out-of-sample R-squared")
    ax.set_title("The price of fairness: accuracy against the largest group payment gap")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_frontier")


def figure_importance():
    t = pd.read_csv(T / "table4_importance.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(t))
    ax.barh(y, t["mean_abs_shap"], xerr=t["sd_across_folds"], color=P["gbm"],
            alpha=0.85, error_kw=dict(ecolor=P["ink"], lw=0.8, capsize=2))
    ax.set_yticks(y, [str(s)[:48] for s in t["label"]])
    ax.set_xlabel("Mean absolute SHAP value, $ (bar: across-fold SD)")
    ax.set_title("What drives the boosted model's predictions (squared-error LightGBM)")
    _save(fig, "fig_importance")


def figure_gameability():
    acc = pd.read_csv(T / "table2_accuracy.csv").set_index("model")["r2"]
    fr = pd.read_csv(T / "table3b_frontier.csv")
    cw = fr[fr["method"] == "Constrained WLS"]
    if len(cw):
        acc.loc["Constrained WLS"] = float(cw["r2"].iloc[0])
    c = pd.read_csv(T / "table5_coding_sensitivity.csv")
    c = c[(c["pool"] == "chronic") & (c["share_affected"] == 0.10) & (c["codes_added"] == 1)]
    c = c.assign(r2=c["model"].map(acc)).dropna(subset=["r2"])
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for _, r in c.iterrows():
        ax.scatter(r["r2"], r["dollars_per_added_code"], s=60,
                   color=P[FAMILY.get(r["model"], "linear")], edgecolor=P["ink"], zorder=3)
        ax.annotate(r["model"], (r["r2"], r["dollars_per_added_code"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Out-of-sample R-squared")
    ax.set_ylabel("Rise in predicted spending per added chronic code ($)")
    ax.set_title("Accuracy against exposure to coding intensity")
    _save(fig, "fig_gameability")


def main():
    print("=== figures ===")
    for f in (figure_accuracy, figure_calibration, figure_group_compensation,
              figure_frontier, figure_importance, figure_gameability):
        try:
            f()
        except FileNotFoundError as e:
            print(f"  skipped {f.__name__}: {e.filename} not built yet")


if __name__ == "__main__":
    main()
