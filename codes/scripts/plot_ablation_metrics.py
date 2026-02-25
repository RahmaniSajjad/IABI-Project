#!/usr/bin/env python3
"""
نمودارهای تحلیل پایداری مدالیتی بر اساس ablation_out/metrics_summary.csv
خروجی: PNG در ablation_out/figures/
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_DIR, "ablation_out", "metrics_summary.csv")
FIG_DIR = os.path.join(PROJECT_DIR, "ablation_out", "figures")

MOD_ORDER = ["flair", "t1ce", "t1", "t2"]
FIXED_REP = ["black", "gaussian", "mean", "interp"]


def load_data():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ["dice_TC", "dice_WT", "dice_ET", "sensitivity_TC", "sensitivity_WT", "sensitivity_ET", "hd95_TC", "hd95_WT", "hd95_ET"]:
                row[k] = float(row[k])
            rows.append(row)
    single = [r for r in rows if r["removed_modality"] in MOD_ORDER]
    baseline = [r for r in rows if r["scenario"] == "baseline"]
    return rows, single, baseline


def rep_order(single):
    seen = set()
    for r in single:
        seen.add(r["replacement_type"])
    rest = sorted([x for x in seen if x not in FIXED_REP and x.startswith("copy_")])
    return FIXED_REP + rest


def get_val(single, mod, rep, key):
    for r in single:
        if r["removed_modality"] == mod and r["replacement_type"] == rep:
            return r[key]
    return np.nan


def fig_dice_by_modality_and_replacement(single, baseline):
    if not single or not baseline:
        return
    b = baseline[0]
    baseline_val = (b["dice_TC"] + b["dice_WT"] + b["dice_ET"]) / 3
    rep_ord = rep_order(single)
    rep_short = {"black": "Black", "gaussian": "Gauss", "mean": "Mean", "interp": "Interp"}
    for r in rep_ord:
        if r not in rep_short:
            rep_short[r] = r.replace("copy_", "C.")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MOD_ORDER))
    width = 0.12
    for i, rep in enumerate(rep_ord):
        vals = []
        for mod in MOD_ORDER:
            v = (get_val(single, mod, rep, "dice_TC") + get_val(single, mod, rep, "dice_WT") + get_val(single, mod, rep, "dice_ET")) / 3
            vals.append(v if not np.isnan(v) else np.nan)
        off = (i - len(rep_ord) / 2) * width + width / 2
        ax.bar(x + off, vals, width, label=rep_short.get(rep, rep))
    ax.axhline(y=baseline_val, color="gray", linestyle="--", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(["FLAIR", "T1ce", "T1", "T2"])
    ax.set_ylabel("Dice (mean TC, WT, ET)")
    ax.set_title("Modality ablation: Dice by removed modality and replacement type")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dice_by_modality_replacement.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dice_by_modality_replacement.png")


def fig_dice_heatmap(single):
    if not single:
        return
    rep_ord = rep_order(single)
    mat = np.zeros((len(MOD_ORDER), len(rep_ord)))
    for i, mod in enumerate(MOD_ORDER):
        for j, rep in enumerate(rep_ord):
            v = (get_val(single, mod, rep, "dice_TC") + get_val(single, mod, rep, "dice_WT") + get_val(single, mod, rep, "dice_ET")) / 3
            mat[i, j] = v if not np.isnan(v) else 0
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(rep_ord)))
    ax.set_yticks(np.arange(len(MOD_ORDER)))
    ax.set_xticklabels([r.replace("copy_", "C.") for r in rep_ord])
    ax.set_yticklabels(["FLAIR", "T1ce", "T1", "T2"])
    for i in range(len(MOD_ORDER)):
        for j in range(len(rep_ord)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Dice (mean)")
    ax.set_title("Dice (mean TC, WT, ET) — Removed modality vs replacement")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dice_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dice_heatmap.png")


def fig_dice_per_region(rows):
    plot_rows = [r for r in rows if r["scenario"] != "baseline"]
    if not plot_rows:
        return
    labels = [r["removed_modality"] + "\n" + r["replacement_type"] for r in plot_rows[:20]]
    n = len(labels)
    plot_rows = plot_rows[:20]
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, [r["dice_TC"] for r in plot_rows], w, label="Dice TC", color="C0")
    ax.bar(x, [r["dice_WT"] for r in plot_rows], w, label="Dice WT", color="C1")
    ax.bar(x + w, [r["dice_ET"] for r in plot_rows], w, label="Dice ET", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Dice")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_title("Dice per region (TC, WT, ET) by scenario")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dice_per_region.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dice_per_region.png")


def fig_sensitivity_by_modality(single, baseline):
    if not single or not baseline:
        return
    b = baseline[0]
    baseline_s = (b["sensitivity_TC"] + b["sensitivity_WT"] + b["sensitivity_ET"]) / 3
    means = []
    for mod in MOD_ORDER:
        vals = [(r["sensitivity_TC"] + r["sensitivity_WT"] + r["sensitivity_ET"]) / 3 for r in single if r["removed_modality"] == mod]
        means.append(np.mean(vals) if vals else 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MOD_ORDER))
    ax.bar(x, means, color="steelblue", edgecolor="navy")
    ax.axhline(y=baseline_s, color="gray", linestyle="--", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(["FLAIR", "T1ce", "T1", "T2"])
    ax.set_ylabel("Sensitivity (mean TC, WT, ET)")
    ax.set_title("Mean sensitivity when each modality is removed (avg over replacement types)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_by_modality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sensitivity_by_modality.png")


def fig_hd95_by_modality(single, baseline):
    if not single or not baseline:
        return
    b = baseline[0]
    baseline_h = (b["hd95_TC"] + b["hd95_WT"] + b["hd95_ET"]) / 3
    means = []
    for mod in MOD_ORDER:
        vals = [(r["hd95_TC"] + r["hd95_WT"] + r["hd95_ET"]) / 3 for r in single if r["removed_modality"] == mod]
        means.append(np.mean(vals) if vals else 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MOD_ORDER))
    ax.bar(x, means, color="coral", edgecolor="darkred")
    ax.axhline(y=baseline_h, color="gray", linestyle="--", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(["FLAIR", "T1ce", "T1", "T2"])
    ax.set_ylabel("HD95 (mean, voxels)")
    ax.set_title("Mean HD95 when each modality is removed (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hd95_by_modality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: hd95_by_modality.png")


def fig_replacement_comparison(single, baseline):
    if not single or not baseline:
        return
    b = baseline[0]
    baseline_val = (b["dice_TC"] + b["dice_WT"] + b["dice_ET"]) / 3
    rep_ord = rep_order(single)
    means = []
    for rep in rep_ord:
        vals = [(r["dice_TC"] + r["dice_WT"] + r["dice_ET"]) / 3 for r in single if r["replacement_type"] == rep]
        means.append(np.mean(vals) if vals else 0)
    labels = [r.replace("copy_", "Copy ") for r in rep_ord]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color="teal", alpha=0.8, edgecolor="black")
    ax.axhline(y=baseline_val, color="gray", linestyle="--", label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Dice (mean TC, WT, ET)")
    ax.set_title("Dice by replacement type (averaged over all removed modalities)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dice_by_replacement_type.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dice_by_replacement_type.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return
    rows, single, baseline = load_data()
    if not baseline:
        print("No baseline row in CSV.")
        return
    print("Generating plots...")
    fig_dice_by_modality_and_replacement(single, baseline)
    fig_dice_heatmap(single)
    fig_dice_per_region(rows)
    fig_sensitivity_by_modality(single, baseline)
    fig_hd95_by_modality(single, baseline)
    fig_replacement_comparison(single, baseline)
    print(f"Done. Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
