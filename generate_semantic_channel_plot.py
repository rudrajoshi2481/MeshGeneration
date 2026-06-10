#!/usr/bin/env python3
"""
Generate semantic channel plot from existing results.json
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRASH = "/media/rudhra/ChenLabData1/Middleware/Nvidia_project/sementic_channel_project/trash"
RESULTS_PATH = os.path.join(TRASH, "semantic_channel_results", "results.json")
PLOT_DIR = os.path.join(TRASH, "semantic_channel_results", "plots")

def plot_accuracy_vs_missing(results: dict, save_path: str):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC", "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    COLORS = {
        "SEDD (interval mask)": "#5B8DB8",
        "DoT (prefix)":         "#F4A35A",
        "Clean":                "#6DBF8A",
        "SEDD full":            "#A48CC4",
        "DoT full":             "#D96B6B",
    }

    if "clean_accuracy" in results:
        ax.axhline(results["clean_accuracy"], color=COLORS["Clean"],
                   linestyle="--", linewidth=1.5, label=f"Clean ({results['clean_accuracy']:.1f}%)")

    if "sedd_full_accuracy" in results:
        ax.axhline(results["sedd_full_accuracy"], color=COLORS["SEDD full"],
                   linestyle=":", linewidth=1.5,
                   label=f"SEDD full gen ({results['sedd_full_accuracy']:.1f}%)")

    if "dot_full_accuracy" in results:
        ax.axhline(results["dot_full_accuracy"], color=COLORS["DoT full"],
                   linestyle=":", linewidth=1.5,
                   label=f"DoT full gen ({results['dot_full_accuracy']:.1f}%)")

    if "sedd_partial" in results:
        xs = [r["pct_missing"] for r in results["sedd_partial"]]
        ys = [r["accuracy"] for r in results["sedd_partial"]]
        ax.plot(xs, ys, "o-", color=COLORS["SEDD (interval mask)"],
                linewidth=2, markersize=6, label="SEDD (interval mask)")

    if "dot_partial" in results:
        xs = [r["pct_missing"] for r in results["dot_partial"]]
        ys = [r["accuracy"] for r in results["dot_partial"]]
        ax.plot(xs, ys, "s-", color=COLORS["DoT (prefix)"],
                linewidth=2, markersize=6, label="DoT (prefix)")

    ax.set_xlabel("% Tokens Missing", labelpad=8, fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", labelpad=8, fontsize=12)
    ax.set_title("Semantic Channel: Accuracy vs Token Loss\n(Point Cloud / ModelNet40)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Saved → {save_path}")

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    
    plot_path = os.path.join(PLOT_DIR, "accuracy_vs_missing_tokens.png")
    plot_accuracy_vs_missing(results, plot_path)
    print(f"[DONE] Plot saved to: {plot_path}")
