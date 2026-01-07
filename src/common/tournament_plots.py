import os

import matplotlib.pyplot as plt
import numpy as np


def save_tournament_plots(sorted_ranking, loaded_models, save_dir, timestamp):
    """
    Generates and saves tournament statistics charts.
    """
    names = []
    elos = []
    algo_types = []

    color_map = {
        "PPO": "#FF9F43",
        "A2C": "#0ABDE3",
        "SAC": "#EE5253",
        "UNKNOWN": "#8395A7",
    }

    for m_id, score in sorted_ranking:
        m_info = loaded_models[m_id]
        names.append(f"{m_info['type']}\n{m_info['display_name']}")
        elos.append(int(score))
        algo_types.append(m_info["type"])

    # --- GRAPH 1: Models ranking ---
    plt.figure(figsize=(14, 7))
    colors = [color_map.get(algo, color_map["UNKNOWN"]) for algo in algo_types]
    bars = plt.bar(names, elos, color=colors, edgecolor="black", alpha=0.8)
    plt.axhline(y=1200, color="red", linestyle="--", alpha=0.6, label="Base ELO (1200)")
    plt.title(f"Individual Model Rankings (ELO) - {timestamp}", fontsize=14)
    plt.ylabel("ELO Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 3,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "elo_ranking_models.png"))
    plt.close()

    # --- GRAPH 2: Averages vs Max ELO per algo ---
    plt.figure(figsize=(10, 6))

    unique_algos = sorted(list(set(algo_types)))
    avg_elos = [
        np.mean([elos[i] for i, a in enumerate(algo_types) if a == algo])
        for algo in unique_algos
    ]
    max_elos = [
        np.max([elos[i] for i, a in enumerate(algo_types) if a == algo])
        for algo in unique_algos
    ]

    x = np.arange(len(unique_algos))
    width = 0.35

    rects1 = plt.bar(
        x - width / 2,
        avg_elos,
        width,
        label="Average",
        color=[color_map.get(a, color_map["UNKNOWN"]) for a in unique_algos],
        edgecolor="black",
        alpha=0.6,
    )

    rects2 = plt.bar(
        x + width / 2,
        max_elos,
        width,
        label="Best (Max)",
        color=[color_map.get(a, color_map["UNKNOWN"]) for a in unique_algos],
        edgecolor="black",
        hatch="//",
        alpha=0.9,
    )

    plt.title("Algorithm Performance: Average vs Best Model ELO", fontsize=14)
    plt.ylabel("ELO Score")
    plt.xticks(x, unique_algos)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    def autolabel(rects, is_float=False):
        for rect in rects:
            height = rect.get_height()
            label = f"{height:.1f}" if is_float else f"{int(height)}"
            plt.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 2,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    autolabel(rects1, is_float=True)
    autolabel(rects2, is_float=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "elo_comparison_algos.png"))
    plt.close()
