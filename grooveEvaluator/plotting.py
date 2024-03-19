import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from pathlib import Path
from typing import Dict
from sklearn.neighbors import KernelDensity
from grooveEvaluator.relativeComparison import ComparisonResult

def plot_distance_metrics(results: Dict[str, ComparisonResult], out_dir: Path, figname: str = "Distance Metrics", x_limit: float=-1, colors: list = None):
    features = list(results.keys())
    kl_divergences = [result.kl_divergence for result in results.values()]
    overlapping_areas = [result.overlapping_area for result in results.values()]

    if not colors:
        colors = cm.rainbow(np.linspace(0, 1, len(features)))
    if x_limit < 0:
        x_limit = max(kl_divergences) + 0.01 * max(kl_divergences)

    plt.figure(figsize=(12, 8))

    for i, feature in enumerate(features):
        plt.scatter(kl_divergences[i], overlapping_areas[i], color=colors[i], label=feature)

    plt.xlabel('KL-D')
    plt.ylabel('OA')
    plt.title(figname)

    plt.axhline(y=0, color='black',linewidth=0.5)
    plt.axvline(x=0, color='black',linewidth=0.5)

    plt.xlim(left=0, right=x_limit)
    plt.ylim(bottom=0, top=1.0)

    # Improve legend placement and ensure it's not cut off
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", title="Features")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect layout to make room for the legend

    plt.savefig(out_dir / f"{figname}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_kdes(kde_dict: Dict[str, KernelDensity], points: np.ndarray, out_dir: Path, figname: str = "KDEs", colors: list = None):
    if not colors:
        colors = cm.rainbow(np.linspace(0, 1, len(kde_dict.keys())))

    plt.figure(figsize=(12, 8))

    for (name, kde), color in zip(kde_dict.items(), colors):
        log_dens = kde.score_samples(points)
        dens = np.exp(log_dens)
        plt.plot(points, dens, color=color, label=name)

    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title(figname)
    plt.legend(title="KDE type")

    plt.savefig(out_dir / f"{figname}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()