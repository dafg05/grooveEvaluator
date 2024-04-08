import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from pathlib import Path
from typing import Dict, List
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

def plot_multiple_distance_metrics(results_1: Dict[str, ComparisonResult], results_2: Dict[str, ComparisonResult], setname_1: str, setname2: str, out_dir: Path, figname: str = "Distance Metrics", x_limit: float=-1, colors: List[str] = None):
    features_1 = list(results_1.keys())
    kl_divergences_1 = [result.kl_divergence for result in results_1.values()]
    overlapping_areas_1 = [result.overlapping_area for result in results_1.values()]

    features_2 = list(results_2.keys())
    kl_divergences_2 = [result.kl_divergence for result in results_2.values()]
    overlapping_areas_2 = [result.overlapping_area for result in results_2.values()]

    if not colors:
        # Using a more distinct color palette
        colors = plt.get_cmap('tab20').colors
    if x_limit < 0:
        x_limit = max(kl_divergences_1 + kl_divergences_2) + 0.01 * max(kl_divergences_1 + kl_divergences_2)

    plt.figure(figsize=(12, 8))

    # Plotting the first set of results with circles
    for i, feature in enumerate(features_1):
        plt.scatter(kl_divergences_1[i], overlapping_areas_1[i], color=colors[i % len(colors)], marker='o', label=f"{feature} ({setname_1})")

    # Plotting the second set of results with squares
    for i, feature in enumerate(features_2):
        plt.scatter(kl_divergences_2[i], overlapping_areas_2[i], color=colors[i % len(colors)], marker='s', label=f"{feature} ({setname2})")

    plt.xlabel('KL-D')
    plt.ylabel('OA')
    plt.title(figname)

    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)

    plt.xlim(left=0, right=x_limit)
    plt.ylim(bottom=0, top=1.0)

    # Improve legend placement and ensure it's not cut off
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Features", fontsize='small', markerscale=1)
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