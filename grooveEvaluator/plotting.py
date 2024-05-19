import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np

from pathlib import Path
from typing import Dict, List
from sklearn.neighbors import KernelDensity
from grooveEvaluator.relativeComparison import ComparisonResult

def plot_distance_metrics(results: Dict[str, ComparisonResult], out_dir: Path, figname: str = "Distance Metrics", x_limit: float=-1, colors: list = None, non_negative_klds: bool = False):
    features = list(results.keys())
    kl_divergences = [result.kl_divergence for result in results.values()]
    if non_negative_klds:
        kl_divergences = [max(0, kl_divergence) for kl_divergence in kl_divergences]
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

def plot_multiple_distance_metrics(results_1, results_2, setname_1, setname_2, out_dir, figname="Distance Metrics", x_right_limit=-1, y_bottom_limit=-1,colors=None, non_negative_klds=False):
    if not colors:
        colors = plt.get_cmap('tab20').colors

    plt.figure(figsize=(9, 5))
    max_kl_divergence = float('-inf')
    min_overlapping_area = float('inf')

    # Plot with specified markers
    for i, (feature, result) in enumerate(results_2.items()):
        if non_negative_klds:
            result.kl_divergence = max(0, result.kl_divergence)
        plt.scatter(result.kl_divergence, result.overlapping_area, color=colors[i], marker='o', label=feature)  # Circles for set 2
        max_kl_divergence = max(max_kl_divergence, result.kl_divergence)
        min_overlapping_area = min(min_overlapping_area, result.overlapping_area)
    for i, (feature, result) in enumerate(results_1.items()):
        if non_negative_klds:
            result.kl_divergence = max(0, result.kl_divergence)
        plt.scatter(result.kl_divergence, result.overlapping_area, color=colors[i], marker='v', label=feature)  # Triangles for set 1
        max_kl_divergence = max(max_kl_divergence, result.kl_divergence)
        min_overlapping_area = min(min_overlapping_area, result.overlapping_area)

    # Adjusting limits
    if x_right_limit < 0:
        x_right_limit = max_kl_divergence + 0.1 * max_kl_divergence
    
    if y_bottom_limit < 0:
        y_bottom_limit = max(0, min_overlapping_area - 0.1 * min_overlapping_area)

    plt.xlim(left=0, right=x_right_limit)
    plt.ylim(bottom=y_bottom_limit, top=1.0)

    # Legend for colors (features)
    legend_elements = [plt.Line2D([0], [0], color=color, marker='s', linestyle='', markersize=10) for color in colors]
    plt.legend(legend_elements, [feature for feature in results_1.keys()], title="Features", title_fontproperties={'weight':'bold'},bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('KL-Divergence', fontweight='bold')
    plt.ylabel('Overlapping Area', fontweight='bold')
    plt.title(figname, fontweight='bold', fontsize = 12)

   # Annotations for set markers
    plt.text(0.5, -0.2, f'{setname_1} ▼ | {setname_2} ●', ha='center', va='center', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 0.75, 1])

    plt.savefig(out_dir / f"{figname}_plot.png", dpi=300)
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