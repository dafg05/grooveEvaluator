import grooveEvaluator.plotting as plotting
import numpy as np
from pathlib import Path
from sklearn.neighbors import KernelDensity
from grooveEvaluator.relativeComparison import ComparisonResult

KDE_A = KernelDensity(kernel='gaussian', bandwidth='scott').fit(np.random.normal(0, 1, 1000).reshape(-1, 1))
KDE_B = KernelDensity(kernel='gaussian', bandwidth='scott').fit(np.random.normal(0, 2, 1000).reshape(-1, 1))
KDE_C = KernelDensity(kernel='gaussian', bandwidth='scott').fit(np.random.normal(-1, 1, 1000).reshape(-1, 1))

POINTS = np.linspace(-5, 5, 1000)

OUT_DIR = Path("tests/out")

COMPARISON_RESULTS_1 = {
    "feature1": ComparisonResult(0.1, 0.2, None, None),
    "feature2": ComparisonResult(0.3, 0.4, None, None),
    "feature3": ComparisonResult(0.5, 0.6, None, None),
    "feature4": ComparisonResult(0.7, 0.8, None, None),
    "feature5": ComparisonResult(0.9, 0.1, None, None),
}

COMPARISON_RESULTS_2 = {
    "feature1": ComparisonResult(0.2, 0.1, None, None),
    "feature2": ComparisonResult(0.5, 0.42, None, None),
    "feature3": ComparisonResult(-40.0, 0.65, None, None),
    "feature4": ComparisonResult(0.65, 0.6, None, None),
    "feature5": ComparisonResult(-1.0, 0.5, None, None),
}

KDE_DICT = {
    "kde_a": KDE_A,
    "kde_b": KDE_B,
    "kde_c": KDE_C
}

def test_plot_distance_metrics():
    plotting.plot_distance_metrics(COMPARISON_RESULTS_1, OUT_DIR, figname="Test Distance Metrics")

def test_plot_multiple_distance_metrics():
    plotting.plot_multiple_distance_metrics(COMPARISON_RESULTS_1, COMPARISON_RESULTS_2,"set_1", "set_2", OUT_DIR, figname="Test Multiple Distance Metrics")

def test_plot_kdes():
    points = POINTS.reshape(-1, 1)
    plotting.plot_kdes(KDE_DICT, points, OUT_DIR, figname="Test KDEs")


if __name__ == "__main__":
    test_plot_distance_metrics()
    test_plot_multiple_distance_metrics()
    test_plot_kdes()