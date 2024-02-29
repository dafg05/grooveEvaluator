import numpy as np
from sklearn.neighbors import KernelDensity
from grooveEvaluator.distanceData import DistanceData
from typing import List

def get_intraset_distance_matrix(feat_values: np.ndarray):
    n = len(feat_values) # number of samples

    # Populate distance matrix of shape (n,n)
    distance_matrix = np.zeros((n, n))
    for sample_ix, value in enumerate(feat_values):
        feat_distances = np.absolute(feat_values - value)
        distance_matrix[sample_ix,:] = feat_distances
    return distance_matrix

def get_interset_distance_matrix(feat_values_a: np.ndarray, feat_values_b: np.ndarray):
    m = len(feat_values_a) # number of samples in a
    n = len(feat_values_b) # number of samples in b

    # Populate distance matrix of shape (m,n)
    distance_matrix = np.zeros((m, n))
    for sample_ix_a, value_a in enumerate(feat_values_a):
        feat_distances = np.absolute(feat_values_b - value_a)
        distance_matrix[sample_ix_a,:] = feat_distances
    return distance_matrix

def kl_divergence(kde_a: KernelDensity, kde_b: KernelDensity, points: np.ndarray):
    """
    Evaluates the Kullback-Leibler divergence between two KDEs at a set of points.

    params: 
    kde_a - KDE of first pdf
    kde_b - KDE of second pdf
    points - points at which to evaluate the divergence

    TODO: Is there a bug? See tester
    TODO: Cite me
    """

    points = points.reshape(-1, 1)

    log_p = kde_a.score_samples(points)
    log_q = kde_b.score_samples(points)

    return np.sum(np.exp(log_p) * (log_p - log_q))

def overlapping_area(kde_a: KernelDensity, kde_b: KernelDensity, points: np.ndarray):
    """
    Use trapezoidal rule to estimate the overlapping area between two KDEs at a set of points.
    Source: https://stackoverflow.com/questions/69570238/is-there-a-way-in-python-to-calculate-the-overlapping-area-between-multiple-curv

    TODO: Test me properly
    TODO: Cite me properly
    TODO: Is this too numerically unstable?
    """

    reshaped_points = points.reshape(-1, 1)

    log_dens_a = kde_a.score_samples(reshaped_points)
    log_dens_b = kde_b.score_samples(reshaped_points)

    dens_a = np.exp(log_dens_a)
    dens_b = np.exp(log_dens_b)

    min_dens = np.minimum(dens_a, dens_b)
    overlapping_area = np.trapz(min_dens, points)

    return overlapping_area

def evaluation_points(dd_list: List[DistanceData], num_points, padding_factor):
    min_val = np.inf
    max_val = -np.inf
    # Calculate min and max values of all distances in all DistanceData objects
    for dd in dd_list:
        min_val = min(min_val, np.min(dd.flattened_distances))
        max_val = max(max_val, np.max(dd.flattened_distances))

    padding = padding_factor * (max_val - min_val)
    min_val -= padding
    max_val += padding

    return np.linspace(min_val, max_val, num_points)
