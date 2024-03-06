import numpy as np
from sklearn.neighbors import KernelDensity
from grooveEvaluator.distanceData import DistanceData
from grooveEvaluator.constants import *
import grooveEvaluator.featureExtractor as featExt
import grooveEvaluator.utils as utils

from tqdm import tqdm

def relative_comparison(generated_set, validation_set, features_to_extract=EVAL_FEATURES, num_points=1000, padding_factor=0.05, use_tqdm = True):
    """
    Runs a relative comparison between two hvo sets. For each feature, it computes the kl_divergence and overlapping_area between the pdf of the generated intraset distances and the pdf of the interset distances.
    Returns the kdes for each set and the interset, the points used to evaluate the kdes, and the kl_divergence and overlapping_area.
    
    :param generated_set: hvo set to be used as the generated set
    :param validation_set: hvo set to be used as the validation set
    :param features_to_extract: list of features to be extracted from the hvo sets
    :param num_points: number of points to be used to evaluate the kdes and metrics
    :param padding_factor: factor to be used to pad the range of the kdes

    :return kde_dicts_by_feat: dict where the keys are the features and the values are dicts with the kdes for the generated intraset, validation intraset and interset
    :return points_by_feat: dict where values are the points used to evaluate the kdes and metrics
    :return metric_dicts_by_feat: dict where values are the kl_divergence and overlapping_area

    TODO: There seems to be some weirdness with the computation of the kdes and metrics. Talk to Matteo about it.
    """

    if features_to_extract != EVAL_FEATURES:
        raise ValueError("Relative comparison currently only supports the default feature set.")
    
    assert len(generated_set) == len(validation_set), "Generated and validation sets must be the same length."
    
    generated_features =  featExt.get_features_dict(generated_set, features_to_extract)
    validation_features = featExt.get_features_dict(validation_set, features_to_extract)

    generated_intraset_dd_dict = featExt.get_intraset_dd_dict(generated_features)
    validation_intraset_dd_dict = featExt.get_intraset_dd_dict(validation_features)

    interset_dd_dict = featExt.get_interset_dd_dict(generated_features, validation_features)

    kde_dicts_by_feat = {feature: {} for feature in features_to_extract}
    metric_dicts_by_feat = {feature: {} for feature in features_to_extract}
    points_by_feat = {feature: np.array([]) for feature in features_to_extract}

    for feature in tqdm(features_to_extract, desc="Computing relative comparison metrics"):
        # compute kl_divergence and overlapping_area for each feature
        generation_dd = generated_intraset_dd_dict[feature]
        validation_dd = validation_intraset_dd_dict[feature]
        interset_dd = interset_dd_dict[feature]

        points = utils.evaluation_points([generation_dd, validation_dd, interset_dd], num_points, padding_factor)
        
        # compute comparison metrics between the generated intraset pdf and the interset pdf
        kl_d = utils.kl_divergence(generation_dd.kde, interset_dd.kde, points)
        oa = utils.overlapping_area(generation_dd.kde, interset_dd.kde, points)

        kde_dict = {
            GENERATED_INTRASET_KEY: generation_dd.kde,
            VALIDATION_INTRASET_KEY: validation_dd.kde,
            INTERSET_KEY: interset_dd.kde
        }

        metrics_dict = {
            KL_DIVERGENCE_KEY: kl_d,
            OVERLAPPING_AREA_KEY: oa
        }

        kde_dicts_by_feat[feature] = kde_dict
        metric_dicts_by_feat[feature] = metrics_dict
        points_by_feat[feature] = np.append(points_by_feat[feature], points)

    return kde_dicts_by_feat, metric_dicts_by_feat, points_by_feat