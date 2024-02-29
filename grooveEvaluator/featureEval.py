import numpy as np
from sklearn.neighbors import KernelDensity
from distanceData import DistanceData
from constants import EVAL_FEATURES
import grooveEvaluator.featureExtractor as fe
import grooveEvaluator.utils as utils

def relative_comparison(generated_set, validation_set, features_to_extract=EVAL_FEATURES, num_points=1000, padding_factor=0.05):
    """
    TODO: test me
    TODO: Docstring
    """
    if features_to_extract != EVAL_FEATURES:
        raise ValueError("Relative comparison currently only supports the default feature set.")
    
    generated_features =  fe.get_features_dict(generated_set, features_to_extract)
    validation_features = fe.get_features_dict(validation_set, features_to_extract)

    generated_intraset_dd_dict = fe.get_intraset_dd_dict(generated_features)
    validation_intraset_dd_dict = fe.get_intraset_dd_dict(validation_features)

    interset_dd_dict = fe.get_interset_dd_dict(generated_features, validation_features)

    kdes_by_feat = {feature: np.ndarray() for feature in features_to_extract}
    points_by_feat = {feature: np.ndarray() for feature in features_to_extract}
    metrics_by_feat = {feature: np.ndarray() for feature in features_to_extract}

    for feature in features_to_extract:
        # compute kl_divergence and overlapping_area for each feature
        generation_dd = generated_intraset_dd_dict[feature]
        validation_dd = validation_intraset_dd_dict[feature]
        interset_dd = interset_dd_dict[feature]

        points = utils.evaluation_points([generation_dd, validation_dd, interset_dd], num_points, padding_factor)
        
        kl_d = utils.kl_divergence(generation_dd.kde, interset_dd.kde, points_by_feat)
        oa = utils.overlapping_area(generation_dd.kde, interset_dd.kde, points_by_feat)

        kdes_by_feat[feature] = np.append(metrics_by_feat[feature], [kl_d, oa])
        points_by_feat = np.append(points_by_feat[feature], points)
        metrics_by_feat[feature] = np.append(metrics_by_feat[feature], [kl_d, oa])

    return kdes_by_feat, points_by_feat, metrics_by_feat