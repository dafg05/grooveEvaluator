import numpy as np
import grooveEvaluator.utils as utils

from grooveEvaluator.utils import DistanceData
from grooveEvaluator.constants import *
from tqdm import tqdm
from typing import Dict
from hvo_sequence.hvo_seq import HVO_Sequence
from sklearn.neighbors import KernelDensity

def get_features_dict(hvo_dataset, features_to_extract, use_tqdm=False) -> Dict[str, np.ndarray]:
    """
    TODO: Docstring
    TODO: Use tqdm
    """
    extracted_features = {}
    for feat in features_to_extract:
        extracted_features[feat] = np.array([])

    iter = range(len(hvo_dataset))
    if use_tqdm:
        iter = tqdm(iter, desc="Extracting features from HVO_Sequence Set")

    for ix in iter:
        sample_hvo = hvo_dataset[ix]
        sample_features = __get_hvo_features(sample_hvo, features_to_extract)
        for feat in features_to_extract:
            extracted_features[feat] = np.append(extracted_features[feat], sample_features[feat])

    return extracted_features

def get_intraset_dd_dict(extracted_features: Dict[str, np.ndarray], use_tqdm=True) -> Dict[str, DistanceData]:
    """
    Calculates the intraset distance matrix for each feature in the extracted_features dictionary.

    TODO: Use tqdm
    """
    intraset_dd_dict = {}
    for feat_name, feat_values in extracted_features.items():
        distance_matirx = utils.get_intraset_distance_matrix(feat_values)
        intraset_dd_dict[feat_name] = DistanceData(distance_matirx, is_intraset=True)
    
    return intraset_dd_dict

def get_interset_dd_dict(extracted_features_a: Dict[str, np.ndarray], extracted_features_b: Dict[str, np.ndarray]) -> Dict[str, DistanceData]:
    """
    Calculates the interset distance matrix for each feature in the each of the extracted_features dictionaries.

    TODO: Use tqdm
    """

    if extracted_features_a.keys() != extracted_features_b.keys():
        raise ValueError("Feature sets must be identical to calculate interset distances.")

    interset_dd_dict = {}
    for feat_name, feat_values_a in extracted_features_a.items():
        # Get the values in second set corresponding to feature
        feat_values_b = extracted_features_b[feat_name]
        distance_matrix = utils.get_interset_distance_matrix(feat_values_a, feat_values_b)
        interset_dd_dict[feat_name] = DistanceData(distance_matrix, is_intraset=False)
    
    return interset_dd_dict

def __get_hvo_features(hvo_sequence: HVO_Sequence, features_to_extract):
    """
    For now, hard code this method to return eval features.
    """
    return __get_eval_hvo_features(hvo_sequence)

def __get_eval_hvo_features(hvo_sequence: HVO_Sequence):
    lmh_sync_info = hvo_sequence.get_low_mid_hi_syncopation_info()
    ac_features = hvo_sequence.get_velocity_autocorrelation_features()
    features = {
        NUM_INSTRUMENTS_KEY : hvo_sequence.get_number_of_active_voices(),
        TOTAL_STEP_DENSITY_KEY : hvo_sequence.get_total_step_density(),
        AVERAGE_VOICE_DENSITY_KEY : hvo_sequence.get_average_voice_density(),
        VEL_SIMILARITY_SCORE_KEY : hvo_sequence.get_velocity_score_symmetry(), # is this right?
        COMBINED_SYNCOPATION_KEY : hvo_sequence.get_combined_syncopation(), # this is wrong?
        POLYPHONIC_SYNCOPATION_KEY : hvo_sequence.get_witek_polyphonic_syncopation(),
        LOW_SYNC_KEY : lmh_sync_info["lowsync"],
        MID_SYNC_KEY : lmh_sync_info["midsync"],
        HIGH_SYNC_KEY: lmh_sync_info["hisync"],
        LOW_SYNESS_KEY : lmh_sync_info["lowsyness"],
        MID_SYNESS_KEY : lmh_sync_info["midsyness"],
        HIGH_SYNESS_KEY: lmh_sync_info["hisyness"],
        COMPLEXITY_KEY : hvo_sequence.get_total_complexity(),
        LAIDBACKNESS_KEY : hvo_sequence.laidbackness(),
        TIMING_ACCURACY_KEY : hvo_sequence.get_timing_accuracy(),
        AUTO_CORR_SKEW_KEY : ac_features["skewness"],
        AUTO_CORR_MAX_KEY : ac_features["max"],
        AUTO_CORR_CENTROID_KEY : ac_features["centroid"],
        AUTO_CORR_HARMONICITY_KEY : ac_features["harmonicity"]
    }

    return features