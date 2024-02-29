import grooveEvaluator.featureExtractor as featExt
from tests.constants import *
from tests.dataset import GrooveMidiDataset
import grooveEvaluator.constants as ge_constants
import numpy as np

HVO_DATASET = GrooveMidiDataset(filters=ROCK_FILTERS)
EXTRACTED_FEATURES_A = {
    "num_instruments" : np.array([1, 2, 3, 4]),
    "total_step_density" : np.array([0.1, 0.2, 0.3, 0.4]),
    "complexity" : np.array([0.01, 0.02, 0.03, 0.04])
}
EXTRACTED_FEATURES_B = {
    "num_instruments" : np.array([4, 3, 2]),
    "total_step_density" : np.array([0.4, 0.3, 0.2]),
    "complexity" : np.array([0.04, 0.03, 0.02])
}

def test_get_features_dict():
    extracted_features = featExt.get_features_dict(HVO_DATASET, features_to_extract=ge_constants.EVAL_FEATURES)

    assert len(extracted_features) == len(ge_constants.EVAL_FEATURES)
    for feat in ge_constants.EVAL_FEATURES:
        assert feat in extracted_features.keys()
        assert len(extracted_features[feat]) == len(HVO_DATASET)

    print("test_get_features_dict passed")

def test_get_intraset_dd_dict():
    intraset_dd_dict = featExt.get_intraset_dd_dict(EXTRACTED_FEATURES_A)
    assert len(intraset_dd_dict) == len(EXTRACTED_FEATURES_A)
    for feat in intraset_dd_dict.keys():
        assert feat in intraset_dd_dict.keys()
        assert intraset_dd_dict[feat].is_intraset
        assert intraset_dd_dict[feat].distance_matrix.shape == (len(EXTRACTED_FEATURES_A[feat]), len(EXTRACTED_FEATURES_A[feat]))

    print("get_intraset_dd_dict passed")

def test_get_interset_dd_dict():
    interset_dd_dict = featExt.get_interset_dd_dict(EXTRACTED_FEATURES_A, EXTRACTED_FEATURES_B)
    assert len(interset_dd_dict) == len(EXTRACTED_FEATURES_B)
    for feat in interset_dd_dict.keys():
        assert feat in interset_dd_dict.keys()
        assert not interset_dd_dict[feat].is_intraset
        assert interset_dd_dict[feat].distance_matrix.shape == (len(EXTRACTED_FEATURES_A[feat]), len(EXTRACTED_FEATURES_B[feat]))
    print("get_interset_dd_dict passed")


if __name__ == "__main__":
    test_get_features_dict()
    test_get_intraset_dd_dict()
    test_get_interset_dd_dict()
