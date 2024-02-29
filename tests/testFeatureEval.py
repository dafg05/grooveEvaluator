import grooveEvaluator.featureEval as featEval
from grooveEvaluator.constants import EVAL_FEATURES
from tests.constants import *
from tests.dataset import GrooveMidiDataset

import numpy as np

GENERATED_SET = GrooveMidiDataset(filters=AFROBEAT_FILTERS)
VALIDATION_SET = GrooveMidiDataset(filters=AFROBEAT_FILTERS)

def test_relative_comparison():

    print(len(GENERATED_SET))
    kdes_by_feat, metrics_by_feat, points_by_feat = featEval.relative_comparison(GENERATED_SET, VALIDATION_SET)
    
    assert len(kdes_by_feat) == len(EVAL_FEATURES)
    assert len(kdes_by_feat) == len(metrics_by_feat)
    assert len(kdes_by_feat) == len(points_by_feat)

    for feat in EVAL_FEATURES:
        # Points check
        assert feat in points_by_feat.keys()
        assert len(points_by_feat[feat]) == 1000, f"Expected 1000 points, got {len(points_by_feat[feat])}."
        reshaped_points = points_by_feat[feat].reshape(-1, 1)

        # KDEs check
        print(kdes_by_feat.keys())
        assert feat in kdes_by_feat.keys(), f"Feature {feat} not found in kdes_by_feat"
        generated_kde = kdes_by_feat[feat][0]
        validation_kde = kdes_by_feat[feat][1]
        interset_kde = kdes_by_feat[feat][2]

        # Since the generated and validation sets are the same, their kdes should be the same
        assert np.all(np.equal(generated_kde.score_samples(reshaped_points), validation_kde.score_samples(reshaped_points))), f"Generated and validation kdes are not equal for feature {feat}"
        assert np.all(np.equal(generated_kde.score_samples(reshaped_points), interset_kde.score_samples(reshaped_points))), f"Generated and interset kdes are not equal for feature {feat}"
        # Metrics check 
        assert feat in metrics_by_feat.keys(), f"Feature {feat} not found in metrics_by_feat"
        assert np.isclose(metrics_by_feat[feat][0], 0, rtol=1e-2), f"KL divergence should be close to zero, is {metrics_by_feat[feat][0]}"
        assert np.isclose(metrics_by_feat[feat][1], 1, rtol=1e-1), f"Overlapping area should be close to 1, is {metrics_by_feat[feat][1]}"

    print("test_relative_comparison passed")



if __name__ == "__main__":
    test_relative_comparison()