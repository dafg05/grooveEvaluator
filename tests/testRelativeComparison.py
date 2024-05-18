# import grooveEvaluator.relativeComparison as featEval
from grooveEvaluator.relativeComparison import *
from grooveEvaluator.constants import *
from tests.constants import *
from tests.dataset import GrooveMidiDataset

import numpy as np

GENERATED_SET = GrooveMidiDataset(filters=AFROBEAT_FILTERS)
VALIDATION_SET = GrooveMidiDataset(filters=AFROBEAT_FILTERS)

def test_relative_comparison():

    print(len(GENERATED_SET))
    comparison_results_by_feat = relative_comparison(GENERATED_SET, VALIDATION_SET)
    
    assert len(comparison_results_by_feat) == len(EVAL_FEATURES), f"Expected {len(EVAL_FEATURES)} comparison results, got {len(comparison_results_by_feat)}"


    for feat in EVAL_FEATURES:
        comparison_result = comparison_results_by_feat[feat]

        # Points check
        assert len(comparison_result.points) == 1000, f"Expected 1000 points, got {len(comparison_result.points)}."
        reshaped_points = comparison_result.points.reshape(-1, 1)

        # KDEs check
        kde_dict = comparison_result.kde_dict
        generated_kde = kde_dict[GENERATED_INTRASET_KEY]
        validation_kde = kde_dict[VALIDATION_INTRASET_KEY]
        interset_kde = kde_dict[INTERSET_KEY]
        # Since the generated and validation sets are the same, their kdes should be the same
        assert np.all(np.equal(generated_kde.score_samples(reshaped_points), validation_kde.score_samples(reshaped_points))), f"Generated and validation kdes are not equal for feature {feat}"
        assert np.all(np.equal(generated_kde.score_samples(reshaped_points), interset_kde.score_samples(reshaped_points))), f"Generated and interset kdes are not equal for feature {feat}"

        # Metrics check 
        assert np.isclose(comparison_result.kl_divergence, 0, rtol=1e-2), f"KL divergence should be close to zero, is {comparison_result.kl_divergence}"
        assert comparison_result.overlapping_area > 0, f"Overlapping area should be greater than zero, is {comparison_result.overlapping_area}"

    print("test_relative_comparison passed")


def test_relative_comparison_simple():
    print(len(GENERATED_SET))
    comparison_results_by_feat = relative_comparison(GENERATED_SET, VALIDATION_SET, simple=True)
    
    assert len(comparison_results_by_feat) == len(EVAL_FEATURES), f"Expected {len(EVAL_FEATURES)} comparison results, got {len(comparison_results_by_feat)}"

    for feat in EVAL_FEATURES:
        comparison_result = comparison_results_by_feat[feat]

        # Points check
        assert len(comparison_result.points) == 1000, f"Expected 1000 points, got {len(comparison_result.points)}."

        # Metrics check 
        assert np.isclose(comparison_result.kl_divergence, 0, rtol=1e-2), f"KL divergence should be close to zero, is {comparison_result.kl_divergence}"
        assert comparison_result.overlapping_area > 0, f"Overlapping area should be greater than zero, is {comparison_result.overlapping_area}"

        # Stats check
        stats_dict = comparison_result.stats_dict
        
        generated_stats = stats_dict[GENERATED_INTRASET_KEY]
        validation_stats = stats_dict[VALIDATION_INTRASET_KEY]
        interset_stats = stats_dict[INTERSET_KEY]

        generated_mean = generated_stats[MEAN_KEY]
        generated_std = generated_stats[STD_KEY]

        validation_mean = validation_stats[MEAN_KEY]
        validation_std = validation_stats[STD_KEY]

        interset_mean = interset_stats[MEAN_KEY]
        interset_std = interset_stats[STD_KEY]

        assert np.isclose(generated_mean, validation_mean, rtol=1e-2), f"Generated and validation means should be close, are {generated_mean} and {validation_mean}"
        assert np.isclose(generated_mean, interset_mean, rtol=1e-2), f"Generated and interset means should be close, are {generated_mean} and {interset_mean}"
        assert np.isclose(generated_std, validation_std, rtol=1e-2), f"Generated and validation stds should be close, are {generated_std} and {validation_std}"
        assert np.isclose(generated_std, interset_std, rtol=1e-2), f"Generated and interset stds should be close, are {generated_std} and {interset_std}" 

    print("test_relative_comparison_simple passed!")


if __name__ == "__main__":
    test_relative_comparison()
    test_relative_comparison_simple()