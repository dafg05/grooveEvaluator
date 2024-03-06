from grooveEvaluator.distanceData import DistanceData

import numpy as np

INTERSET_DISTANCE_MATRIX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def test_distance_data():
    is_intraset = False
    expected_flattened_distances = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    distance_data = DistanceData(INTERSET_DISTANCE_MATRIX, is_intraset)

    assert np.array_equal(distance_data.distance_matrix, INTERSET_DISTANCE_MATRIX)
    assert np.array_equal(distance_data.flattened_distances, expected_flattened_distances)
    assert not distance_data.is_intraset
    print("test_distance_data_interset passed")


if __name__ == "__main__":
    test_distance_data()