from grooveEvaluator.distanceData import DistanceData

import numpy as np

INTRASET_DISTANCE_MATRIX = np.array([[0, 1, 2], [1, 0, 3], [1, 2, 0]])
INTERSET_DISTANCE_MATRIX = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def test_distance_data_intraset():
    is_intraset = True
    expected_flattened_distances = np.array([1, 2, 3])

    distance_data = DistanceData(INTRASET_DISTANCE_MATRIX, is_intraset)

    assert np.array_equal(distance_data.distance_matrix, INTRASET_DISTANCE_MATRIX)
    assert np.array_equal(distance_data.flattened_distances, expected_flattened_distances)
    assert distance_data.is_intraset
    print("test_distance_data_intraset passed")


def test_distance_data_interset():
    is_intraset = False
    expected_flattened_distances = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    distance_data = DistanceData(INTERSET_DISTANCE_MATRIX, is_intraset)

    assert np.array_equal(distance_data.distance_matrix, INTERSET_DISTANCE_MATRIX)
    assert np.array_equal(distance_data.flattened_distances, expected_flattened_distances)
    assert not distance_data.is_intraset
    print("test_distance_data_interset passed")


if __name__ == "__main__":
    test_distance_data_intraset()
    test_distance_data_interset()