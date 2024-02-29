from sklearn.neighbors import KernelDensity
import numpy as np

class DistanceData:
    """
    TODO: Docstring
    """
    def __init__(self, distance_matrix, is_intraset):
        self.distance_matrix = distance_matrix
        self.is_intraset = is_intraset
        # TODO: if is_intraset, check if distance_matrix is symmetrical
        self.flattened_distances = self.__flattened_relevant_distances()
        self.kde = self.__kde_from_distances()
    
    def __flattened_relevant_distances(self):
        if self.is_intraset:
            # if intraset, we only need the upper triangle distances, as 
            # the main diagonal is all zeroes and the matrix is symmetrical
            flattened_distances = self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)]
        else:
            # if interset, we need all distances
            flattened_distances = self.distance_matrix.flatten()
        return flattened_distances
    
    def __kde_from_distances(self):
        reshaped_distances = self.flattened_distances.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(reshaped_distances)

        return kde