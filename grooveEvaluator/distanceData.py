from sklearn.neighbors import KernelDensity
import numpy as np

class DistanceData:
    """
    Class to hold feature distance data. Initialized with a distance matrix and a boolean indicating if the distance matrix is intraset or interset.
    """
    def __init__(self, distance_matrix, is_intraset):
        self.distance_matrix = distance_matrix
        self.is_intraset = is_intraset
        # TODO: if is_intraset, check if distance_matrix is symmetrical
        self.flattened_distances = self.__flattened_relevant_distances()
        self.kde = self.__kde_from_distances()
    
    def __flattened_relevant_distances(self):
        return self.distance_matrix.flatten()
    
    def __kde_from_distances(self):
        reshaped_distances = self.flattened_distances.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(reshaped_distances)

        return kde