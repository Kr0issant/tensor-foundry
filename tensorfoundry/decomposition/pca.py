import numpy as np

class PCA:
    def fit_transform(data, n_components):
        mean = np.mean(data, axis=0)
        std_deviation = np.std(data, axis=0)
        std_deviation[std_deviation == 0] = 1e-8

        centered_data = (data - mean) / std_deviation

        covariance_matrix = np.cov(centered_data, rowvar=False)

        e_values, e_vectors = np.linalg.eig(covariance_matrix)
        e_values = e_values.real
        e_vectors = e_vectors.real

        sorted_indices = np.argsort(e_values)[::-1]
        e_vectors = e_vectors[:, sorted_indices]

        projection_matrix = e_vectors[:, :n_components]
        projected_data = np.dot(centered_data, projection_matrix)

        return projected_data