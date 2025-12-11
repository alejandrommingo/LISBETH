import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample

class SubspaceAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_eigenvalues(self, data_matrix):
        """
        Computes the singular values (and derived eigenvalues) of the data matrix.
        Data should be centered before calling this.
        """
        # We use TruncatedSVD for efficiency, asking for full rank approx to get all SVs relevant
        # effective rank is min(n_samples, n_features)
        n_components = min(data_matrix.shape) - 1
        # Limit to a reasonable max to avoid memory issues with huge matrices if any
        n_components = min(n_components, 100) 
        
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        svd.fit(data_matrix)
        
        # Singular values
        return svd.singular_values_

    def horns_parallel_analysis(self, data_matrix, num_simulations=20, percentile=95):
        """
        Determines the optimal number of dimensions (k) using Horn's Parallel Analysis.
        Compares real singular values against the distribution of singular values 
        from random noise matrices of the same shape.
        """
        n_samples, n_features = data_matrix.shape
        real_sv = self.get_eigenvalues(data_matrix)
        
        noise_sv_accum = []
        
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(num_simulations):
            # Generate random noise matrix with same shape and variance structure approx
            # Ideally standard normal distribution
            noise_matrix = rng.randn(n_samples, n_features)
            
            # Compute SV for noise
            noise_sv = self.get_eigenvalues(noise_matrix)
            noise_sv_accum.append(noise_sv)
            
        # Stack to Calculate Percentile
        # Pad with zeros if lengths differ (due to solver convergence in some edge cases)
        max_len = max(len(sv) for sv in noise_sv_accum)
        noise_sv_matrix = np.zeros((num_simulations, max_len))
        for i, sv in enumerate(noise_sv_accum):
            noise_sv_matrix[i, :len(sv)] = sv
            
        # Calculate threshold (e.g., 95th percentile)
        thresholds = np.percentile(noise_sv_matrix, percentile, axis=0)
        
        # Compare
        # Optimal k is the count where Real > Threshold
        # We handle length mismatch if real has fewer components than max noise
        limit = min(len(real_sv), len(thresholds))
        
        k_optimal = 0
        for i in range(limit):
            if real_sv[i] > thresholds[i]:
                k_optimal += 1
            else:
                break
                
        return k_optimal, real_sv, thresholds

    def bootstrap_stability(self, data_matrix, k, n_boot=50):
        """
        Checks stability of the subspace spanned by top-k components.
        Returns the average cosine similarity between original top-k subspace
        and bootstrapped top-k subspaces.
        (Simplified approach: Average Grassmanian distance or Principal Angles implication)
        """
        if k < 1:
            return 0.0
            
        # Original subspace basis
        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        svd.fit(data_matrix)
        original_components = svd.components_ # Shape (k, n_features)
        
        similarities = []
        
        for i in range(n_boot):
            # Resample rows
            boot_data = resample(data_matrix, random_state=self.random_state + i)
            
            svd_boot = TruncatedSVD(n_components=k, random_state=self.random_state + i)
            svd_boot.fit(boot_data)
            boot_components = svd_boot.components_
            
            # Project original basis onto bootstrapped basis
            # Canonical correlation / Principal Angles
            # Simpler metric: subspace overlap. Tr(A' B B' A) / k
            # Let U = original_components.T, V = boot_components.T
            # Overlap = Trace( (U.T @ V) @ (V.T @ U) ) / k
            # Ranges from 0 (orthogonal) to 1 (identical)
            
            U = original_components.T
            V = boot_components.T
            
            # Projection matrix P_v = V @ V.T
            # We want correlation between subspaces.
            # SVD of U.T @ V gives cosines of principal angles. 
            # Mean of squared cosines is a good stability metric.
            
            cross_prod = np.dot(U.T, V)
            _, s, _ = np.linalg.svd(cross_prod)
            # s contains cosines of principal angles
            mean_similarity = np.mean(s) # Average cosine
            similarities.append(mean_similarity)
            
        return np.mean(similarities), np.std(similarities)
