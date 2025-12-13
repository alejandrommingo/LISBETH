import numpy as np
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample

logger = logging.getLogger(__name__)

class DimensionalitySelector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def select_k_horns(self, data_matrix: np.ndarray, num_simulations: int = 20, percentile: int = 95) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Implements Horn's Parallel Analysis to select optimal K.
        Returns: k_optimal, real_singular_values, noise_thresholds
        """
        n_samples, n_features = data_matrix.shape
        # Ensure centered data? usually PCA assumes centered. 
        # We will center it here to be safe if not already.
        data_matrix = data_matrix - np.mean(data_matrix, axis=0)
        
        # 1. Real Eigenvalues (Singular Values)
        real_sv = self._compute_singular_values(data_matrix)
        
        # 2. Noise Simulations
        noise_sv_list = []
        for _ in range(num_simulations):
            # Create random noise with same shape
            # Ideally match the variance of the original data? 
            # Standard PA uses N(0,1), but some variants permute the data.
            # We'll use N(0,1) for standard Horn's.
            noise = self.rng.randn(n_samples, n_features)
            noise_sv = self._compute_singular_values(noise)
            noise_sv_list.append(noise_sv)
            
        # 3. Calculate Thresholds
        # Pad if necessary (unlikely with same shape)
        min_len = min(len(sv) for sv in noise_sv_list)
        noise_sv_matrix = np.array([sv[:min_len] for sv in noise_sv_list])
        
        thresholds = np.percentile(noise_sv_matrix, percentile, axis=0)
        
        # 4. Compare
        k_optimal = 0
        limit = min(len(real_sv), len(thresholds))
        
        # Guard against 0 dimensions
        if limit == 0:
            return 1, real_sv, thresholds
            
        for i in range(limit):
            if real_sv[i] > thresholds[i]:
                k_optimal += 1
            else:
                break
                
        # Enforce at least k=1 if data exists
        k_optimal = max(1, k_optimal)
        
        return k_optimal, real_sv, thresholds

    def check_stability_bootstrap(self, data_matrix: np.ndarray, k: int, n_boot: int = 20) -> float:
        """
        Checks stability of top-K subspace using bootstrapping.
        Returns average cosine similarity (0 to 1).
        """
        if k < 1:
            return 0.0
            
        # Original subspace
        # Center data
        data_matrix = data_matrix - np.mean(data_matrix, axis=0)
        
        U_orig = self._get_basis(data_matrix, k, seed=self.random_state)
        
        similarities = []
        
        for i in range(n_boot):
            # Resample
            boot_data = resample(data_matrix, random_state=self.random_state + i)
            # Center boot data? Yes.
            boot_data = boot_data - np.mean(boot_data, axis=0)
            
            U_boot = self._get_basis(boot_data, k, seed=self.random_state + i)
            
            # Subspace Similarity: Trace(U_orig.T @ U_boot @ U_boot.T @ U_orig) / k
            # = || U_orig.T @ U_boot ||_F^2 / k
            
            cross = np.dot(U_orig.T, U_boot)
            sim = np.linalg.norm(cross)**2 / k
            similarities.append(sim)
            
        return float(np.mean(similarities))

    def _compute_singular_values(self, X):
        n = min(X.shape)
        # Use full SVD or Truncated up to n
        svd = TruncatedSVD(n_components=min(n, 100), random_state=self.random_state)
        svd.fit(X)
        return svd.singular_values_
        
    def _get_basis(self, X, k, seed):
        svd = TruncatedSVD(n_components=k, random_state=seed)
        svd.fit(X)
        return svd.components_.T # Return shapes (n_features, k) as basis
