import numpy as np
import pandas as pd
import logging
from scipy.linalg import svd, sqrtm, inv

logger = logging.getLogger(__name__)

class SociologicalMetrics:
    def calculate_drift(self, U_prev: np.ndarray, U_curr: np.ndarray) -> float:
        """
        Calculates Semantic Drift using Grassmannian distance.
        d(S1, S2) = sqrt(k - sum(cos^2(theta_i)))
        where theta_i are principal angles.
        
        Equivalently, related to || U1^T U2 ||_F.
        Since we aligned them, simple Euclidean might work, but Grassmannian is rotation-invariant 
        (robust even if alignment wasn't perfect).
        
        We'll use Projection metric:
        similarity = || U_prev.T @ U_curr ||_F^2 / k
        drift = 1 - similarity
        """
        k = U_prev.shape[1]
        if U_curr.shape[1] != k:
             # Handle dimension change
             # Overlap measure
             min_k = min(k, U_curr.shape[1])
             # Just use what we have
             pass

        # Cosines of principal angles
        # S are the singular values of U_prev.T @ U_curr
        cross = np.dot(U_prev.T, U_curr)
        _, S, _ = svd(cross)
        
        # Squared cosines sum
        cos_sq_sum = np.sum(S**2)
        
        # Max possible is min(k1, k2)
        max_overlap = min(U_prev.shape[1], U_curr.shape[1])
        
        # Normalized drift metric? 
        # 0 = identical, 1 = orthogonal
        drift = 1.0 - (cos_sq_sum / max_overlap)
        
        return float(drift)

    def calculate_entropy(self, singular_values: np.ndarray) -> float:
        """
        Calculates Shannon entropy of the normalized singular values.
        Hiher entropy = more complex/distributed meaning.
        """
        # Normalize to probability distribution
        s_sum = np.sum(singular_values)
        if s_sum == 0:
            return 0.0
            
        probs = singular_values / s_sum
        
        # Entropy
        # Handle 0s
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalize by log(k)? Optional.
        # Returning raw Shannon bits (natural log usually, or base 2)
        # Using base e
        return float(entropy)

    def calculate_frame_projection(self, subspace_basis: np.ndarray, anchors_df: pd.DataFrame) -> dict:
        """
        Project the subspace onto sociological anchor dimensions.
        Expects anchors_df to have 'dimension' and 'embedding'.
        """
        # 1. Aggregate anchors per dimension
        # Group by dimension, average embedding (?) or keep separate?
        # Usually each dimension is defined by a vector. 
        # Method: Centroid of anchors for that dimension.
        
        if anchors_df.empty:
            return {}
            
        dim_vectors = {}
        if 'embedding' in anchors_df.columns:
            emb_col = 'embedding'
        elif 'embedding_contextual' in anchors_df.columns:
            emb_col = 'embedding_contextual'
        else:
            logger.error("Anchors dataframe missing embedding column")
            return {}

        for dim, group in anchors_df.groupby('dimension'):
            # Stack embeddings
            emb_list = np.stack(group[emb_col].values) # (N, d)
            # Centroid
            centroid = np.mean(emb_list, axis=0)
            dim_vectors[dim] = centroid
            
        # 2. Orthogonalize Dimensions (Löwdin)
        # We want the axes to be distinct (Social vs Functional etc).
        dims = list(dim_vectors.keys())
        V = np.stack([dim_vectors[d] for d in dims]).T # (feature, num_dims)
        
        # Löwdin Orthogonalization: V_orth = V @ (V^T V)^(-1/2)
        # Ensure V is centered? Directions usually from 0.
        
        # Gram Matrix
        G = np.dot(V.T, V)
        
        # Inverse Square Root of G
        # Eigen decomp G = Q L Q^T -> G^-1/2 = Q L^-1/2 Q^T
        evals, evecs = np.linalg.eigh(G)
        
        # Guard against small eigenvalues
        evals[evals < 1e-10] = 1e-10
        
        inv_sqrt_L = np.diag(1.0 / np.sqrt(evals))
        inv_sqrt_G = evecs @ inv_sqrt_L @ evecs.T
        
        V_orth = V @ inv_sqrt_G # (features, num_dims)
        
        # 3. Project Subspace onto Axes
        # For each axis v_i (column of V_orth):
        # Projection = || U^T @ v_i ||
        # Represents how much of the subspace is "aligned" with this axis.
        
        projections = {}
        for i, dim in enumerate(dims):
            v_orth = V_orth[:, i]
            # Normalize just in case, though Lowdin usually preserves if inputs normalized?
            # Lowdin yields orthonormal columns if inputs were independent.
            v_orth = v_orth / np.linalg.norm(v_orth)
            
            # Projection magnitude
            # vector v projected onto subspace U: P_U v = U U^T v.
            # We want norm of that: || U^T v ||
            
            proj_len = np.linalg.norm(np.dot(subspace_basis.T, v_orth))
            projections[dim] = float(proj_len)
            
        return projections
