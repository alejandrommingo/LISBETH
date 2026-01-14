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


class MetricCalculator:
    """
    Combines Metric Calculators into a cohesive class.
    """
    
    def load_anchors(self, variant: str, strategy: str, condition: str = "raw") -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Loads anchors for a specific run.
        """
        from src.subspace_analysis.schemas import Phase3Config
        filename = f"anchors_{variant}_{strategy}.npz"
        path = Phase3Config.ANCHORS_DIR / filename
        if not path.exists():
            raise RuntimeError(f"FAIL: Anchors file not found: {path}")
            
        data = np.load(path)
        A = data['A'] # (d, 3)
        dims = data['dimensions'] # ["funcional", "social", "afectiva"]
        
        # Create map for easy access
        anchor_map = {}
        for i, dim in enumerate(dims):
            anchor_map[dim] = A[:, i]
            
        return anchor_map, A
        
    def calculate_entropy(self, singular_values: np.ndarray) -> float:
        """Entropy Calculator"""
        if len(singular_values) == 0: return 0.0
        
        s_sum = np.sum(singular_values)
        if s_sum == 0: return 0.0
        
        p = singular_values / s_sum
        # Avoid log(0)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        return float(entropy)

    def calculate_drift_procrustes(self, U_prev: np.ndarray | None, U_curr: np.ndarray) -> tuple[float, float]:
        """Drift & Procrustes Calculator (Grassmann + Procrustes)"""
        if U_prev is None:
            return np.nan, np.nan
            
        # 1. Grassmann Drift
        M = U_prev.T @ U_curr
        # full_matrices=False, compute_uv=False for just singular values
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        
        # Clip for numerical stability
        s_clipped = np.clip(s, 0.0, 1.0)
        thetas = np.arccos(s_clipped)
        drift = float(np.linalg.norm(thetas))
        
        # 2. Procrustes
        k_prev = U_prev.shape[1]
        k_curr = U_curr.shape[1]
        k_max = max(k_prev, k_curr)
        
        def pad(U, k_target):
            n, k = U.shape
            if k < k_target:
                return np.hstack([U, np.zeros((n, k_target - k))])
            return U
            
        U_p_pad = pad(U_prev, k_max)
        U_c_pad = pad(U_curr, k_max)
        
        m_cross = U_c_pad.T @ U_p_pad
        u_p, _, vh_p = np.linalg.svd(m_cross)
        R = u_p @ vh_p
        
        diff = U_c_pad @ R - U_p_pad
        procrustes_error = float(np.linalg.norm(diff, ord='fro'))
        
        return drift, procrustes_error

    def calculate_centroid_projection(self, mu: np.ndarray, anchor_map: dict[str, np.ndarray]) -> dict[str, float]:
        """Project Centroid onto Anchors"""
        norm_mu = np.linalg.norm(mu)
        if norm_mu == 0:
            raise RuntimeError("FAIL: Centroid norm is 0")
        
        mu_hat = mu / norm_mu
        
        results = {}
        for dim, a_vec in anchor_map.items():
            proj = float(np.dot(mu_hat, a_vec))
            results[f"centroid_proj_{dim}"] = proj
            
        return results

    def calculate_subspace_projection(self, U: np.ndarray, anchor_map: dict[str, np.ndarray]) -> dict[str, float]:
        """Project Subspace onto Anchors"""
        results = {}
        for dim, a_vec in anchor_map.items():
            proj_vec = U.T @ a_vec
            norm_proj = float(np.linalg.norm(proj_vec))
            results[f"subspace_proj_{dim}"] = norm_proj
            
        return results
