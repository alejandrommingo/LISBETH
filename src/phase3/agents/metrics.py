import numpy as np
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from src.phase3.schemas import Phase3Config

logger = logging.getLogger(__name__)

class MetricCalculator:
    """
    Combines Agents 8, 9, 10 into a cohesive calculator.
    """
    
    def load_anchors(self, variant: str, strategy: str) -> Dict[str, np.ndarray]:
        """
        Loads anchors for a specific run.
        Returns dictionary {dimension: vector(d,)}?
        No, Protocol says 'A shape (d, 3)'.
        Agente 9/10 inputs: 'anchors_*.npz'.
        """
        filename = f"anchors_{variant}_{strategy}.npz"
        path = Phase3Config.ANCHORS_DIR / filename
        if not path.exists():
            raise RuntimeError(f"FAIL: Anchors file not found: {path}")
            
        data = np.load(path)
        A = data['A'] # (d, 3)
        dims = data['dimensions'] # ["funcional", "social", "afectiva"]
        
        # Create map for easy access
        # Columns of A correspond to dimensions in 'dims'
        anchor_map = {}
        for i, dim in enumerate(dims):
            anchor_map[dim] = A[:, i]
            
        return anchor_map, A
        
    def calculate_entropy(self, singular_values: np.ndarray) -> float:
        """Agent 8: Entropy"""
        if len(singular_values) == 0: return 0.0
        
        s_sum = np.sum(singular_values)
        if s_sum == 0: return 0.0
        
        p = singular_values / s_sum
        # Avoid log(0)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        return float(entropy)

    def calculate_drift_procrustes(self, U_prev: Optional[np.ndarray], U_curr: np.ndarray) -> Tuple[float, float]:
        """Agent 8: Drift & Procrustes (Grassmann + Procrustes)"""
        if U_prev is None:
            return np.nan, np.nan
            
        # Ensure shapes compatible
        # U is (d, k). k can change!
        # Grassmann drift: Principal angles between subspaces of different dimensions?
        # Standard def: cosines are singular values of U_prev.T @ U_curr.
        # Angles usually defined for min rank.
        
        # 1. Grassmann Drift
        # M = U_prev.T @ U_curr
        # s = svd(M)
        # theta = arccos(clip(s, 0, 1))
        # drift = norm(theta)
        
        M = U_prev.T @ U_curr
        # full_matrices=False, compute_uv=False for just singular values
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        
        # Clip for numerical stability
        s_clipped = np.clip(s, 0.0, 1.0)
        thetas = np.arccos(s_clipped)
        drift = float(np.linalg.norm(thetas))
        
        # 2. Procrustes
        # Optimally rotate U_curr to match U_prev (or vice versa? "U_curr R - U_prev")
        # Procrustes usually requires same shape. If k differs, we pad with zeros?
        # Protocol: "estimar error residual = ||U_curr R - U_prev||_F"
        # If dimensions differ, orthogonal Procrustes is tricky.
        # "Generalized Procrustes" or just pad the smaller logical subspace with 0s?
        # If k changes, the "space" changes.
        # A common approach if k1 != k2:
        # Align the shared subspace.
        # For simplicity (and strictness?), let's assume we pad to max(k1, k2) or project?
        # Let's check strict protocol: "estimar rotación R óptima (SVD de U_curr.T @ U_prev)".
        # This implies standard Orthogonal Procrustes for non-square?
        # Scipy orthogonal_procrustes requires same shape.
        # I will pad with zeros to match larger k. This assumes the "missing" dimensions are 0.
        
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
        
        # Scipy orthogonal_procrustes(A, B) finds R s.t. ||A @ R - B|| is min.
        # We want to rotate U_curr to match U_prev.
        # R -> U_curr @ R approx U_prev.
        # svd(U_curr.T @ U_prev) -> U_, s_, Vh_. R = U_ @ Vh_.
        
        m_cross = U_c_pad.T @ U_p_pad
        u_p, _, vh_p = np.linalg.svd(m_cross)
        R = u_p @ vh_p
        
        diff = U_c_pad @ R - U_p_pad
        procrustes_error = float(np.linalg.norm(diff, ord='fro'))
        
        return drift, procrustes_error

    def calculate_centroid_projection(self, mu: np.ndarray, anchor_map: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Agent 9: Project Centroid onto Anchors"""
        # "Normalizar mu: mu_hat = mu / ||mu||"
        norm_mu = np.linalg.norm(mu)
        if norm_mu == 0:
            raise RuntimeError("FAIL: Centroid norm is 0")
        
        mu_hat = mu / norm_mu
        
        results = {}
        for dim, a_vec in anchor_map.items():
            # centroid_proj_j = mu_hat.T @ a_j
            proj = float(np.dot(mu_hat, a_vec))
            results[f"centroid_proj_{dim}"] = proj
            
        return results

    def calculate_subspace_projection(self, U: np.ndarray, anchor_map: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Agent 10: Project Subspace onto Anchors"""
        # "subspace_proj_j = || U.T @ a_j ||_2"
        results = {}
        for dim, a_vec in anchor_map.items():
            # U is (d, k), a_vec is (d,)
            # proj_vec is (k,)
            proj_vec = U.T @ a_vec
            norm_proj = float(np.linalg.norm(proj_vec))
            results[f"subspace_proj_{dim}"] = norm_proj
            
        return results
