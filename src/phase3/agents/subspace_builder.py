import numpy as np
import pandas as pd
import json
import logging
from typing import Tuple, Dict, Any, List
from scipy.linalg import svd
from sklearn.utils import resample
from src.phase3.schemas import Phase3Config

logger = logging.getLogger(__name__)

class Agent4MatrixBuilder:
    """
    Agente 4 — BUILDER DE MATRICES CRUDAS + CENTROIDES (pre-centering)
    """
    def run(self, window_df: pd.DataFrame, variant: str, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        # Select column
        col_map = {
            ("baseline", "penultimate"): Phase3Config.COL_EMB_BASELINE_PENULTIMATE,
            ("baseline", "last4_concat"): Phase3Config.COL_EMB_BASELINE_LAST4,
            ("dapt", "penultimate"): Phase3Config.COL_EMB_DAPT_PENULTIMATE,
            ("dapt", "last4_concat"): Phase3Config.COL_EMB_DAPT_LAST4,
        }
        col = col_map.get((variant, strategy))
        if not col:
            raise ValueError(f"Unknown combination: {variant} {strategy}")
            
        # Parse
        try:
            # We assume agent 1 checked format, but safe to be robust
            # Using list comprehension which is faster than apply for large jsons
            matrix = np.array([json.loads(x) for x in window_df[col]])
        except Exception as e:
            raise RuntimeError(f"FAIL: Matrix parse error for {variant}/{strategy}: {e}")
            
        if matrix.ndim != 2:
             raise RuntimeError(f"FAIL: Matrix has wrong dims: {matrix.shape}")
             
        # Validate finite
        if not np.all(np.isfinite(matrix)):
             raise RuntimeError(f"FAIL: Matrix contains NaNs or Infs")
             
        # Calc mu
        mu = np.mean(matrix, axis=0)
        
        return matrix, mu

class Agent5Centerer:
    """
    Agente 5 — CENTRADO
    """
    def run(self, X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        Xc = X - mu
        if not np.all(np.isfinite(Xc)):
            raise RuntimeError("FAIL: NaNs post-centering")
        return Xc

class Agent6KSelector:
    """
    Agente 6 — SELECTOR DE k (Horn + Bootstrap)
    """
    def run(self, Xc: np.ndarray, B_HORN: int = 200, B_BOOT: int = 200, seed: int = 42) -> Tuple[int, int, int]:
        n, d = Xc.shape
        rng = np.random.RandomState(seed)
        
        # 1. Real Eigenvalues (PCA)
        # We need eigenvalues of covariance matrix C = Xc.T @ Xc / (n-1)
        # Or singular values of Xc: s. lambda = s^2 / (n-1)
        # Economy SVD is O(n d^2) or O(d n^2). n is usually < 1000 here? 
        # Window size: 20 minimum. 
        # If n < d, we get n singular values.
        
        # For Horn's PA, we typically compare eigenvalues of correlation or covariance matrix.
        # Here we use covariance (implied by centering).
        
        # Compute real singular values
        # svd returns s sorted descending
        _, s_real, _ = np.linalg.svd(Xc, full_matrices=False)
        eigen_real = (s_real ** 2) / (n - 1)
        
        # 2. Horn's Parallel Analysis
        # Generate random noise matrices of same shape (n, d)
        # Permutation strategy is often better than random normal for non-normal data,
        # but "normal(0,1)" was suggested as option. I'll use permutation per column 
        # to preserve marginal distributions but destroy correlations (standard PA).
        
        rand_eigen_accum = []
        for _ in range(B_HORN):
            # Efficient permutation
            X_rand = np.zeros_like(Xc)
            for j in range(d):
                X_rand[:, j] = rng.permutation(Xc[:, j])
            
            # Recenter random matrix? PA usually assumes generated data is centered.
            # Permutation preserves mean, so if Xc col mean is 0, X_rand col mean is 0.
            
            _, s_rand, _ = np.linalg.svd(X_rand, full_matrices=False)
            e_rand = (s_rand ** 2) / (n - 1)
            rand_eigen_accum.append(e_rand)
            
        rand_eigen_accum = np.array(rand_eigen_accum) # (B, min(n,d))
        
        # 95th percentile
        eigen_rand95 = np.percentile(rand_eigen_accum, 95, axis=0)
        
        # k_horn
        # How many real > random?
        k_horn = 0
        min_dim = min(n, d)
        for i in range(min_dim):
            if eigen_real[i] > eigen_rand95[i]:
                k_horn += 1
            else:
                break
                
        # 3. Bootstrap Stability
        # We want to see how many dimensions are "stable" to resampling?
        # A common heuristic is: how many eigenvalues are consistently > 0 or separated?
        # Protocol: "Definir criterio de estabilidad -> derivar k_bootstrap"
        # STRICT RULE: Document in run_manifest.
        # My Metric:
        # Cross-validation error? reconstruction?
        # Or: Overlap of subspaces?
        # Let's use a simpler eigenvalue stability metric for robustness (as complex bootstrap methods are slow).
        # "Significant eigenvalues" via bootstrap intervals.
        # If the lower bound of CI(eigenvalue_k) > upper bound of CI(eigenvalue_k+1)? No, that's separation.
        # Let's stick effectively to: k_bootstrap = median k from Horn on bootstrap samples?
        # Or simply: k that explains X% variance? No, protocol forbids arbitrary cutoffs.
        
        # I will implement: "Bootstrap k_horn". 
        # Calculate k_horn on B_BOOT resampled datasets. Take the median k or min k.
        # This keeps the logic consistent.
        
        boot_ks = []
        for _ in range(B_BOOT):
            # Resample Xc
            X_boot = resample(Xc, replace=True, n_samples=n, random_state=rng)
            # Center boot sample
            X_boot = X_boot - np.mean(X_boot, axis=0) 
            
            # Quick check against original random baseline? 
            # Re-running full PA B_BOOT times is B_HORN * B_BOOT SVDs -> Too expensive.
            
            # Alternative Bootstrap Criterio:
            # "DaPT" paper often uses "k such that cumulative variance > 90%"? No.
            # Let's use a fail-safe: average k_horn is robust but expensive. 
            
            # Let's use a Simpler Bootstrap stability:
            # SVD on resample.
            # Look at variance of eigenvalues.
            # If std dev of lambda_k is large compared to gap (lambda_k - lambda_k+1), it's unstable.
            # But we get k_selected = min(k_horn, k_boot).
            # If k_boot is not calculated, we compromise the protocol.
            
            # Proposed FAST Bootstrap strategy:
            # Use Kaiser Criterion on bootstrap samples? (Eigen > mean eigenvalue).
            # Or just return k_horn as k_bootstrap for now if computational limit?
            # User Protocol says: "k_selected = min(k_horn, k_bootstrap)".
            # I will set k_bootstrap to be visually conservative:
            # Count how many eigenvalues are "significant" ( > sorted random noise mean?).
            
            # Let's re-read carefully: "Repetir B_BOOT: resample... calcular eigenvalues. Definir criterio...".
            # I'll implement: For each bootstrap sample, count component where lambda > lambda_noise_95 (precomputed from original).
            # This is "Does this sample support this component against the ORIGINAL noise model?"
            
            _, s_boot, _ = np.linalg.svd(X_boot, full_matrices=False)
            e_boot = (s_boot ** 2) / (n - 1)
            
            k_b = 0
            for i in range(min(len(e_boot), len(eigen_rand95))):
                if e_boot[i] > eigen_rand95[i]:
                    k_b += 1
                else: 
                    break
            boot_ks.append(k_b)
            
        k_bootstrap_val = int(np.floor(np.percentile(boot_ks, 5))) # Conservative: 5th percentile of supported k
        
        k_selected = min(k_horn, k_bootstrap_val)
        if k_selected < 1: k_selected = 1
        
        return k_horn, k_bootstrap_val, k_selected

class Agent7SubspacePersister:
    """
    Agente 7 — SUBESPACIOS + PERSISTENCIA
    """
    def run(
        self, 
        Xc: np.ndarray, 
        mu: np.ndarray, 
        k: int, 
        window_meta: Dict, 
        variant: str, 
        strategy: str
    ) -> str:
        # SVD
        # U, s, Vh = svd(Xc)
        # U is (n, n) or (n, d).
        # We want the principal components in feature space.
        # If Xc is (n, sample) x (d, feature)
        # PCA eigenvectors are V (d x d).
        # SVD: X = U S V^T. V is (d, d). Columns of V are principal directions.
        # Agent 3's "U" (subspace) usually refers to the Basis.
        # Protocol: "U (float32) shape (d, k)". This confirms we want V^T rows 0..k, transposed.
        # numpy svd: u, s, vh. vh is V^T. rows of vh are eigenvectors.
        # So we want vh[:k, :].T -> shape (d, k).
        
        u, s, vh = np.linalg.svd(Xc, full_matrices=False)
        
        # Check shapes
        # s shape (min(n, d),)
        # vh shape (min(n, d), d)
        
        if k > len(s):
             raise RuntimeError(f"FAIL: k={k} > rank={len(s)}")
             
        U_subspace = vh[:k, :].T # (d, k)
        singular_values = s
        
        # Validate
        if not np.all(np.isfinite(U_subspace)):
             raise RuntimeError("FAIL: NaNs in Subspace U")
        
        filename = f"window_{window_meta['start']}_{variant}_{strategy}.npz"
        path = Phase3Config.SUBSPACES_DIR / filename
        
        np.savez_compressed(
            path,
            U=U_subspace,
            singular_values=singular_values,
            mean_vector=mu,
            k_selected=k,
            window_start_month=window_meta['start'],
            window_end_month=window_meta['end'],
            variant=variant,
            strategy=strategy
        )
        return str(path)
