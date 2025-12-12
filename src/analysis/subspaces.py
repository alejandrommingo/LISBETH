import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from src.analysis.dimensionality import SubspaceAnalyzer

class Subspace:
    """
    Container for a semantic subspace at a specific time t.
    """
    def __init__(self, label, basis, centroid, eigenvalues, k, time_window=None, extra_data=None):
        self.label = label
        self.basis = basis # Shape (k, n_features) - The principal components
        self.centroid = centroid # Shape (n_features,) - The mean vector
        self.eigenvalues = eigenvalues
        self.k = k
        self.time_window = time_window # Dict with start_date, end_date
        self.extra_data = extra_data or {} # Container for other centroids (e.g. static)

class SubspaceConstructor:
    def __init__(self, analyzer: SubspaceAnalyzer = None):
        self.analyzer = analyzer if analyzer else SubspaceAnalyzer()
        self.subspaces = [] # List of Subspace objects in chronological order

    def build_subspaces(self, windows, fixed_k=None, align=True):
        """
        Constructs aligned subspaces for each provided window.
        
        Args:
            windows: Iterator or list of window dicts (from TemporalSegmenter).
            fixed_k: If set, forces specific k. If None, uses Horn's PA per window.
            align: If True, aligns basis V_t to V_{t-1} using Procrustes.
            
        Returns:
            List of Subspace objects.
        """
        self.subspaces = []
        previous_basis = None
        
        for w in windows:
            label = w['label']
            data_df = w['data']
            
            # Extract embeddings matrix (N_samples, N_features)
            # Ensure proper stacking
            X = np.vstack(data_df['embedding'].values)
            
            # 1. Centering (Crucial for PCA/SVD interpretation of variance)
            centroid = np.mean(X, axis=0)
            X_centered = X - centroid
            
            # 1.1 Calculate Extra Centroids (Static-Compatible) if available
            extra_centroids = {}
            if 'embedding_static' in data_df.columns:
                # Check for nulls or issues
                try:
                    X_static = np.vstack(data_df['embedding_static'].values)
                    centroid_static = np.mean(X_static, axis=0)
                    extra_centroids['static'] = centroid_static
                except Exception as e:
                    print(f"Warning: Failed to calc static centroid for {label}: {e}")
            
            # 2. Dimensionality Selection
            if fixed_k is not None:
                k = fixed_k
                # Get eigenvalues anyway for info
                all_sv = self.analyzer.get_eigenvalues(X_centered)
                eigenvalues = all_sv[:k] if len(all_sv) >= k else all_sv
            else:
                # Use Horn's PA
                # Note: Horn's PA inside analyzer generates random noise comparison
                k, eigenvalues, _ = self.analyzer.horns_parallel_analysis(X_centered)
                # Fail-safe for very low k
                if k < 2: k = 2 
            
            # 3. SVD Decomposition
            svd = TruncatedSVD(n_components=k, random_state=42)
            svd.fit(X_centered)
            basis = svd.components_ # Shape (k, n_features)
            
            # 4. Procrustes Alignment
            if align and previous_basis is not None:
                # We want to align current 'basis' to 'previous_basis'.
                # Both should have shape (k, n_features).
                # If k differs, we align on the intersection min(k_t, k_{t-1}).
                # For simplicity in this TFM, we usually assume k is relatively stable 
                # or we just align the available dimensions.
                
                min_k = min(basis.shape[0], previous_basis.shape[0])
                
                # Slicing to intersection
                A = previous_basis[:min_k, :]
                B = basis[:min_k, :]
                
                # orthogonal_procrustes calculates R such that A is close to B @ R
                # Actually solves ||A - B @ R||_F.
                # Here our rows are dimensions (components). 
                # We want to rotate the *axes* (rows of basis).
                # Let's consider standard definition:
                # We want V_new approx V_old.
                # R, scale = orthogonal_procrustes(B.T, A.T) -> NO, this is for shapes
                
                # We use scipy.linalg.orthogonal_procrustes(A, B) 
                # solving ||A - B @ R||. 
                # A and B are (M, N). 
                # We treat the K components as M "points" in N-dim space? 
                # Or N features as points in K-dim space?
                # We want to align the coordinate systems.
                # Valid approach: Align the axes vectors themselves.
                # A=Prev(k, N), B=Curr(k, N).
                # R minimizes ||Prev.T - Curr.T @ R|| ? 
                # Usually we align the transpose: (N, k).
                
                # Standard approach for Subspace Alignment (e.g. Hamilton et al):
                # R = OPA(Current.T, Previous.T) -> Current.T @ R approx Previous.T
                
                R, _ = orthogonal_procrustes(basis.T, previous_basis.T)
                
                # Apply rotation to basis
                # aligned = (basis.T @ R).T = R.T @ basis
                basis_aligned = np.dot(basis.T, R).T
                
                # If sizes mismatched, we only aligned the top min_k components
                # But we need to keep full basis.
                # For this implementation, let's assume we update the top min_k rows
                if min_k < basis.shape[0]:
                    basis[:min_k, :] = basis_aligned
                    # Lower components kept unaligned (or could rotate them too if R was full rank?)
                    # R is (min_k, min_k) if we passed slices.
                    # Wait, orthogonal_procrustes(A, B) requires same shape.
                    # So R is derived from top components. 
                    # We accept this limitation for varying k.
                    pass 
                else:
                    basis = basis_aligned
            
            previous_basis = basis
            
            # Store
            subspace = Subspace(
                label=label,
                basis=basis,
                centroid=centroid,
                eigenvalues=eigenvalues,
                k=k,
                time_window=w,
                extra_data={'extra_centroids': extra_centroids}
            )
            self.subspaces.append(subspace)
            
        return self.subspaces
