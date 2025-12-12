import numpy as np
import pandas as pd
from scipy.stats import entropy
from src.analysis.subspaces import Subspace

class SociologicalMetrics:
    def __init__(self):
        pass

    def calculate_drift(self, subspaces: list):
        """
        Calculates the semantic stability (inverse of drift) between consecutive subspaces.
        Metric: Average Cosine Similarity of the aligned bases.
        
        Returns:
            pd.DataFrame with 'date', 'similarity', 'drift' columns.
        """
        results = []
        
        # First window has no predecessor, so drift is undefined (or 0)
        if not subspaces:
            return pd.DataFrame()
            
        results.append({
            'date': subspaces[0].label,
            'similarity': 1.0, 
            'drift': 0.0
        })
        
        for i in range(1, len(subspaces)):
            s_prev = subspaces[i-1]
            s_curr = subspaces[i]
            
            # 1. Similarity
            # We assume bases are already aligned by Procrustes in Construction phase.
            # We calculate mean absolute cosine similarity between corresponding basis vectors.
            # If k differs, we take min k.
            min_k = min(s_prev.k, s_curr.k)
            
            # Dot product of rows. Diagonals of product matrix if they are aligned row-by-row.
            # If V_curr approx V_prev, then V_curr @ V_prev.T should be Identity-like.
            # We want the trace of that product for the overlapping dimensions.
            
            # Basis shape: (k, features)
            # Alignment quality: Trace( | V_prev @ V_curr.T | ) / k
            
            alignment_matrix = np.dot(s_prev.basis[:min_k], s_curr.basis[:min_k].T)
            # We only care about diagonal (matching dimensions) if strictly aligned
            # But Procrustes aligns the whole subspace.
            # Let's use the trace of the absolute alignment matrix to capture total overlap?
            # Or better, Singular Values of the cross-product (Principal Angles).
            # "Cosine similarity of subspaces" = mean of cosines of principal angles.
            
            _, s, _ = np.linalg.svd(alignment_matrix)
            subspace_similarity = np.mean(s) # Mean of cosines of principal angles
            
            drift = 1.0 - subspace_similarity
            
            results.append({
                'date': s_curr.label,
                'similarity': subspace_similarity,
                'drift': drift
            })
            
        return pd.DataFrame(results)

    def _orthogonalize_anchors(self, anchor_vectors: dict) -> dict:
        """
        Applies Lowdin Symmetric Orthogonalization.
        Finds the orthogonal basis closest (Frobenius norm) to the original non-orthogonal vectors.
        This treats all dimensions equally, unlike Gram-Schmidt.
        
        Args:
           anchor_vectors: dict {name: np.array}
           
        Returns:
           dict: Orthogonalized vectors
        """
        # Identify the triplet of dimensions
        hierarchy = ['funcional', 'social', 'afectiva']
        suffixes = ['_contextual', '_static']
        
        processed_vectors = anchor_vectors.copy()
        
        for suffix in suffixes:
            # Gather the 3 vectors for this type
            keys = [f'{dim}{suffix}' for dim in hierarchy]
            
            # Check if all exist
            vectors = []
            valid_keys = []
            for k in keys:
                if k in anchor_vectors:
                    vectors.append(anchor_vectors[k])
                    valid_keys.append(k)
            
            if not vectors:
                continue
                
            # Matrix C of shape (d_dim, n_vecs) -> Transpose to (n_vecs, d_dim) for calculation?
            # Standard: Columns are vectors. shape (D, N)
            # C = [v1, v2, v3]
            C = np.stack(vectors, axis=1) # (3072, 3)
            
            # Singular Value Decomposition for S^(-1/2)
            # C = U Sigma V.T
            # Lowdin Orthogonalization: C_orth = U V.T (Polar Decomposition factor)
            # Or explicit formula: C_orth = C (C.T C)^(-1/2)
            
            # Let's use SVD for stability:
            # U, S, Vt = svd(C)
            # C_orth = U @ Vt
            
            try:
                # Thin SVD
                U, S_vals, Vt = np.linalg.svd(C, full_matrices=False)
                # This gives the closest orthogonal matrix (Polar Decomposition)
                C_orth = np.dot(U, Vt)
                
                # Assign back
                for i, k in enumerate(valid_keys):
                   processed_vectors[k] = C_orth[:, i]
                   
            except np.linalg.LinAlgError:
                print(f"Warning: SVD failed for {suffix}. Keeping originals.")
                
        return processed_vectors

    def calculate_projections(self, subspaces: list, anchors_df: pd.DataFrame, orthogonalize=True):
        """
        Projects the primary dimensions of each subspace onto the theoretical anchors.
        
        Args:
            subspaces: List of Subspace objects.
            anchors_df: DataFrame with 'keyword'/'dimension' and 'embedding_static'/'embedding_contextual'.
            orthogonalize: Bool. If True, applies Gram-Schmidt to anchors before projection.
                        
        Returns:
            pd.DataFrame with columns like 'score_funcional_contextual', 'score_funcional_static'.
        """
        # Prepare Anchor Vectors for both types
        anchor_vectors = {}
        dimensions = anchors_df['dimension'].unique()
        vector_types = ['contextual', 'static']
        
        for dim in dimensions:
            dim_data = anchors_df[anchors_df['dimension'] == dim]
            
            for v_type in vector_types:
                col_name = f'embedding_{v_type}'
                if col_name in dim_data.columns:
                    # Stack vectors
                    vectors = np.vstack(dim_data[col_name].values)
                    centroid = np.mean(vectors, axis=0)
                    centroid = centroid / np.linalg.norm(centroid)
                    anchor_vectors[f"{dim}_{v_type}"] = centroid
        
        # Apply Orthogonalization if requested
        if orthogonalize:
            anchor_vectors = self._orthogonalize_anchors(anchor_vectors)
            
        results = []
        
        for s in subspaces:
            row = {'date': s.label}
            
            # --- PROJECT THE CENTROID (Meaning Drift) ---
            # Now we have TWO centroids per subspace if we have dual data. 
            # Subspace object `s` currently only holds the main construction embedding (Contextual).
            # We need to compute the "Static-Compatible" Centroid on the fly or pass it in.
            
            # Since `subspaces` lists contain Subspace objects built from `embedding` (Contextual),
            # `s.centroid` is the Mean Contextual Vector.
            
            # We need the Static Centroid. 
            # For now, if the Subspace object doesn't have it, we calculate it from the raw window if provided.
            # But the metric class doesn't see raw data.
            # Hack: We will rely on s.centroid (Contextual) for Contextual Anchors.
            # For Static Anchors, we can't easily get the Static Centroid without raw data access.
            # However, the USER asked to use "Sum of Last 3" for static comparison.
            # Let's assume for now the user will re-run Phase 3 twice if they want full separation,
            # OR we update this method to accept an optional mapping of {date: static_centroid}.
            
            # Since Phase 3 constructor only takes one column, let's update `run_phase3_pipeline` to calculate 
            # the static centroid for each window and inject it into the valid subspaces list or a separate dict.
            
            # Let's try to infer it here or assume `s.extra_centroids` exists.
            
            # Retrieve extra_centroids from the new extra_data generic field
            extra = getattr(s, 'extra_data', {})
            extra_cents = extra.get('extra_centroids', {})
            
            centroid_ctx = s.centroid / np.linalg.norm(s.centroid)
            
            # Get static centroid if available
            centroid_static = extra_cents.get('static')
            if centroid_static is not None:
                centroid_static = centroid_static / np.linalg.norm(centroid_static)
            else:
                # Fallback to ctx if missing, but print warn? No, valid fallback for backwards compat
                centroid_static = centroid_ctx
            
            for key, anchor_vec in anchor_vectors.items():
                # Determine which centroid to use based on key suffix
                if '_static' in key:
                    current_centroid = centroid_static
                else:
                    current_centroid = centroid_ctx
                    
                score = np.dot(current_centroid, anchor_vec)
                row[f'score_centroid_{key}'] = score
                
            # 2. Basis Vectors Projection (Structure Orientation)
            # Check how each Latent Dimension of Yape aligns with the Anchors
            # We analyze up to the first 3 dimensions if available
            n_dims = len(s.basis)
            for i in range(min(n_dims, 3)):
                basis_vec = s.basis[i]
                # No need to normalize if SVD output is orthonormal, but safe to do so
                basis_vec = basis_vec / np.linalg.norm(basis_vec)
                
                for key, anchor_vec in anchor_vectors.items():
                    # Check dimension compatibility
                    if basis_vec.shape[0] != anchor_vec.shape[0]:
                        # Cannot project Concat-4 Basis onto Sum-3 Anchor
                        continue
                        
                    # Absolute dot product because orientation (+/-) in SVD is arbitrary
                    # We care about "Parallelism", not direction for Basis
                    score = abs(np.dot(basis_vec, anchor_vec))
                    row[f'score_dim{i+1}_{key}'] = score
                
            results.append(row)
            
        return pd.DataFrame(results)

    def calculate_entropy(self, subspaces: list):
        """
        Calculates the Shannon entropy of the singular values (normalized).
        High entropy = Meaning is distributed across many dimensions (Complex).
        Low entropy = Meaning is concentrated in Dim 1 (Simple/Monolithic).
        """
        results = []
        for s in subspaces:
            sv = s.eigenvalues
            # Normalize to prob distribution
            total_var = np.sum(sv)
            if total_var > 0:
                probs = sv / total_var
                ent = entropy(probs)
            else:
                ent = 0.0
                
            results.append({
                'date': s.label,
                'entropy': ent,
                'k': s.k
            })
            
        return pd.DataFrame(results)
