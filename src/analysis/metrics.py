import numpy as np
from scipy.linalg import svd, orth, norm
from sklearn.utils import resample

def compute_subspace(X, k=None):
    """
    Computes subspace basis U and singular values S from matrix X.
    X: (n_samples, d_features)
    Returns: U (d, k), S (k,), k
    """
    # Centering
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    # X = U_full S V_T -> In numpy svd(X) returns u, s, vh
    # But usually for PCA/Subspace we want eigenvectors of Covariance matrix X.T @ X 
    # Or Right Singular Vectors of X (V).
    # Shapes: X (n,d). U (n,n), S (min(n,d)), Vh (d,d)
    # The basis of the row space (which captures variance) are the rows of Vh => V columns.
    
    u, s, vh = svd(X_centered, full_matrices=False)
    
    # Intrinsic dimensionality selection if k not provided (e.g. 95% variance)
    # But we expect k to be provided from Horn's.
    if k is None:
        # Default fallback: 95% variance
        var_explained = np.cumsum(s**2) / np.sum(s**2)
        k = np.searchsorted(var_explained, 0.95) + 1
        
    basis = vh[:k, :].T # (d, k) - Columns are basis vectors
    return basis, s[:k], k

def horn_parallel_analysis(X, n_iter=20, p_value=0.05):
    """
    Determines intrinsic dimensionality k using Horn's Parallel Analysis.
    Compares real eigenvalues vs random shuffle eigenvalues.
    """
    n, d = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    # Real eigenvalues (of covariance) ~ s**2
    _, s_real, _ = svd(X_centered, full_matrices=False)
    eig_real = s_real**2
    
    # Synthetic eigenvalues
    eig_synth_dist = []
    for _ in range(n_iter):
        X_synth = np.apply_along_axis(np.random.permutation, 0, X_centered) # Shuffle each column
        _, s_synth, _ = svd(X_synth, full_matrices=False)
        eig_synth_dist.append(s_synth**2)
        
    eig_synth_dist = np.array(eig_synth_dist) # (n_iter, min(n,d))
    
    # Thresholds (e.g. 95th percentile of random noise)
    eig_synth_thresh = np.percentile(eig_synth_dist, 100 * (1 - p_value), axis=0)
    
    # Select k where real > synth
    # Ensure min(n,d) length match
    limit = min(len(eig_real), len(eig_synth_thresh))
    k_horn = np.sum(eig_real[:limit] > eig_synth_thresh[:limit])
    
    return max(1, k_horn), eig_real, eig_synth_thresh

def orthogonal_procrustes(U_target, U_source):
    """
    Aligns U_source to U_target using rotation R.
    Minimizes || U_target - U_source @ R ||_F
    Returns: U_aligned, R, error
    """
    # M = Source^T @ Target
    M = U_source.T @ U_target
    u, s, vh = svd(M)
    
    # R = u @ vh
    R = u @ vh
    
    U_aligned = U_source @ R
    error = norm(U_target - U_aligned, ord='fro')
    return U_aligned, R, error

def grassmannian_distance(U1, U2):
    """
    Computes distance between two subspaces U1, U2.
    Based on principal angles.
    d = sqrt(sum(theta_i^2)) or max(theta_i) or ||sin(theta)||_2
    We use standard Binet-Cauchy or projection metric.
    Robust Implementation: Projection metric = || P1 - P2 ||_F / sqrt(2) = || sin theta ||_2
    Angle based: theta_i = arccos(singular values of U1.T @ U2)
    """
    # Ensure orthogonality
    # SVD of overlap matrix
    # Assumes U1, U2 are orthonormal basis (d, k)
    
    # U1.T @ U2 -> singular values correspond to cos(theta)
    S = svd(U1.T @ U2, compute_uv=False)
    
    # Clip for numerical stability
    S = np.clip(S, 0.0, 1.0)
    
    thetas = np.arccos(S)
    dist = np.linalg.norm(thetas) # Geodesic distance
    
    return dist

def semantic_entropy(singular_values):
    """
    Computes entropy of the normalized singular value spectrum.
    Represents complexity/richness of the semantic space.
    """
    s_sq = singular_values**2
    total_var = np.sum(s_sq)
    if total_var == 0: return 0.0
    
    probs = s_sq / total_var
    # Filter zeros for log
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log(probs))
    # Normalized entropy (0-1)? Ideally yes, but standard Shannon is fine for relative comparison.
    # Theoretical max is log(k).
    return entropy

def lowdin_orthogonalization(vectors):
    """
    Symmetric orthogonalization (Löwdin).
    vectors: (d, m) matrix where columns are vectors to orthogonalize.
    Preserves maximum similarity to original vectors.
    """
    ndim, m = vectors.shape
    
    # Gram matrix
    S = vectors.T @ vectors
    
    # S^(-1/2)
    # Eigendecomp S = V D V.T
    d, V = np.linalg.eigh(S)
    
    # Inverse squareroot of eigenvalues
    # Handle small eigenvalues
    d_inv_sqrt = np.array([1.0/np.sqrt(x) if x > 1e-10 else 0.0 for x in d])
    
    S_inv_sqrt = V @ np.diag(d_inv_sqrt) @ V.T
    
    # V_orth = V_orig @ S^(-1/2) 
    # Check dimensions? 
    # Usually X_orth = X @ S^(-1/2) if rows are samples.
    # Here columns are vectors.
    # Let's assume input X is (m, d) for standard formula, but here (d, m)
    # Formula for columns: V_orth = V @ S^(-1/2) works?
    # (V S^-1/2).T (V S^-1/2) = S^-1/2 S S^-1/2 = I
    
    V_orth = vectors @ S_inv_sqrt
    return V_orth

def project_on_frame(subspace_U, frame_vectors):
    """
    Projects subspace U onto the frame axes.
    Returns square cosine similarity (energy captured).
    frame_vectors: (d, m) orthogonal columns.
    subspace_U: (d, k) orthogonal columns.
    
    Projection of axis v onto U: || U.T @ v ||^2
    """
    # Projections: vector of size m
    # For each frame axis j: measure overlap with U
    
    projections = []
    for j in range(frame_vectors.shape[1]):
        v = frame_vectors[:, j]
        # P = || U^T v ||
        p_val = norm(subspace_U.T @ v)
        projections.append(p_val) # This is cos(theta) approx, square it?
        # User defined: P = || U.T v ||. This is the length of projection. 
        # Range 0-1 if v is unit.
        
    return np.array(projections)
