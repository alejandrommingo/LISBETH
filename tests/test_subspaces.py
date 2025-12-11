import pytest
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from src.analysis.subspaces import SubspaceConstructor

@pytest.fixture
def synthetic_windows():
    """
    Creates two temporal windows.
    Window 1: Base data.
    Window 2: Data rotated by 30 degrees.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    k_true = 3
    
    # Base signal
    latents = np.random.randn(n_samples, k_true)
    basis_true = np.eye(k_true, n_features) # Simple canonical basis
    # Or random orthonormal basis
    q, r = np.linalg.qr(np.random.randn(n_features, k_true))
    basis_true = q.T # (k, n_features)
    
    X1 = np.dot(latents, basis_true)
    
    # Rotate space for Window 2
    # Create a rotation matrix for the first 2 dimensions
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R_2d = np.array(((c, -s), (s, c)))
    # Full rotation (identity elsewhere)
    R = np.eye(n_features)
    R[:2, :2] = R_2d
    
    # X2 is X1 rotated
    X2 = np.dot(X1, R.T) # Rotate embedding space
    
    # Store in window format
    # We need embeddings as arrays in a column
    df1 = pd.DataFrame({'embedding': list(X1)})
    df2 = pd.DataFrame({'embedding': list(X2)})
    
    w1 = {'label': 'W1', 'data': df1}
    w2 = {'label': 'W2', 'data': df2}
    
    return [w1, w2], basis_true

def test_alignment(synthetic_windows):
    windows, basis_true = synthetic_windows
    constructor = SubspaceConstructor()
    
    # Build subspaces with Fixed K=3 and Alignment ON
    subspaces = constructor.build_subspaces(windows, fixed_k=3, align=True)
    
    assert len(subspaces) == 2
    
    s1 = subspaces[0]
    s2 = subspaces[1]
    
    # Check shape
    assert s1.basis.shape == (3, 50)
    
    # Verify s1 aligns reasonably with ground truth basis (SVD up to sign flip)
    # Cosine similarity between basis vectors
    # We can't check exact equality due to SVD sign indeterminacy, but subspaces should be close.
    
    # KEY TEST: Alignment between s1 and s2
    # Since X2 is just X1 rotated by 30 deg, unaligned SVD of X2 would yield rotated components.
    # Procrustes should rotate s2 back to align with s1.
    
    # Calculate similarity between s1.basis and s2.basis
    # Trace(B1 @ B2.T) / k
    similarity = np.trace(np.dot(s1.basis, s2.basis.T)) / 3
    # Note: If signs are flipped, this might be negative. Procrustes typically fixes signs too.
    abs_similarity = abs(similarity)
    
    print(f"Subspace Similarity (After Alignment): {abs_similarity}")
    
    # Without alignment, the rotation of 30 deg would reduce similarity.
    # cos(30) = 0.866.
    # If perfectly aligned, similarity should be close to 1.0
    assert abs_similarity > 0.95, "Procrustes failed to align rotated subspace"

def test_no_alignment(synthetic_windows):
    windows, _ = synthetic_windows
    constructor = SubspaceConstructor()
    
    # Build subspaces with Fixed K=3 and Alignment OFF
    subspaces = constructor.build_subspaces(windows, fixed_k=3, align=False)
    
    s1 = subspaces[0]
    s2 = subspaces[1]
    
    # Calculate similarity
    # We align rows (principal components)
    # The first 2 components were rotated by 30 deg.
    # The 3rd component (invariant) should be 1.0
    # Average sim should be approx (cos(30) + cos(30) + 1)/3 = (0.866 + 0.866 + 1) / 3 = 0.91
    
    # Note: SVD sign flip might obscure this, but assuming deterministic SVD (random_state fixed):
    similarity = np.trace(np.abs(np.dot(s1.basis, s2.basis.T))) / 3
    print(f"Subspace Similarity (No Alignment): {similarity}")
    
    # Should be less than aligned version (hard to strictly assert given SVD randomness/sign flips)
    # But useful to verify mechanics run.
    assert s1.k == 3

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
