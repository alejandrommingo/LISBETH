import pytest
import numpy as np
from src.analysis.dimensionality import SubspaceAnalyzer

@pytest.fixture
def synthetic_data():
    """
    Creates a (100, 50) matrix with 3 strong signal dimensions + noise.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # 3 Latent variables
    latents = np.random.randn(n_samples, 3)
    # Mixing matrix
    mixing = np.random.randn(3, n_features)
    # Signal
    signal = np.dot(latents, mixing)
    
    # Noise
    noise = np.random.randn(n_samples, n_features) * 0.5 # Noise level
    
    data = signal + noise
    return data

def test_horns_parallel_analysis(synthetic_data):
    analyzer = SubspaceAnalyzer(random_state=42)
    k, real_sv, thresholds = analyzer.horns_parallel_analysis(synthetic_data, num_simulations=20)
    
    # We expect roughly 3 dimensions
    print(f"Optimal k found: {k}")
    assert k == 3, f"Expected k=3, got {k}"
    
    # First 3 eigenvalues should be well above threshold
    assert np.all(real_sv[:3] > thresholds[:3])
    # 4th eigenvalue likely below or close to threshold
    if len(real_sv) > 3:
         assert real_sv[3] < thresholds[3] * 1.5 # Allow some margin, but ideally < threshold

def test_bootstrap_stability(synthetic_data):
    analyzer = SubspaceAnalyzer(random_state=42)
    # Stable subspace (k=3)
    mean_sim, std_sim = analyzer.bootstrap_stability(synthetic_data, k=3, n_boot=10)
    print(f"Stability k=3: {mean_sim} +/- {std_sim}")
    
    assert mean_sim > 0.8 # Should be very stable
    
    # Unstable subspace (k=20, mostly noise)
    mean_sim_noise, _ = analyzer.bootstrap_stability(synthetic_data, k=20, n_boot=10)
    print(f"Stability k=20: {mean_sim_noise}")
    
    # Ideally subspace with noise is less stable implies lower cosine similarity on average 
    # for the noise components, pulling down the average.
    assert mean_sim_noise < mean_sim

def test_pure_noise():
    np.random.seed(42)
    noise_data = np.random.randn(100, 50)
    analyzer = SubspaceAnalyzer(random_state=42)
    k, _, _ = analyzer.horns_parallel_analysis(noise_data, num_simulations=20)
    
    # Should find 0 or nearly 0 dimensions
    assert k <= 1, f"Found {k} dimensions in pure noise"

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
