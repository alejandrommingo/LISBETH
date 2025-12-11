import pytest
import numpy as np
import pandas as pd
from src.analysis.metrics import SociologicalMetrics
from src.analysis.subspaces import Subspace

@pytest.fixture
def mock_anchors():
    """
    Creates dummy anchors for 'social' and 'funcional'.
    """
    # Create orthogonal vectors
    vec_social = np.zeros(10)
    vec_social[0] = 1.0 # Logic: Social is purely Dim 0
    
    vec_funcional = np.zeros(10)
    vec_funcional[1] = 1.0 # Logic: Functional is purely Dim 1
    
    # DataFrame structure
    df = pd.DataFrame([
        {'dimension': 'social', 'embedding_contextual': vec_social},
        {'dimension': 'funcional', 'embedding_contextual': vec_funcional}
    ])
    return df

@pytest.fixture
def mock_subspaces():
    """
    Creates a sequence of subspaces.
    S1: Aligned with Social
    S2: Aligned with Social (Stable)
    S3: Aligned with Functional (Drift)
    """
    feature_dim = 10
    
    # S1: Basis 0 = Social vector
    basis1 = np.zeros((2, feature_dim))
    basis1[0, 0] = 1.0 # Dim 0
    basis1[1, 2] = 1.0 # Dim 2 (just another orthogonal dim)
    s1 = Subspace("Jan", basis1, np.zeros(feature_dim), np.array([10, 1]), 2)
    
    # S2: Identical to S1
    s2 = Subspace("Feb", basis1, np.zeros(feature_dim), np.array([10, 1]), 2)
    
    # S3: Basis 0 = Functional vector
    basis3 = np.zeros((2, feature_dim))
    basis3[0, 1] = 1.0 # Dim 1 (Functional)
    basis3[1, 3] = 1.0 # Dim 3 (Orthogonal to both Dim 0 and Dim 2)
    s3 = Subspace("Mar", basis3, np.zeros(feature_dim), np.array([10, 1]), 2)
    
    return [s1, s2, s3]

def test_drift(mock_subspaces):
    metrics = SociologicalMetrics()
    df_drift = metrics.calculate_drift(mock_subspaces)
    
    print(df_drift)
    
    # Jan: No drift (first)
    assert df_drift.iloc[0]['drift'] == 0.0
    
    # Feb: Identical to Jan -> Drift 0 (Sim 1.0)
    assert df_drift.iloc[1]['drift'] == 0.0
    assert df_drift.iloc[1]['similarity'] == 1.0
    
    # Mar: Orthogonal to Feb -> Drift 1.0 (Sim 0.0)
    # Basis 1 [1,0..] vs Basis 3 [0,1..] -> Dot product 0
    assert df_drift.iloc[2]['similarity'] == 0.0
    assert df_drift.iloc[2]['drift'] == 1.0

def test_projections(mock_subspaces, mock_anchors):
    metrics = SociologicalMetrics()
    df_proj = metrics.calculate_projections(mock_subspaces, mock_anchors)
    
    print(df_proj)
    
    # Jan (Social aligned)
    # Score Social should be 1.0
    assert df_proj.iloc[0]['score_social'] > 0.99
    # Score Functional should be 0.0
    assert df_proj.iloc[0]['score_funcional'] < 0.01
    
    # Mar (Functional aligned)
    # Score Social should be 0.0
    assert df_proj.iloc[2]['score_social'] < 0.01
    # Score Functional should be 1.0
    assert df_proj.iloc[2]['score_funcional'] > 0.99

def test_entropy():
    metrics = SociologicalMetrics()
    
    # High entropy (uniform distribution)
    s_high = Subspace("High", None, None, np.array([10, 10, 10]), 3)
    # Low entropy (concentrated)
    s_low = Subspace("Low", None, None, np.array([100, 1, 1]), 3)
    
    df_ent = metrics.calculate_entropy([s_high, s_low])
    
    assert df_ent.iloc[0]['entropy'] > df_ent.iloc[1]['entropy']

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
