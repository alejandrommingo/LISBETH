import pytest
import os
import shutil
import pandas as pd
import torch
from src.nlp.model import LisbethModel
from src.nlp.dapt import dapt
from src.nlp.extract import extract_embeddings
from src.nlp.build_anchors import build_anchors

# Use a small model for testing or mock
TEST_MODEL = "xlm-roberta-base"

@pytest.fixture(scope="module")
def sandbox_dir():
    path = "tests/test_sandbox"
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)

@pytest.fixture(scope="module")
def sample_corpus(sandbox_dir):
    file_path = os.path.join(sandbox_dir, "corpus.txt")
    with open(file_path, "w") as f:
        f.write("Este es un texto de prueba para yapear.\n" * 5)
    return file_path

@pytest.fixture(scope="module")
def sample_csv(sandbox_dir):
    file_path = os.path.join(sandbox_dir, "yape_test.csv")
    df = pd.DataFrame({
        "plain_text": ["Quiero yapear ahora mismo.", "Usa Plin o Yape en la tienda.", "Nada relevante aquí."],
        "published_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "newspaper": ["Test1", "Test2", "Test3"]
    })
    df.to_csv(file_path, index=False)
    return sandbox_dir

@pytest.fixture(scope="module")
def sample_anchors_json(sandbox_dir):
    file_path = os.path.join(sandbox_dir, "anchors.json")
    import json
    data = {
        "funcional": [{"keyword": "rapidez", "sentence": "La rapidez es clave en el servicio."}],
        "social": [{"keyword": "amigos", "sentence": "Salgo con mis amigos a comer."}]
    }
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path

def test_model_loading_and_fallback():
    """Test that LisbethModel loads correctly and resolver works."""
    # Test valid model
    model = LisbethModel(model_name=TEST_MODEL)
    assert model.tokenizer is not None
    assert model.model is not None
    
    # Test fallback logic (mocking failure of a "gov_roberta" like request)
    # We can't easily force failure without network block, but we can check the registry
    assert "gov_roberta" in LisbethModel.MODEL_REGISTRY

def test_extract_occurrences():
    """Test dual extraction strategy."""
    model = LisbethModel(model_name=TEST_MODEL)
    text = "Voy a yapear dinero."
    keywords = ["yapear"]
    
    occurrences = model.extract_occurrences(text, keywords)
    
    assert occurrences is not None
    assert len(occurrences) == 1
    occ = occurrences[0]
    
    assert "embedding_last4_concat" in occ
    assert "embedding_penultimate" in occ
    
    # Check dimensions
    # xlm-roberta-base: hidden=768
    # last4_concat = 768 * 4 = 3072
    assert occ["embedding_last4_concat"].shape[0] == 768 * 4
    assert occ["embedding_penultimate"].shape[0] == 768

def test_extract_occurrences_multi_keyword():
    model = LisbethModel(model_name=TEST_MODEL)
    text = "Prefiero Yape pero a veces uso Plin."
    keywords = ["Yape", "Plin"]
    
    occurrences = model.extract_occurrences(text, keywords)
    assert len(occurrences) == 2
    
    found_kws = {o["keyword"] for o in occurrences}
    assert "Yape" in found_kws
    assert "Plin" in found_kws

def test_dapt_execution(sample_corpus, sandbox_dir):
    """Test DAPT function runs (smoke test)."""
    output_dir = os.path.join(sandbox_dir, "model_output")
    try:
        # Run 1 epoch
        dapt(TEST_MODEL, sample_corpus, output_dir, epochs=1)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
    except Exception as e:
        pytest.fail(f"DAPT execution failed: {e}")

def test_extraction_pipeline(sample_csv, sandbox_dir):
    """Test extract_embeddings function."""
    output_file = os.path.join(sandbox_dir, "embeddings.parquet")
    keywords = ["Yape", "yapear", "Plin"]
    
    extract_embeddings(sample_csv, output_file, keywords, TEST_MODEL)
    
    assert os.path.exists(output_file)
    df = pd.read_parquet(output_file)
    assert len(df) > 0
    assert "embedding_last4_concat" in df.columns
    # Check context
    assert df.iloc[0]["context_sentence"] is not None

def test_anchors_pipeline(sample_anchors_json, sandbox_dir):
    """Test build_anchors."""
    output_file = os.path.join(sandbox_dir, "anchors.parquet")
    build_anchors(sample_anchors_json, output_file, TEST_MODEL)
    
    assert os.path.exists(output_file)
    df = pd.read_parquet(output_file)
    assert len(df) == 2 # 2 anchors provided
    assert "is_anchor" in df.columns
