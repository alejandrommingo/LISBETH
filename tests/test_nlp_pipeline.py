import pytest
import os
import shutil
import pandas as pd
from src.nlp.model import LisbethModel
from src.nlp.dapt import dapt
from src.nlp.extract import extract_embeddings

# Use a small model for testing to avoid heavy downloads/memory usage
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
        f.write("Este es un texto de prueba para yapear.\n" * 10)
    return file_path

@pytest.fixture(scope="module")
def sample_csv(sandbox_dir):
    file_path = os.path.join(sandbox_dir, "yape_test.csv")
    df = pd.DataFrame({
        "plain_text": ["Quiero yapear ahora.", "Usa Plin o Yape.", "Nada relevante."],
        "published_at": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "newspaper": ["Test1", "Test2", "Test3"]
    })
    df.to_csv(file_path, index=False)
    return sandbox_dir

def test_model_loading():
    """Test that LisbethModel loads correctly."""
    model = LisbethModel(model_name=TEST_MODEL)
    assert model.tokenizer is not None
    assert model.model is not None

def test_subword_pooling_logic():
    """Test that subword pooling returns a vector of correct shape."""
    model = LisbethModel(model_name=TEST_MODEL)
    text = "Voy a yapear dinero."
    # "yapear" likely splits in xlm-roberta-base
    embeddings = model.extract_embedding(text, "yapear")
    
    assert embeddings is not None
    assert len(embeddings) == 1
    # Check dimensions: 768 hidden size * 4 layers = 3072
    assert embeddings[0].shape[0] == 768 * 4

def test_dapt_execution(sample_corpus, sandbox_dir):
    """Test DAPT function runs without error on small corpus."""
    output_dir = os.path.join(sandbox_dir, "model_output")
    try:
        dapt(TEST_MODEL, sample_corpus, output_dir, epochs=1)
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
    except Exception as e:
        pytest.fail(f"DAPT execution failed: {e}")

def test_extraction_execution(sample_csv, sandbox_dir):
    """Test extract_embeddings function produces parquet output."""
    output_file = os.path.join(sandbox_dir, "embeddings.parquet")
    keywords = ["Yape", "yapear", "Plin"]
    
    extract_embeddings(sample_csv, output_file, keywords=keywords, model_name=TEST_MODEL, layers=1) # Faster extraction
    
    assert os.path.exists(output_file)
    df = pd.read_parquet(output_file)
    assert len(df) > 0
    assert "keyword" in df.columns
    assert "embedding" in df.columns
    # Check if we captured expected keywords
    found_keywords = df["keyword"].unique()
    assert "yapear" in found_keywords or "Yape" in found_keywords
