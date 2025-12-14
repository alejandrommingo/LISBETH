
import os
import shutil
import sys
# Ensure src is in path
sys.path.append(os.getcwd())

import pandas as pd
import json
import logging
from src.nlp.pipeline import PipelineOrchestrator

logging.basicConfig(level=logging.INFO)

def create_dummy_data(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([
        {
            "text": "Ayer salí a Yapear con mis amigos. Fue genial.",
            "published_at": "2023-01-01",
            "newspaper": "TestNews",
            "url": "http://example.com/1"
        },
         {
            "text": "No me gusta el Yapeo, prefiero efectivo. Pero Yape es rapido.",
            "published_at": "2023-01-02",
            "newspaper": "TestNews",
            "url": "http://example.com/2"
        },
        # Case with no keywords
        {
            "text": "Texto irrelevante.",
            "published_at": "2023-01-03",
            "newspaper": "TestNews",
            "url": "http://example.com/3"
        }
    ])
    df.to_csv(path, index=False)
    print(f"Created dummy data at {path}")

def verify_pipeline():
    # Setup
    test_dir = "data/test_phase2"
    input_file = f"{test_dir}/test_input.csv"
    output_file = f"{test_dir}/test_output.csv"
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    create_dummy_data(input_file)
    
    # Run Orchestrator
    # We use 'beto' for both baseline and dapt to avoid large downloads/loading times during test
    print("Running orchestrator...")
    orch = PipelineOrchestrator(baseline_model_name="beto", dapt_model_name="beto")
    orch.run(test_dir, output_file)
    
    # Verify Output
    if not os.path.exists(output_file):
        print("FAIL: Output file not created.")
        return
        
    df = pd.read_csv(output_file)
    print(f"Output has {len(df)} rows.")
    
    # Expectations:
    # Row 1: "Yapear"
    # Row 2: "Yapeo"
    # Row 2: "Yape"
    # Total 3 occurrences.
    
    expected_keywords = ["Yapear", "Yapeo", "Yape"]
    found_keywords = df["keyword_found"].tolist()
    
    # We allow order variances but generally it should be stable
    print("Found keywords:", found_keywords)
    
    assert len(df) == 3, f"Expected 3 occurrences, found {len(df)}"
    
    # Check embeddings
    row0 = df.iloc[0]
    emb_keys = [
        "embedding_baseline_last4_concat", "embedding_baseline_penultimate",
        "embedding_dapt_last4_concat", "embedding_dapt_penultimate"
    ]
    
    for k in emb_keys:
        val = json.loads(row0[k])
        assert isinstance(val, list), f"{k} is not a list"
        assert len(val) > 0, f"{k} is empty"
        # BETO is 768 dims. Last4 concat = 768*4 = 3072. Penultimate = 768.
        if "last4" in k:
            assert len(val) == 3072, f"{k} dimension mismatch: {len(val)}"
        else:
            assert len(val) == 768, f"{k} dimension mismatch: {len(val)}"
            
    print("SUCCESS: All verifications passed.")

if __name__ == "__main__":
    verify_pipeline()
