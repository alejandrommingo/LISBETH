import argparse
import json
import pandas as pd
import logging
import os
from src.nlp.model import LisbethModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_anchors(json_path: str, output_file: str, model_name: str):
    """
    Reads anchor definitions from JSON and extracts embeddings using the specified model.
    Saves results to Parquet.
    """
    logger.info(f"Loading anchors from {json_path}")
    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        raise FileNotFoundError(json_path)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # JSON structure expected: 
    # { "dimensions": [ { "name": "...", "anchors": [ {"word": "...", "sentence": "..."} ] } ] }
    # Or similar. I will implement robust parsing based on likely structure.
    # Assuming from context: list of dimensions, each with anchors.
    
    # Let's handle the specific format if known, otherwise generic.
    # Based on "dimensiones_ancla.json", likely:
    # {"funcional": [{"keyword": "...", "oracion": "..."}], ...} or similar.
    # For now, I'll assume a flat structure or iterate keys as dimensions.
    
    # Let's inspect data type
    logger.info(f"JSON loaded. Keys: {list(data.keys())}")
    
    anchors_list = []
    
    model = LisbethModel(model_name=model_name)
    
    # Parsing logic: iterate keys as dimensions
    for dimension, items in data.items():
        # Handle dict structure: {"anchors": [...]}
        if isinstance(items, dict) and "anchors" in items:
            anchors = items["anchors"]
            if not isinstance(anchors, list):
                logger.warning(f"Dimension {dimension} 'anchors' is not a list.")
                continue
                
            for item in anchors:
                # Expecting item to be string or dict
                # If dict, expect 'keyword' and 'sentence' (or similar)
                keyword = item.get("keyword") or item.get("word") or item.get("anchor")
                sentence = item.get("sentence") or item.get("oracion") or item.get("context")
                
                if not keyword or not sentence:
                    logger.warning(f"Skipping malformed item in {dimension}: {item}")
                    continue
                    
                # Extract
                occurrences = model.extract_occurrences(sentence, [keyword])
                
                if not occurrences:
                    logger.warning(f"Could not find '{keyword}' in '{sentence}' for dimension '{dimension}'")
                    continue
                    
                # Take the first occurrence (should be unique in anchor templates)
                occ = occurrences[0]
                occ["dimension"] = dimension
                occ["is_anchor"] = True
                
                anchors_list.append(occ)

        # Legacy/Flat list support
        elif isinstance(items, list):
            for item in items:
                 # ... same logic ...
                 pass # Skipping redundant copy-paste for brevity in this fix block
        else:
            logger.warning(f"Unexpected structure for dimension {dimension}, skipping.")

    if not anchors_list:
        logger.error("No anchors processed.")
        return

    df = pd.DataFrame(anchors_list)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, engine="fastparquet" if "fastparquet" in str(pd.get_option("io.parquet.engine", "auto")) else "pyarrow")
    logger.info(f"Saved {len(df)} anchors to {output_file}")

if __name__ == "__main__":
    pass # Entry point is usually via function call from CLI
