
import pandas as pd
import numpy as np
import hashlib
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_occurrence_id(row):
    # Hash of date + media + context snippet
    s = f"{row['published_at']}_{row['newspaper']}_{row['context_sentence'][:50]}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def standardize_data(csv_path: str, emb_path: str, output_path: str):
    logger.info(">>> Standardizing Data for Phase 3 <<<")
    
    # 1. Load CSV (Metadata Truth)
    logger.info(f"Loading CSV: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    
    # Normalize CSV
    df_csv['published_at'] = pd.to_datetime(df_csv['publish_date'], errors='coerce')
    df_csv = df_csv.dropna(subset=['published_at'])
    df_csv['newspaper'] = df_csv['domain'].str.lower().str.strip()
    df_csv['keyword_canonical'] = df_csv['keyword'].str.lower().str.strip()
    df_csv['url'] = df_csv['url']
    
    # Create join keys
    df_csv['join_date'] = df_csv['published_at'].dt.floor('D')
    
    logger.info(f"CSV Rows Cleaned: {len(df_csv)}")

    # 2. Load Embeddings (Vector Truth)
    logger.info(f"Loading Embeddings: {emb_path}")
    df_emb = pd.read_parquet(emb_path)
    
    # Normalize Parquet
    if 'date' in df_emb.columns:
        df_emb['date'] = pd.to_datetime(df_emb['date'], errors='coerce')
        df_emb['join_date'] = df_emb['date'].dt.floor('D')
        
    if 'media' in df_emb.columns:
        df_emb['newspaper'] = df_emb['media'].str.lower().str.strip()
        
    if 'context' in df_emb.columns:
        df_emb['context_sentence'] = df_emb['context']
        
    if 'original_word' in df_emb.columns:
        df_emb['keyword_matched'] = df_emb['original_word']
    else:
        df_emb['keyword_matched'] = 'unknown'

    # 3. Merge
    # We join on date and newspaper to find potential matches
    logger.info("Merging...")
    
    # Select columns to merge from embeddings
    emb_cols = ['join_date', 'newspaper', 'embedding', 'context_sentence', 'keyword_matched']
    if 'embedding_static' in df_emb.columns:
        emb_cols.append('embedding_static')
        
    merged = pd.merge(
        df_csv,
        df_emb[emb_cols],
        on=['join_date', 'newspaper'],
        how='inner'
    )
    
    logger.info(f"Merged Rows: {len(merged)}")
    
    if len(merged) == 0:
        logger.error("Merge failed (0 rows). Check join keys.")
        return

    # 4. Canonical Transformation
    logger.info("Applying Canonical Schema...")
    
    # IDs and Logic
    merged['occurrence_id'] = merged.apply(generate_occurrence_id, axis=1)
    merged['run_id'] = "phase2_legacy_recovery"
    merged['model_id'] = "spanish_sota" # Assuming source
    merged['model_variant'] = "baseline" # Default to baseline if not specified
    merged['layer_strategy'] = "last4_concat" # Default assumption
    merged['pooling_strategy'] = "subword_mean" 
    
    # Time
    merged['year'] = merged['published_at'].dt.year
    merged['month'] = merged['published_at'].dt.month
    merged['year_month'] = merged['published_at'].dt.strftime('%Y-%m')
    
    # Source
    merged['source_api'] = merged.get('source', 'unknown')
    
    # Embeddings
    # Rename 'embedding' to 'embedding_contextual_last4'
    merged['embedding_contextual_last4'] = merged['embedding']
    merged['embedding_contextual_penultimate'] = None # Missing in current data
    if 'embedding_static' in merged.columns:
         merged['embedding_static_last4'] = merged['embedding_static']
    else:
         merged['embedding_static_last4'] = None
         
    # Dummy Spans (since we don't have them in recover)
    merged['char_start'] = -1
    merged['char_end'] = -1
    merged['token_start'] = -1
    merged['token_end'] = -1
    
    # Select final columns
    final_cols = [
        'occurrence_id', 'run_id', 'model_id', 'model_variant', 
        'layer_strategy', 'pooling_strategy',
        'published_at', 'year', 'month', 'year_month',
        'newspaper', 'source_api', 'url',
        'keyword_canonical', 'keyword_matched',
        'char_start', 'char_end', 'token_start', 'token_end',
        'context_sentence',
        'embedding_contextual_last4', 'embedding_contextual_penultimate',
        'embedding_static_last4'
    ]
    
    # Ensure all exist
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = None
            
    df_final = merged[final_cols]
    
    # Drop duplicates just in case merge exploded
    df_final = df_final.drop_duplicates(subset=['occurrence_id'])
    
    # 5. Save
    logger.info(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path)
    logger.info("Success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--emb_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    standardize_data(args.csv_path, args.emb_path, args.output)
