import argparse
import os
import glob
import pandas as pd
import logging
from src.nlp.model import LisbethModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_embeddings(data_dir: str, output_file: str, keywords: list, model_name: str, baseline_result: bool = False):
    """
    Iterates over CSV files in data_dir, extracts embeddings for keywords, and saves to Parquet.
    """
    logger.info(f"Initializing extraction with model: {model_name}")
    try:
        model = LisbethModel(model_name=model_name)
    except Exception as e:
        logger.error(f"Cannot initialize model: {e}")
        return

    # Find CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return

    logger.info(f"Found {len(csv_files)} files to process.")
    
    all_occurrences = []
    
    for file_path in csv_files:
        logger.info(f"Processing {os.path.basename(file_path)}...")
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns
            text_col = "text" if "text" in df.columns else "plain_text"
            if text_col not in df.columns:
                logger.warning(f"Skipping {file_path}: neither 'text' nor 'plain_text' column found. Columns: {df.columns}")
                continue
                
            # Iterate (this could be batched for performance, but keeping simple logic for now)
            for idx, row in df.iterrows():
                text = str(row.get(text_col, ""))
                if not text or text == "nan": 
                    continue
                    
                occurrences = model.extract_occurrences(text, keywords)
                
                # Enrich with metadata
                for occ in occurrences:
                    occ["source_file"] = os.path.basename(file_path)
                    occ["published_at"] = row.get("published_at", None)
                    occ["newspaper"] = row.get("newspaper", None)
                    occ["url"] = row.get("url", None)
                    occ["model_variant"] = "baseline" if baseline_result else "dapt" # Or inferred
                    
                    all_occurrences.append(occ)
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    if not all_occurrences:
        logger.warning("No occurrences found for any keyword.")
        return

    logger.info(f"Saving {len(all_occurrences)} occurrences to {output_file}")
    df_out = pd.DataFrame(all_occurrences)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to Parquet
    df_out.to_parquet(output_file, engine="fastparquet" if "fastparquet" in str(pd.get_option("io.parquet.engine", "auto")) else "pyarrow")
    logger.info("Extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for keywords")
    parser.add_argument("--data_dir", required=True, help="Directory containing input CSVs")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--keywords", nargs="+", required=True, help="Keywords to extract")
    parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Model name or path")
    
    args = parser.parse_args()
    
    extract_embeddings(args.data_dir, args.output, args.keywords, args.model)
