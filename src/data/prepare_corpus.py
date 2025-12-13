import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_corpus(input_file: str, output_file: str):
    """
    Reads the specific CSV file, extracts text, cleans it, and saves for DAPT.
    """
    logger.info(f"Reading input file: {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    try:
        # Load CSV
        df = pd.read_csv(input_file)
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Determine text column
        text_col = "text" if "text" in df.columns else "plain_text"
        if text_col not in df.columns:
            logger.error(f"Text column not found. Available: {df.columns}")
            return
            
        logger.info(f"Using column '{text_col}' for text extraction.")
        
        # Filter and clean
        initial_count = len(df)
        df_clean = df.dropna(subset=[text_col])
        texts = df_clean[text_col].astype(str).tolist()
        
        logger.info(f"Dropped {initial_count - len(texts)} null rows.")
        
        valid_lines = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for text in texts:
                # Basic cleaning: remove newlines to flatten docs
                clean_text = text.replace("\n", " ").replace("\r", " ").strip()
                
                # Filter short noise
                if len(clean_text) > 20: 
                    f.write(clean_text + "\n")
                    valid_lines += 1
                    
        logger.info(f"Corpus preparation complete. {valid_lines} documents written to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to process corpus: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Yape_19_23.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/corpus_phase2.txt", help="Output txt file")
    args = parser.parse_args()
    
    prepare_corpus(args.input, args.output)
