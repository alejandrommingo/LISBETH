import argparse
import sys
import os
import logging

# Ensure src is in path
sys.path.append(os.getcwd())

from src.nlp.dapt import dapt
from src.nlp.extract import extract_embeddings
from src.nlp.build_anchors import build_anchors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Lisbeth NLP CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # --- DAPT Command ---
    dapt_parser = subparsers.add_parser("dapt", help="Run Domain Adaptive Pretraining")
    dapt_parser.add_argument("--data", required=True, help="Path to training corpus (txt)")
    dapt_parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Base model")
    dapt_parser.add_argument("--output", default="models/roberta-adapted", help="Output directory")
    dapt_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    dapt_parser.add_argument("--fp16", action="store_true", help="Use FP16")
    
    # --- Extract Command ---
    extract_parser = subparsers.add_parser("extract", help="Extract contextual embeddings")
    extract_parser.add_argument("--data_dir", required=True, help="Directory with CSV files")
    extract_parser.add_argument("--output", required=True, help="Output Parquet/CSV file")
    extract_parser.add_argument("--keywords", nargs="+", default=["Yape"], help="List of keywords (Ignored in Phase 2 pipeline)")
    extract_parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Baseline Model")
    extract_parser.add_argument("--dapt_model", default=None, help="DAPT Model path (optional)")
    # New flags for Phase 2 comparison
    extract_parser.add_argument("--baseline", action="store_true", help="Flag to indicate this is a baseline run (Legacy)")
    
    # --- Anchors Command ---
    anchors_parser = subparsers.add_parser("anchors", help="Build anchors dataset from JSON")
    anchors_parser.add_argument("--json", required=True, help="Path to dimensions_anchors.json")
    anchors_parser.add_argument("--output", required=True, help="Output Parquet file")
    anchors_parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Model to use")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "dapt":
        try:
            dapt(args.model, args.data, args.output, args.epochs, fp16=args.fp16)
        except Exception as e:
            logger.error(f"DAPT failed: {e}")
            sys.exit(1)
            
    elif args.command == "extract":
        try:
            extract_embeddings(args.data_dir, args.output, args.keywords, args.model, baseline_result=args.baseline, dapt_model_name=args.dapt_model)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            sys.exit(1)

    elif args.command == "anchors":
        try:
            build_anchors(args.json, args.output, args.model)
        except Exception as e:
            logger.error(f"Anchors build failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
