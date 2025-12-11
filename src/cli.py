import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.nlp.dapt import dapt
from src.nlp.extract import extract_embeddings

def main():
    parser = argparse.ArgumentParser(description="Lisbeth NLP CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # DAPT Command
    dapt_parser = subparsers.add_parser("dapt", help="Run Domain Adaptive Pretraining")
    dapt_parser.add_argument("--data", required=True, help="Path to training corpus (txt)")
    dapt_parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Base model")
    dapt_parser.add_argument("--output", default="models/roberta-adapted", help="Output directory")
    dapt_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    
    # Extract Command
    extract_parser = subparsers.add_parser("extract", help="Extract contextual embeddings")
    extract_parser.add_argument("--data_dir", default="data", help="Directory with CSV files")
    extract_parser.add_argument("--output", default="data/embeddings.parquet", help="Output Parquet file")
    extract_parser.add_argument("--keywords", nargs="+", default=["Yape", "Yapear", "Yapame", "Yapeo", "Plin"], help="List of keywords to extract (default: Yape Yapear Yapame Yapeo Plin)")
    extract_parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Model to use")
    
    args = parser.parse_args()
    
    if args.command == "dapt":
        print(f"Running DAPT on {args.data}...")
        try:
            dapt(args.model, args.data, args.output, args.epochs)
        except Exception as e:
            print(f"DAPT failed: {e}")
            sys.exit(1)
            
    elif args.command == "extract":
        print(f"Extracting embeddings for '{args.keywords}'...")
        try:
            extract_embeddings(args.data_dir, args.output, args.keywords, args.model)
        except Exception as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
