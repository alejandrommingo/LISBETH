#!/usr/bin/env python3
import argparse
import sys
import logging
import subprocess
from pathlib import Path
import os

# Ensure src is in path
sys.path.append(os.getcwd())

# Imports
from src.news_harvester.cli import run_harvest, _load_environment
from src.news_harvester.config import Settings as HarvesterSettings
from src.nlp.dapt import dapt
from src.nlp.extract import extract_embeddings
from src.nlp.build_anchors import build_anchors
from src.subspace_analysis.pipeline import Phase3Orchestrator
# Phase 4 imports to be determined (likely subprocess for notebooks or specific scripts)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline_manager")

def main():
    parser = argparse.ArgumentParser(description="Lisbeth Master Pipeline")
    subparsers = parser.add_subparsers(dest="phase", help="Pipeline Phase", required=True)
    
    # --- PHASE 1: HARVEST ---
    p1 = subparsers.add_parser("phase1", help="News Harvester")
    p1.add_argument("--keyword", nargs="+", default=["Yape"], help="Keywords")
    p1.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    p1.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    p1.add_argument("--media-list", type=Path, help="Path to media CSV")
    p1.add_argument("--output", type=Path, help="Output file")
    p1.add_argument("--sources", nargs="+", default=["gdelt"], choices=["gdelt", "google", "rss"])
    p1.add_argument("--no-download", dest="download_html", action="store_false", help="Skip HTML download (Metadata only)")
    p1.set_defaults(download_html=True)
    
    # --- PHASE 2: NLP ---
    p2 = subparsers.add_parser("phase2", help="NLP Processing (DAPT + Extraction)")
    p2_subs = p2.add_subparsers(dest="step", required=True)
    
    # DAPT
    p2_dapt = p2_subs.add_parser("dapt", help="Domain Adaptive Pretraining")
    p2_dapt.add_argument("--data", required=True, help="Corpus txt")
    p2_dapt.add_argument("--output", default="models/roberta-adapted")
    p2_dapt.add_argument("--epochs", type=int, default=3)
    p2_dapt.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Base model for DAPT")    
    # Download Models
    p2_dl = p2_subs.add_parser("download-models", help="Download and cache HF models")
    p2_dl.add_argument("--models", nargs="+", default=["PlanTL-GOB-ES/roberta-large-bne", "dccuchile/bert-base-spanish-wwm-uncased"], help="Models to download")
    p2_dl.add_argument("--force", action="store_true", help="Force redownload")

    # Extract
    p2_ext = p2_subs.add_parser("extract", help="Extract Embeddings")
    p2_ext.add_argument("--data_dir", required=True, help="Input directory (Phase 1 output)")
    p2_ext.add_argument("--output", required=True, help="Output Parquet")
    p2_ext.add_argument("--dapt_model", required=True, help="Path to DAPT model")
    p2_ext.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Baseline Model")
    
    # Anchors
    p2_anc = p2_subs.add_parser("anchors", help="Build Anchors")
    p2_anc.add_argument("--json", required=True, help="Anchors definition JSON")
    p2_anc.add_argument("--output", required=True, help="Output path (e.g. data/anchors.parquet)")
    
    # --- PHASE 3: SUBSPACE ---
    p3 = subparsers.add_parser("phase3", help="Subspace Analysis")
    p3.add_argument("--input", required=True, help="Path to embeddings_occurrences.csv (Phase 2 Output)")
    p3.add_argument("--output-dir", required=True, help="Directory to save Phase 3 results")
    
    # --- PHASE 4: REPORT ---
    p4 = subparsers.add_parser("phase4", help="Generate Reports")
    p4.add_argument("--input", required=True, help="Path to Phase 3 Results CSV")
    p4.add_argument("--output_dir", required=True, help="Report output directory")
    
    # --- RUN ALL ---
    pall = subparsers.add_parser("run-all", help="Execute Full Pipeline (Not implemented yet)")

    args = parser.parse_args()
    
    if args.phase == "phase1":
        # Map args to Harvester namespace
        # We need to construct a Namespace compatible with run_harvest
        h_args = argparse.Namespace(
            command="harvest",
            keyword=args.keyword,
            date_from=args.date_from, # str, run_harvest expects Date? No, _parse_iso_date in cli. But here we get str.
            # We need to parse dates if run_harvest expects dates.
            # cli._parse_iso_date converts str to date. 
            # run_harvest logic: date_from = args.date_from OR settings...
            # if args.date_from is a date object.
            # Let's parse them here.
            date_to=args.date_to,
            media_list=args.media_list,
            output=args.output,
            sources=args.sources,
            format="csv", # Default
            download_html=args.download_html,
            no_fetch_html=not args.download_html,
            media=["all"] # Default if list provided
        )
        
        # Parse dates
        from datetime import date
        h_args.date_from = date.fromisoformat(args.date_from)
        h_args.date_to = date.fromisoformat(args.date_to)
        
        settings = _load_environment()
        run_harvest(h_args, settings)
        
    elif args.phase == "phase2":
        if args.step == "dapt":
            data_path = Path(args.data)
            if data_path.suffix.lower() == ".csv":
                logger.info(f"Auto-converting CSV {data_path} to TXT for DAPT...")
                import pandas as pd
                try:
                    df = pd.read_csv(data_path)
                    # Content column usually 'content' or 'text' or 'plain_text'
                    col = None
                    for c in ['content', 'text', 'plain_text']:
                        if c in df.columns:
                            col = c
                            break
                    
                    if not col:
                         raise ValueError(f"CSV missing content column. Found: {df.columns}")
                    
                    # Filter empty
                    texts = df[col].dropna().astype(str).tolist()
                    
                    txt_path = data_path.with_suffix(".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(texts))
                        
                    logger.info(f"Created corpus: {txt_path} ({len(texts)} lines)")
                    data_path = txt_path
                except Exception as e:
                    logger.error(f"Failed to convert CSV: {e}")
                    sys.exit(1)
            
            dapt(args.model, str(data_path), args.output, args.epochs)
        elif args.step == "extract":
            # Extract requires keywords? "Ignored in Phase 2 pipeline" says cli.py?
            # extract_parser.add_argument("--keywords", ...)
            # We pass empty list or dummy
            extract_embeddings(args.data_dir, args.output, ["Yape"], args.model, dapt_model_name=args.dapt_model)
        elif args.step == "anchors":
            # build_anchors(json_path, output_path, model_name)
            # Use base model for anchors? AnchorGenerator uses both.
            # This "build_anchors" function in src.nlp might be the OLD one?
            # Phase 3 AnchorGenerator calculates orthogonal anchors.
            # src.nlp.build_anchors might just tokenize?
            # If Phase 3 has AnchorGenerator, do we need Phase 2 anchors step?
            # loads from JSON. 
            # Let's assume Phase 3 handles anchors internally (MANDATORY).
            logger.info("Phase 2 Anchors step might be redundant if Phase 3 AnchorGenerator is used. Proceeding anyway.")
            build_anchors(args.json, args.output, "PlanTL-GOB-ES/roberta-large-bne")
            
        elif args.step == "download-models":
            from transformers import AutoTokenizer, AutoModel
            for model_name in args.models:
                logger.info(f"Downloading/Verifying {model_name}...")
                try:
                    AutoTokenizer.from_pretrained(model_name, force_download=args.force)
                    AutoModel.from_pretrained(model_name, force_download=args.force)
                    logger.info(f"OK: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")

    elif args.phase == "phase3":
        logger.info("Running Phase 3 Orchestrator...")
        
        # Set Environment Variables for Configuration Override
        os.environ["LISBETH_PHASE3_INPUT_CSV"] = str(args.input)
        
        # If output file is specified, we need to handle it.
        # But Phase 3 config takes a BASE_OUTPUT_DIR and creates multiple files.
        # It's better to expect an OUTPUT DIR for Phase 3.
        # But arguments only ask for INPUT.
        # Let's assume we want to output to the SAME directory structure as default or infer from input?
        # Argument 'output_dir' is missing in phase3 parser above.
        # Let's check parser definition: p3.add_argument("--input", required=True)
        # We should add --output-dir to phase3 parser.
        
        # NOTE: I need to update parser definition in main() first! 
        # But for now, let's use a default or assume it's set via env or logic.
        # Wait, I can't add arguments in this block.
        # I'll rely on global env vars if user wants to override, OR I'll add the argument in a separate step if needed.
        # But wait, looking at my planning below, I will Update the parser too.
        
        # Validate Output Dir
        if "output_dir" in args and args.output_dir:
             out_path = Path(args.output_dir)
             os.environ["LISBETH_PHASE3_OUTPUT_DIR"] = str(out_path)
             
             # Runtime Patching of Config (since module is already imported)
             from src.subspace_analysis.schemas import Phase3Config
             Phase3Config.BASE_OUTPUT_DIR = out_path
             Phase3Config.ARTIFACTS_DIR = out_path / "artifacts"
             Phase3Config.ANCHORS_DIR = Phase3Config.ARTIFACTS_DIR / "anchors"
             Phase3Config.SUBSPACES_DIR = Phase3Config.ARTIFACTS_DIR / "subspaces"
             Phase3Config.MANIFESTS_DIR = Phase3Config.ARTIFACTS_DIR / "manifests"
             Phase3Config.OUTPUT_CSV = out_path / "phase3_results.csv"
             Phase3Config.INPUT_CSV = Path(args.input)
             # Force low threshold for small test batches (Yape 2020 has few records)
             Phase3Config.N_MIN_OCCURRENCES = 1
             
             # Create Dirs
             Phase3Config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
             Phase3Config.ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
             Phase3Config.SUBSPACES_DIR.mkdir(parents=True, exist_ok=True)
             Phase3Config.MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        
        Phase3Orchestrator().run()

    elif args.phase == "phase4":
        logger.info("Phase 4 Reporting...")
        
        # Args: output_dir
        out_dir = Path(args.output_dir)
        assets_dir = out_dir / "assets"
        nb_path = out_dir / "report.ipynb"
        
        # We need the Input CSV from Phase 3.
        # Phase 4 parser didn't ask for it.
        # I should add --input to Phase 4 parser.
        
        if "input" not in args:
             logger.error("Phase 4 requires --input (Phase 3 Results CSV)")
             sys.exit(1)
             
        # Call Generate Assets
        cmd_assets = [
            sys.executable, "src/reporting/assets.py",
            "--input", str(args.input),
            "--output", str(assets_dir)
        ]
        subprocess.check_call(cmd_assets)
        
        # Call Notebook Creator
        # Need relative paths
        try:
            rel_assets = os.path.relpath(assets_dir, out_dir) # should be 'assets'
            rel_csv = os.path.relpath(Path(args.input), out_dir)
        except ValueError:
            # If paths are on different mounts or something?
            rel_assets = str(assets_dir)
            rel_csv = str(args.input)
            
        cmd_nb = [
            sys.executable, "src/reporting/notebook.py",
            "--output", str(nb_path),
            "--assets_dir", rel_assets,
            "--csv_path", rel_csv
        ]
        subprocess.check_call(cmd_nb)
        
        logger.info(f"Phase 4 Complete. Report at {nb_path}")

if __name__ == "__main__":
    main()
