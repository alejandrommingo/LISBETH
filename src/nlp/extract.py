import argparse
import os
import glob
import pandas as pd
import logging
from src.nlp.model import LisbethModel


from src.nlp.pipeline import PipelineOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_embeddings(data_dir: str, output_file: str, keywords: list, model_name: str, baseline_result: bool = False, dapt_model_name: str = None):
    """
    Orchestrates the Phase 2 pipeline.
    
    Args:
        data_dir: Input directory containing Phase 1 CSVs.
        output_file: Final CSV output path.
        keywords: (Ignored now, hardcoded in Expander as per spec, or passed if we want flexibility) 
                 The prompt specified canonical list in code, but good to keep signature compatible.
        model_name: Baseline model.
        baseline_result: (Ignored, new pipeline always produces both).
        dapt_model_name: DAPT model path/name.
    """
    # Fallback default for dapt if not provided (though it should be)
    if not dapt_model_name:
        logger.warning("No DAPT model provided. Using Baseline model for both slots (Debugging/Testing only!)")
        dapt_model_name = model_name
        
    logger.info(f"Starting Phase 2 Pipeline.")
    logger.info(f"Baseline: {model_name}")
    logger.info(f"DAPT: {dapt_model_name}")
    
    orchestrator = PipelineOrchestrator(baseline_model_name=model_name, dapt_model_name=dapt_model_name, keywords=keywords)
    orchestrator.run(data_dir, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for keywords using NLP Pipeline")
    parser.add_argument("--data_dir", required=True, help="Directory containing input CSVs")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--keywords", nargs="+", default=[], help="Keywords (Ignored, internal specs used)")
    parser.add_argument("--model", default="PlanTL-GOB-ES/roberta-large-bne", help="Baseline Model")
    parser.add_argument("--dapt_model", default=None, help="DAPT Model path")
    
    args = parser.parse_args()
    
    extract_embeddings(args.data_dir, args.output, args.keywords, args.model, dapt_model_name=args.dapt_model)

