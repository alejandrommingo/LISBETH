#!/usr/bin/env python3
import sys
import logging
import os

# Add project root to sys.path so src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.phase3.pipeline import Phase3Orchestrator

def main():
    # Ensure logging goes to scripts dir or root? 
    # Let's log to root for visibility or same dir as data. 
    # User said "scripts" for script, implying independent execution.
    # Log to stdout mainly.
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
            # logging.FileHandler("../phase3_run.log") # Optional
        ]
    )
    
    logger = logging.getLogger("MAIN")
    logger.info("Initializing Phase 3 CLI (Scripts)...")
    
    orchestrator = Phase3Orchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
