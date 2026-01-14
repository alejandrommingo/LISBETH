from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import datetime
import pathlib

import os

# --- 0.3 Global Parameters (Fixed) ---
class Phase3Config:
    WINDOW_MONTHS = 3
    STEP_MONTHS = 1
    N_MIN_OCCURRENCES = 20
    LOW_DENSITY_FLAG = 30
    VARIANTS = ["baseline", "dapt"] # DAPT enabled (BETO-adapted)
    STRATEGIES = ["penultimate", "last4_concat"]
    CONDITIONS = ["raw", "corrected"]
    DIMENSIONS = ["funcional", "social", "afectiva"]
    SEED = 42
    B_HORN = 200
    B_BOOT = 200
    CENTERING = True  # Only for subspaces
    
    # Paths (Dynamic via Env)
    BASE_OUTPUT_DIR = pathlib.Path(os.getenv("LISBETH_PHASE3_OUTPUT_DIR", "data/phase3"))
    ARTIFACTS_DIR = BASE_OUTPUT_DIR / "artifacts"
    ANCHORS_DIR = ARTIFACTS_DIR / "anchors"
    SUBSPACES_DIR = ARTIFACTS_DIR / "subspaces"
    MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
    
    OUTPUT_CSV = BASE_OUTPUT_DIR / "phase3_results.csv"
    
    INPUT_CSV = pathlib.Path(os.getenv("LISBETH_PHASE3_INPUT_CSV", "data/phase2/embeddings_occurrences.csv"))
    ANCHOR_DEF_JSON = pathlib.Path(os.getenv("LISBETH_PHASE3_ANCHORS_JSON", "data/dimensiones_ancla.json"))
    
    # Phase 2 Schema Columns
    COL_OCCURRENCE_ID = "occurrence_id"
    COL_PUBLISHED_AT = "published_at"
    COL_URL = "url"
    COL_EMB_BASELINE_PENULTIMATE = "embedding_baseline_penultimate"
    COL_EMB_BASELINE_LAST4 = "embedding_baseline_last4_concat"
    COL_EMB_DAPT_PENULTIMATE = "embedding_dapt_penultimate"
    COL_EMB_DAPT_LAST4 = "embedding_dapt_last4_concat"

@dataclass
class AnchorDefinition:
    dimension: str
    keyword: str
    sentence: str

@dataclass
class Phase3RunContext:
    run_timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    anchors_run_id: Optional[str] = None
    baseline_model_fingerprint: Optional[str] = None
    dapt_model_fingerprint: Optional[str] = None
    valid_windows: List[tuple] = field(default_factory=list) # List of (start_month, end_month)
    
    # Runtime cache could go here if needed
