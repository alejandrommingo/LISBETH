import json
import logging
import hashlib
import numpy as np
import pandas as pd
import torch
import os
from typing import List, Dict, Any, Optional
from scipy.linalg import fractional_matrix_power
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from src.subspace_analysis.schemas import Phase3Config, AnchorDefinition

logger = logging.getLogger(__name__)

class AnchorGenerator:
    """
    Anchor Generator
    Responsibility: Load models, extract anchors, orthogonalize, persist.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_baseline = None
        self.model_baseline = None
        self.tokenizer_dapt = None
        self.model_dapt = None
        self.model_fingerprint_baseline = None
        self.model_fingerprint_dapt = None
        
    def run(self, context_update_callback) -> None:
        """
        Executes the anchor generation process.
        updates the context with run_id and model fingerprints via callback.
        """
        logger.info("AnchorGenerator: Generating Anchors...")
        
        # 1. Load Anchor Definitions
        anchors = self._load_anchor_definitions()
        
        # 2. Load Models
        self._load_models()
        
        # 3. Extract Embeddings
        anchor_embeddings = []
        
        for anchor in anchors:
            emb_data = self._extract_anchor_embeddings(anchor)
            anchor_embeddings.append(emb_data)
            
        df_anchors = pd.DataFrame(anchor_embeddings)
        
        # 4. Generate Run ID
        run_id_str = self._generate_run_id(df_anchors)
        df_anchors['anchors_run_id'] = run_id_str
        df_anchors['model_fingerprint_baseline'] = self.model_fingerprint_baseline
        df_anchors['model_fingerprint_dapt'] = self.model_fingerprint_dapt
        
        # 4. Save CSV
        csv_path = Phase3Config.ARTIFACTS_DIR / "embeddings_anchors.csv"
        df_anchors.to_csv(csv_path, index=False)
        logger.info(f"Saved raw anchor embeddings to {csv_path}")
        
        # 5. Orthogonalize and Persist .npz (for each variant x strategy)
        combinations = [
            (v, s) for v in Phase3Config.VARIANTS for s in Phase3Config.STRATEGIES
        ]
        
        for variant, strategy in combinations:
            self._process_and_save_orthogonal_anchors(df_anchors, variant, strategy, run_id_str)
            
        # Update context
        context_update_callback(
            anchors_run_id=run_id_str,
            baseline_fp=self.model_fingerprint_baseline,
            dapt_fp=self.model_fingerprint_dapt
        )
        logger.info("AnchorGenerator: Success.")

    def _load_anchor_definitions(self) -> List[AnchorDefinition]:
        try:
            with open(Phase3Config.ANCHOR_DEF_JSON, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"FAIL: Could not load anchor definitions: {e}")
            
        anchors = []
        for dim, content in data.items():
            if dim not in Phase3Config.DIMENSIONS:
                 if dim not in ["funcional", "social", "afectiva"]: 
                     logger.warning(f"Found dimension '{dim}' in JSON not in config DIMENSIONS. Ignoring?")
                     continue
            
            if isinstance(content, list):
                items = content
            elif isinstance(content, dict) and "anchors" in content:
                items = content["anchors"]
            else:
                logger.warning(f"Skipping dimension '{dim}': invalid structure")
                continue

            for item in items:
                if "keyword" not in item or "sentence" not in item:
                    raise RuntimeError(f"FAIL: Invalid anchor item format: {item}")
                
                anchors.append(AnchorDefinition(
                    dimension=dim,
                    keyword=item["keyword"],
                    sentence=item["sentence"]
                ))
        
        dims_found = set(a.dimension for a in anchors)
        missing = set(Phase3Config.DIMENSIONS) - dims_found
        if missing:
            raise RuntimeError(f"FAIL: Missing dimensions in anchor file: {missing}")
            
        return anchors

    def _load_models(self):
        baseline_model = Phase3Config.BASELINE_MODEL
        
        self.model_baseline = None
        
        logger.info(f"Attempting to load Baseline: {baseline_model}...")
        try:
            self.tokenizer_baseline = AutoTokenizer.from_pretrained(baseline_model)
            self.model_baseline = AutoModel.from_pretrained(baseline_model).to(self.device).eval()
            self.model_fingerprint_baseline = baseline_model
            logger.info(f"Successfully loaded Baseline: {baseline_model}")
        except Exception as e:
            logger.warning(f"Failed to load Baseline ({baseline_model}): {e}")
            raise RuntimeError(f"FAIL: Could not load Baseline Model: {e}")

        dapt_path = Phase3Config.DAPT_MODEL_PATH
        logger.info(f"Loading DAPT: {dapt_path}")
        if not os.path.exists(dapt_path):
             logger.error(f"FAIL: DAPT model path not found: {dapt_path}")
             raise RuntimeError(f"FAIL: DAPT model path not found: {dapt_path}")
             
        try:
            self.tokenizer_dapt = AutoTokenizer.from_pretrained(dapt_path)
            self.model_dapt = AutoModel.from_pretrained(dapt_path).to(self.device).eval()
            self.model_fingerprint_dapt = f"dapt_local_{dapt_path}"
        except Exception as e:
             raise RuntimeError(f"FAIL: Could not load DAPT model: {e}")

    def _extract_anchor_embeddings(self, anchor: AnchorDefinition) -> Dict[str, Any]:
        sentence = anchor.sentence
        
        emb_base_pen, emb_base_last4 = self._get_embeddings_from_model(
            self.model_baseline, self.tokenizer_baseline, sentence, anchor.keyword
        )
        
        emb_dapt_pen, emb_dapt_last4 = self._get_embeddings_from_model(
            self.model_dapt, self.tokenizer_dapt, sentence, anchor.keyword
        )
        
        return {
            "dimension": anchor.dimension,
            "anchor_keyword": anchor.keyword,
            "anchor_sentence": sentence,
            "embedding_baseline_penultimate": json.dumps(emb_base_pen.tolist()),
            "embedding_baseline_last4_concat": json.dumps(emb_base_last4.tolist()),
            "embedding_dapt_penultimate": json.dumps(emb_dapt_pen.tolist()),
            "embedding_dapt_last4_concat": json.dumps(emb_dapt_last4.tolist())
        }

    def _get_embeddings_from_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str, keyword: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        start_char = text.lower().find(keyword.lower())
        if start_char == -1:
            raise RuntimeError(f"FAIL: Keyword '{keyword}' not found in sentence: '{text}'")
        end_char = start_char + len(keyword)
        
        token_start, token_end = self._char_to_token_span(offset_mapping, start_char, end_char)
        if token_start is None:
             raise RuntimeError(f"FAIL: Could not map keyword '{keyword}' to tokens in '{text}'")
             
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        penultimate = hidden_states[-2][0]
        vec_penultimate = torch.mean(penultimate[token_start:token_end], dim=0).cpu().numpy()
        
        last4 = torch.cat(hidden_states[-4:], dim=-1)[0]
        vec_last4 = torch.mean(last4[token_start:token_end], dim=0).cpu().numpy()
        
        return vec_penultimate, vec_last4

    def _char_to_token_span(self, offset_mapping, char_start, char_end):
        tokens_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if end == 0 and start == 0: continue
            if end > char_start and start < char_end:
                 tokens_indices.append(i)
        if not tokens_indices: return None, None
        return tokens_indices[0], tokens_indices[-1] + 1

    def _generate_run_id(self, df_anchors: pd.DataFrame) -> str:
        content_str = df_anchors[['dimension', 'anchor_keyword', 'anchor_sentence']].to_json()
        payload = f"{content_str}_{Phase3Config.SEED}_{Phase3Config.DIMENSIONS}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _process_and_save_orthogonal_anchors(self, df: pd.DataFrame, variant: str, strategy: str, run_id: str):
        col_name = f"embedding_{variant}_{strategy}"
        
        all_vecs = []
        dim_order = Phase3Config.DIMENSIONS
        
        for dim in dim_order:
            raw_vecs = [np.array(json.loads(x)) for x in df[df['dimension'] == dim][col_name]]
            if not raw_vecs:
                raise RuntimeError(f"FAIL: No anchors for dimension {dim}")
            all_vecs.extend(raw_vecs)
            
        global_mean = np.mean(all_vecs, axis=0) 
        
        dim_vectors = []
        
        for dim in dim_order:
            raw_vecs = [np.array(json.loads(x)) for x in df[df['dimension'] == dim][col_name]]
            # NOTE: We DO NOT center relative to global_mean here because with only 3 dimensions, 
            # centering relative to their own mean enforces sum=0, reducing rank to 2.
            # We use raw directions from origin (BERT anisotropy handles semantic cone).
            
            mean_vec = np.mean(raw_vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm < 1e-9:
                raise RuntimeError(f"FAIL: Zero vector for dimension {dim}")
            unit_vec = mean_vec / norm
            dim_vectors.append(unit_vec)
            
        V = np.stack(dim_vectors, axis=1) # (d, 3)
        
        mtx = V.T @ V 
        mtx_inv_sqrt = fractional_matrix_power(mtx, -0.5)
        A = V @ mtx_inv_sqrt 
        
        check = A.T @ A
        identity = np.eye(3)
        diff = np.linalg.norm(check - identity, ord='fro')
        
        if diff > 1e-3:
             raise RuntimeError(f"FAIL: Orthogonalization failed quality check. Frobenius diff: {diff}")
             
        filename = f"anchors_{variant}_{strategy}.npz"
        path = Phase3Config.ANCHORS_DIR / filename
        np.savez_compressed(
            path,
            A=A,
            dimensions=dim_order,
            anchors_run_id=run_id,
            orthogonalization_method="lowdin",
            variant=variant,
            strategy=strategy
        )
