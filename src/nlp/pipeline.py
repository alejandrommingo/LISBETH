import pandas as pd
import logging
import hashlib
import json
import torch
import numpy as np
from typing import List, Dict, Generator, Optional
from tqdm import tqdm
from src.nlp.model import LisbethModel

logger = logging.getLogger(__name__)

class OccurrenceExpander:
    """
    Agente 1: Transforms document-level data into occurrence-level data.
    """
    CANONICAL_KEYWORD = "yape"
    KEYWORDS = [
        "Yape", "Yapear", "Yapeo", "Yapeando", "Yapeas", "Yapea", "Yapean",
        "Yapeé", "Yapeó", "Yapeado", "Yapeame", "Yapearte", "Yapearon", "Yapearía"
    ]

    def __init__(self):
        pass

    def process(self, df: pd.DataFrame) -> Generator[Dict, None, None]:
        """
        Yields occurrences from the input DataFrame.
        """
        text_col = "text" if "text" in df.columns else "plain_text"
        if text_col not in df.columns:
            logger.warning(f"Skipping dataframe provided, columns: {df.columns} do not contain 'text' or 'plain_text'")
            return

        for idx, row in df.iterrows():
            text = str(row.get(text_col, ""))
            if not text or text.lower() == "nan":
                continue

            text_lower = text.lower()
            
            # 1. Find ALL candidates
            candidates = []
            for kw in self.KEYWORDS:
                kw_lower = kw.lower()
                start_char = 0
                while True:
                    idx_found = text_lower.find(kw_lower, start_char)
                    if idx_found == -1:
                        break
                    
                    end_char = idx_found + len(kw)
                    candidates.append({
                        "start": idx_found,
                        "end": end_char,
                        "keyword_matched": text[idx_found:end_char], # Real form from text
                        "kw_len": len(kw)
                    })
                    start_char = end_char # Move past this one (for this keyword)

            # 2. Filter overlaps (Longest Match First)
            # Sort by length desc, then start position
            candidates.sort(key=lambda x: x["kw_len"], reverse=True)
            
            final_matches = []
            covered_indices = set()
            
            for cand in candidates:
                # Check if any char in this candidate is already covered
                # This is aggressive: if "Yapear" is taken, "Yape" at same start is skipped.
                # If "Yape" is at overlap?
                # Using a set of indices is robust.
                
                is_covered = False
                for i in range(cand["start"], cand["end"]):
                    if i in covered_indices:
                        is_covered = True
                        break
                
                if not is_covered:
                    final_matches.append(cand)
                    for i in range(cand["start"], cand["end"]):
                        covered_indices.add(i)
                        
            # Sort back by position for natural reading order
            final_matches.sort(key=lambda x: x["start"])
            
            # 3. Yield occurrences
            for match in final_matches:
                idx_found = match["start"]
                end_char = match["end"]
                
                # Extract sentence context
                # Search backwards for sentence end (.!?)
                sent_start = max(0, text.rfind('.', 0, idx_found) + 1)
                
                sent_end = text.find('.', end_char)
                if sent_end == -1: 
                    sent_end = len(text)
                else:
                    sent_end += 1 # Include the dot
                
                context_sentence = text[sent_start:sent_end].strip()
                
                if not context_sentence:
                    continue

                yield {
                    "plain_text_doc_id": idx,
                    "published_at": row.get("published_at") or row.get("publish_date"),
                    "newspaper": row.get("newspaper") or row.get("domain"),
                    "source_api": row.get("source_dataset") or row.get("source"),
                    "url": row.get("url"),
                    "keyword_canonical": self.CANONICAL_KEYWORD,
                    "keyword_matched": match["keyword_matched"],
                    "char_start_in_doc": idx_found,
                    "char_end_in_doc": end_char,
                    "context_sentence": context_sentence,
                    "char_start_in_sent": idx_found - sent_start,
                    "char_end_in_sent": end_char - sent_start
                }

class TokenizerAgent:
    """
    Agente 2: Translates character spans to token spans for a specific model tokenizer.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, occurrence: Dict) -> Optional[Dict]:
        """
        Enriches occurrence with token spans.
        """
        text = occurrence["context_sentence"]
        char_start = occurrence["char_start_in_sent"]
        char_end = occurrence["char_end_in_sent"]

        # Tokenize
        # We need return_offsets_mapping=True
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0] # (seq, 2)
        
        # Find token span matching char span
        token_start = None
        token_end = None
        
        found_start = False
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0: continue
            
            # Token [start, end)
            # Match if token overlaps with [char_start, char_end)
            # A token is part of the word if it intersects.
            
            # Intersection logic:
            # max(start, char_start) < min(end, char_end)
            if max(start, char_start) < min(end, char_end):
                if not found_start:
                    token_start = i
                    found_start = True
                token_end = i # Keep updating until we leave the zone
        
        if token_start is not None and token_end is not None:
             # inclusive end index for slicing in python is +1
            occurrence["token_start"] = token_start
            occurrence["token_end"] = token_end + 1
            occurrence["input_ids"] = inputs.input_ids
            occurrence["attention_mask"] = inputs.attention_mask
            return occurrence
        
        return None

class EmbeddingWorker:
    """
    Agente 3: Extracts contextual embeddings.
    """
    def __init__(self, baseline_model: LisbethModel, dapt_model: LisbethModel):
        self.baseline_model = baseline_model
        self.dapt_model = dapt_model

    def process_batch(self, batch_occurrences: List[Dict]) -> List[Dict]:
        """
        Runs both models on the batch and enriches occurrences.
        """
        # We process simple loop here for clarity, but ideally we batch the inference.
        # Given "One row = one occurrence", let's do inference on the single sentence context.
        # Optimization: Group by sentence? 
        # For now, simplistic implementation: predict one by one (or rely on LisbethModel batching if we add it).
        # We will assume process_batch receives a list, we can batch-tensor them.
        
        if not batch_occurrences:
            return []

        # Prepare inputs
        texts = [o["context_sentence"] for o in batch_occurrences]
        
        # We need to run baseline and dapt
        for model_variant, model in [("baseline", self.baseline_model), ("dapt", self.dapt_model)]:
             # Tokenize batch
            inputs = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                outputs = model.model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            
            # Strategies
            # Last 4
            layers_last4 = hidden_states[-4:]
            emb_last4_full = torch.cat(layers_last4, dim=-1) # (batch, seq, dim*4)
            
            # Penultimate
            emb_penultimate_full = hidden_states[-2] # (batch, seq, dim)
            
            # Extract per occurrence
            for i, occ in enumerate(batch_occurrences):
                # We need to realign tokens because batch tokenizer might have different padding/offsets than the Agent2 single tokenizer?
                # Actually Agent2 isn't used for THE inference tokenization here if we re-tokenize.
                # CRITICAL: We should reuse the token alignment or re-calculate it.
                # Re-calculating with the same tokenizer is safer for batching.
                
                # Wait, TokenizerAgent already computed spans on "context_sentence". 
                # If we batch tokenize with padding, the token indices for the actual tokens (not padding) should remain same relative to start?
                # Only if we don't truncate differently.
                # Let's trust offset mapping from batch tokenizer.
                
                # Check mapping again for this batch
                offset_mapping = inputs.offset_mapping[i] if "offset_mapping" in inputs else None # AutoTokenizer with generic call doesn't always return this unless requested
                
                # To be simpler and safer for this specific task:
                # We can run inference individually or ensure we get offsets.
                # Let's request offsets_mapping in this batch call too if possible?
                # Transformers tokenizer batch_encode_plus supports return_offsets_mapping.
                pass
        
        # -- REDO STRATEGY for Reliability --
        # We will iterate row by row for maximum safety regarding span alignment,
        # unless performance is critical (it is, but correctness is higher).
        # Or, we make TokenizerAgent simply prepare the inputs for the model, and we assume models use SAME tokenizer class/vocab?
        # Yes, Baseline DAPT usually share vocab.
        # But if DAPT has new tokens? Unlikely for RoBERTa DAPT usually just tunes weights.
        # Safety: Process one by one or small batches where we track spans carefully.
        
        processed = []
        for occ in batch_occurrences:
            new_occ = occ.copy()
            
            for variant_name, model_obj in [("baseline", self.baseline_model), ("dapt", self.dapt_model)]:
                # Use the helper we'll add to LisbethModel, or do it here.
                # Doing it here ensures Agent3 logic is self-contained.
                
                # Tokenize (single)
                inp = model_obj.tokenizer(occ["context_sentence"], return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True).to(model_obj.device)
                offset_mapping = inp.pop("offset_mapping")[0]
                
                t_start, t_end = self._get_token_span(offset_mapping, occ["char_start_in_sent"], occ["char_end_in_sent"])
                
                # Save spans if baseline (Canonical spans for the row)
                if variant_name == "baseline":
                     new_occ["token_start"] = t_start
                     new_occ["token_end"] = t_end
                     new_occ["model_baseline"] = model_obj.resolved_model_name
                elif variant_name == "dapt":
                     new_occ["model_dapt"] = model_obj.resolved_model_name

                if t_start is None:
                    # matches are subset?
                    continue

                with torch.no_grad():
                    out = model_obj.model(**inp, output_hidden_states=True)
                
                hs = out.hidden_states
                
                # Pooling
                # Last 4
                vec_last4 = torch.cat(hs[-4:], dim=-1)[0, t_start:t_end].mean(dim=0).cpu().numpy().tolist()
                # Penultimate
                vec_penultimate = hs[-2][0, t_start:t_end].mean(dim=0).cpu().numpy().tolist()
                
                new_occ[f"embedding_{variant_name}_last4_concat"] = vec_last4
                new_occ[f"embedding_{variant_name}_penultimate"] = vec_penultimate
            
            processed.append(new_occ)
            
        return processed

    def _get_token_span(self, offsets, char_start, char_end):
        # Helper similar to model.py but standalone
        idxs = []
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0: continue
            if max(s, char_start) < min(e, char_end):
                idxs.append(i)
        if not idxs: return None, None
        return idxs[0], idxs[-1] + 1


class CSVBuilder:
    """
    Agente 5: Ensamblador final.
    """
    def build(self, occurrences: List[Dict], run_id: str) -> pd.DataFrame:
        if not occurrences:
            return pd.DataFrame()
            
        rows = []
        for occ in occurrences:
            # Generate ID
            # Hash of (url, published_at, char_start_in_doc) -> unique occurrence
            unique_str = f"{occ.get('url')}-{occ.get('published_at')}-{occ['char_start_in_doc']}"
            occ_id = hashlib.md5(unique_str.encode()).hexdigest()
            
            # Parse date
            pub_date = str(occ.get("published_at", ""))
            try:
                dt = pd.to_datetime(pub_date, errors='coerce')
                year = dt.year if pd.notnull(dt) else None
                month = dt.month if pd.notnull(dt) else None
                ym = dt.strftime("%Y-%m") if pd.notnull(dt) else None
            except:
                year, month, ym = None, None, None
                
            row = {
                "occurrence_id": occ_id,
                "run_id": run_id,
                "keyword": occ["keyword_canonical"],
                "keyword_found": occ["keyword_matched"],
                "context_sentence": occ["context_sentence"],
                "published_at": occ.get("published_at"),
                "year": year,
                "month": month,
                "year_month": ym,
                "newspaper": occ.get("newspaper"),
                "url": occ.get("url"),
                
                # Metadata & Spans (Contract Compliance)
                "char_start": occ.get("char_start_in_doc"),
                "char_end": occ.get("char_end_in_doc"),
                "token_start": occ.get("token_start"),
                "token_end": occ.get("token_end"),
                "source_api": occ.get("source_api"),
                "model_baseline": occ.get("model_baseline"),
                "model_dapt": occ.get("model_dapt"),
                
                # Embeddings (JSON strings)
                "embedding_baseline_last4_concat": json.dumps(occ.get("embedding_baseline_last4_concat")),
                "embedding_baseline_penultimate": json.dumps(occ.get("embedding_baseline_penultimate")),
                "embedding_dapt_last4_concat": json.dumps(occ.get("embedding_dapt_last4_concat")),
                "embedding_dapt_penultimate": json.dumps(occ.get("embedding_dapt_penultimate")),
            }
            rows.append(row)
            
        return pd.DataFrame(rows)

class PipelineOrchestrator:
    """
    Agente 0: Orquestador.
    """
    def __init__(self, baseline_model_name, dapt_model_name):
        logger.info("Initializing Pipeline Orchestrator...")
        self.baseline_model = LisbethModel(model_name=baseline_model_name)
        self.dapt_model = LisbethModel(model_name=dapt_model_name)
        
        self.expander = OccurrenceExpander()
        self.worker = EmbeddingWorker(self.baseline_model, self.dapt_model)
        self.builder = CSVBuilder()
        
    def run(self, input_dir: str, output_file: str):
        import glob
        import os
        
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        logger.info(f"Found {len(csv_files)} input files.")
        
        all_processed = []
        
        for f in csv_files:
            logger.info(f"Reading {f}...")
            df = pd.read_csv(f)
            
            # 1. Expand
            occurrences_gen = self.expander.process(df)
            
            # Simple batching from generator
            batch = []
            BATCH_SIZE = 32
            
            params_gen = tqdm(occurrences_gen, desc=f"Processing {os.path.basename(f)}")
            for occ in params_gen:
                batch.append(occ)
                if len(batch) >= BATCH_SIZE:
                    # 3. Worker (Includes implicit Agent 2 & 3 logic per my Worker implementation)
                    processed_batch = self.worker.process_batch(batch)
                    all_processed.extend(processed_batch)
                    batch = []
            
            if batch:
                processed_batch = self.worker.process_batch(batch)
                all_processed.extend(processed_batch)
                
        # 5. Build CSV
        logger.info(f"Building final CSV with {len(all_processed)} occurrences...")
        run_id = hashlib.md5(str(pd.Timestamp.now()).encode()).hexdigest()[:8]
        final_df = self.builder.build(all_processed, run_id)
        
        # Invariants Check
        if final_df.empty:
            logger.error("Pipeline produced empty DataFrame. Aborting save.")
            return

        # Check for missing embeddings
        required_emb_cols = [
            "embedding_baseline_last4_concat", "embedding_baseline_penultimate",
            "embedding_dapt_last4_concat", "embedding_dapt_penultimate"
        ]
        if final_df[required_emb_cols].isnull().any().any():
             logger.error("Missing embeddings detected!")
             # In strict mode we might abort, but let's log heavily
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")
