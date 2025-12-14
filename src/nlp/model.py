import torch
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LisbethModel:
    MODEL_REGISTRY = {
        "gov_roberta": "PlanTL-GOB-ES/roberta-large-bne",
        "beto": "dccuchile/bert-base-spanish-wwm-uncased"
    }

    def __init__(self, model_name: str = "PlanTL-GOB-ES/roberta-large-bne", device: str = None, allow_fallback: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Determine strict model name or fallback
        self._load_model(model_name, allow_fallback)

    def _load_model(self, model_input: str, allow_fallback: bool):
        """
        Attempts to load the requested model. Falls back to BETO if gov_roberta fails and allow_fallback is True.
        """
        # Resolve alias if present
        target_model = self.MODEL_REGISTRY.get(model_input, model_input)
        
        try:
            logger.info(f"Loading model: {target_model} to {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(target_model)
            self.model = AutoModel.from_pretrained(target_model).to(self.device)
            self.model.eval()
            self.resolved_model_name = target_model # Public property to know what ran
        except Exception as e:
            logger.error(f"Failed to load {target_model}: {e}")
            if allow_fallback and (target_model == self.MODEL_REGISTRY["gov_roberta"] or "roberta" in target_model):
                logger.warning("Attempting fallback to BETO...")
                fallback_model = self.MODEL_REGISTRY["beto"]
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.model = AutoModel.from_pretrained(fallback_model).to(self.device)
                    self.model.eval()
                    self.resolved_model_name = fallback_model
                    logger.info(f"Fallback successful: {fallback_model}")
                    return
                except Exception as e2:
                    raise RuntimeError(f"Fallback also failed: {e2}")
            else:
                raise e

    def extract_occurrences(self, text: str, keywords: List[str]) -> List[Dict]:
        """
        Extracts embeddings for all occurrences of the given keywords in the text.
        Returns a list of occurrences with metadata and dual embeddings.
        """
        if not text or not keywords:
            return []

        # Tokenize whole text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs.pop("offset_mapping")[0] # (seq_len, 2)
        input_ids = inputs.input_ids[0]
        
        # Lowercase text for matching (if model is cased, we still want to match varied casing if requested)
        # However, it's safer to tokenize the keyword and look for that sequence of IDs.
        # But casing mismatches between query and text are tricky if the tokenizer is case-sensitive (RoBERTa-BNE IS case-sensitive).
        # Strategy: We rely on the caller to provide variants (Yape, yape, YAPE). 
        # OR better: We rely on string finding in the original text, then map to tokens.
        
        occurrences = []
        
        # Run model once
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        # Prepare embeddings strategies
        # 1. Last 4 Concat
        layers_last4 = hidden_states[-4:]
        emb_last4_full = torch.cat(layers_last4, dim=-1)[0] # (seq, dim*4)
        
        # 2. Penultimate
        emb_penultimate_full = hidden_states[-2][0] # (seq, dim)

        text_lower = text.lower()
        
        for kw in keywords:
            # We search for the keyword in the text to get char spans, then map to tokens
            # This is robust to tokenizer differences
            kw_lower = kw.lower()
            start_char = 0
            while True:
                idx = text_lower.find(kw_lower, start_char)
                if idx == -1:
                    break
                
                end_char = idx + len(kw)
                
                # Find token span
                token_start, token_end = self._char_span_to_token_span(offset_mapping, idx, end_char)
                
                if token_start is not None and token_end is not None:
                    # Extract embeddings (Mean Pooling)
                    vec_last4 = torch.mean(emb_last4_full[token_start:token_end], dim=0).cpu().numpy()
                    vec_penultimate = torch.mean(emb_penultimate_full[token_start:token_end], dim=0).cpu().numpy()
                    
                    # Extract sentence context (naive split)
                    # Ideally allow a function for this, but simplistic here:
                    # Find last . before and first . after
                    sent_start = text.rfind('.', 0, idx) + 1
                    sent_end = text.find('.', end_char)
                    if sent_end == -1: sent_end = len(text)
                    context_sentence = text[sent_start:sent_end].strip()

                    occurrences.append({
                        "keyword": kw, # canonical? No, this is the one we searched for.
                        "keyword_found": text[idx:end_char], # Surface form
                        "char_start": idx,
                        "char_end": end_char,
                        "token_start": token_start,
                        "token_end": token_end,
                        "context_sentence": context_sentence,
                        "embedding_last4_concat": vec_last4,
                        "embedding_penultimate": vec_penultimate,
                        "model": self.resolved_model_name
                    })
                
                start_char = end_char
                
        return occurrences

    def _char_span_to_token_span(self, offset_mapping, char_start, char_end):
        """
        Maps character range to token range.
        offset_mapping is tensor (seq_len, 2)
        """
        tokens_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if end == 0 and start == 0: continue # Special tokens often 0,0
            
            # Check overlap
            # Token is [start, end)
            # Target is [char_start, char_end)
            if end > char_start and start < char_end:
                 tokens_indices.append(i)
                 
        if not tokens_indices:
            return None, None
            
        return tokens_indices[0], tokens_indices[-1] + 1

    def get_static_embedding_for_anchor(self, sentence: str, anchor_word: str):
        """
        Extracts embedding for an anchor word in a specific defining context.
        """
        return self.extract_occurrences(sentence, [anchor_word])

# Deprecated/Alias methods for compatibility if needed, but better to cut clean.
