import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class LisbethModel:
    def __init__(self, model_name="PlanTL-GOB-ES/roberta-large-bne", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} to {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_token_indices(self, input_ids, token_str):
        """
        Finds the start and end indices of the sub-tokens corresponding to token_str.
        Returns a list of (start, end) tuples.
        """
        # Tokenize the target word to see how it splits
        target_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        target_len = len(target_ids)
        
        matches = []
        input_ids_list = input_ids.tolist()[0] # Batch size 1 assumption for now
        
        for i in range(len(input_ids_list) - target_len + 1):
            if input_ids_list[i : i + target_len] == target_ids:
                matches.append((i, i + target_len))
                
        return matches

        return final_embeddings
        
    def extract_dual_embedding(self, text, target_word, layers_context=4, layers_static=3):
        """
        Extracts two versions of the embedding:
        1. Contextual: Concat last `layers_context` (Default 4). For Dynamic Subspace.
        2. Static-Compatible: Sum last `layers_static` (Default 3). For Projection onto Static Anchors.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        # 1. Contextual Strategy (Concat last N)
        selected_layers_ctx = hidden_states[-layers_context:]
        concat_embedding = torch.cat(selected_layers_ctx, dim=-1) # (1, seq, dim*4)
        
        # 2. Static Strategy (Sum last M)
        # Note: We take last M *before* the very last layer? Or truly last M?
        # User said "Last 3 layers". Usually last layer is too specialized, but let's stick to last 3.
        selected_layers_static = hidden_states[-layers_static:] 
        # Stack then sum: (3, 1, seq, dim) -> (1, seq, dim)
        stacked_static = torch.stack(selected_layers_static, dim=0)
        sum_embedding = torch.sum(stacked_static, dim=0) 
        
        indices = self.get_token_indices(inputs.input_ids, target_word)
        if not indices:
            return None
            
        results = []
        for start, end in indices:
            # Contextual
            sub_vec_ctx = concat_embedding[0, start:end, :]
            pooled_ctx = torch.mean(sub_vec_ctx, dim=0)
            
            # Static
            sub_vec_static = sum_embedding[0, start:end, :]
            pooled_static = torch.mean(sub_vec_static, dim=0)
            
            results.append({
                "contextual": pooled_ctx.cpu().numpy(),
                "static": pooled_static.cpu().numpy()
            })
            
        return results

    def get_static_embedding(self, word, layers=4):
        """
        Extracts the 'static' embedding for a word in isolation, BUT using the same
        layer strategy as the contextual embeddings (default: Concat last 4 layers).
        This ensures mathematical compatibility (same dimensionality, e.g. 3072)
        for projections.
        """
        inputs = self.tokenizer(word, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack hidden states: (n_layers, batch, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states
        
        # Concatenate last n layers
        selected_layers = hidden_states[-layers:]
        concat_embedding = torch.cat(selected_layers, dim=-1)
        
        # Mean pool if the word was split into multiple subwords
        # shape: (1, seq_len, hidden_dim*layers) -> (hidden_dim*layers,)
        vector = torch.mean(concat_embedding[0], dim=0)
        
        return vector.cpu().numpy()

if __name__ == "__main__":
    # Test
    model = LisbethModel()
    text = "Voy a yapear el dinero mañana."
    word = "yapear"
    emb = model.extract_embedding(text, word)
    if emb:
        print(f"Embedding found for '{word}': {len(emb)} instance(s), Shape: {emb[0].shape}")
    else:
        print(f"Word '{word}' not found in tokens.")
