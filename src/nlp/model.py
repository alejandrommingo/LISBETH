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

    def extract_embedding(self, text, target_word, layers=4):
        """
        Extracts the embedding for the target_word in the text.
        Implements Subword Mean Pooling if the word splits into multiple tokens.
        Uses concatenation of the last `layers` hidden states.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Stack hidden states: (n_layers, batch, seq_len, hidden_dim)
        # We want the last n layers
        hidden_states = outputs.hidden_states
        # Concatenate last n layers along the feature dimension
        # hidden_states tuple length is n_layers + 1 (embeddings)
        selected_layers = hidden_states[-layers:] 
        # Shape: batch, seq_len, hidden_dim * layers
        concat_embedding = torch.cat(selected_layers, dim=-1)
        
        # Find token indices
        indices = self.get_token_indices(inputs.input_ids, target_word)
        
        if not indices:
            return None
            
        final_embeddings = []
        for start, end in indices:
            # Extract relevant vectors (shape: end-start, hidden_dim*layers)
            sub_vectors = concat_embedding[0, start:end, :]
            # Mean Pooling over subwords
            pooled_vector = torch.mean(sub_vectors, dim=0)
            final_embeddings.append(pooled_vector.cpu().numpy())
            
        return final_embeddings

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
