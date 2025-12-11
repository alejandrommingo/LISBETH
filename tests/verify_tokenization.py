from src.nlp.model import LisbethModel
import sys

def verify_tokenization():
    try:
        model = LisbethModel()
        words = ["Yape", "Yapear", "Plin", "yapear"]
        text = "Voy a yapear el dinero mañana por Yape o Plin."
        
        print(f"Text: '{text}'")
        tokens = model.tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        ids = model.tokenizer.encode(text, add_special_tokens=False)
        print(f"Token IDs: {ids}")
        
        for word in words:
            print(f"\n--- Verifying '{word}' ---")
            emb = model.extract_embedding(text, word)
            if emb:
                print(f"SUCCESS: Embedding extracted for '{word}'. Shape: {emb[0].shape}")
                # Check mean pooling if it split
                word_tokens = model.tokenizer.tokenize(word)
                print(f"Subwords: {word_tokens}")
                if len(word_tokens) > 1:
                    print("Note: Word was split, Mean Pooling was applied.")
            else:
                print(f"FAILURE: Could not extract embedding for '{word}'")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_tokenization()
