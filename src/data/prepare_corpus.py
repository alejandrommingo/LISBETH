import pandas as pd
import glob
import os

def prepare_corpus(data_dir="data", output_file="data/corpus.txt"):
    csv_files = glob.glob(os.path.join(data_dir, "yape_*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if "plain_text" in df.columns:
                    # Filter out empty texts
                    texts = df["plain_text"].dropna().astype(str).tolist()
                    # Clean: replace newlines with spaces to have one doc per line or keep structure?
                    # For DAPT (MLM), usually line-by-line is fine if lines are sentences.
                    # Or full doc. RoBERTa handles long sequences.
                    # Let's write full doc per line, replacing internal newlines with spaces.
                    for text in texts:
                        clean_text = text.replace("\n", " ").strip()
                        if len(clean_text) > 50: # Minimal length filter
                            f_out.write(clean_text + "\n")
                    print(f"Processed {file}: {len(texts)} documents.")
                else:
                    print(f"Skipping {file}: 'plain_text' column not found.")
            except Exception as e:
                print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    prepare_corpus()
