
import json
import sys

def extract_code(nb_path):
    print(f"--- Extracting code from {nb_path} ---")
    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                # Filter for plotting code to reduce noise
                if 'plt.' in source or 'sns.' in source or 'heatmap' in source or 'plot' in source:
                    print(f"\n[Cell {i}]")
                    print(source)
                    print("-" * 20)
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            extract_code(p)
