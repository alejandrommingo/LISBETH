import zipfile
import xml.etree.ElementTree as ET
import argparse
import os

def extract_text_from_docx(docx_path, output_path):
    try:
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        paragraphs = []
        
        for p in tree.findall('.//w:p', ns):
            texts = [node.text for node in p.findall('.//w:t', ns) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
            else:
                paragraphs.append('') # Preserve empty lines/paragraphs
                
        full_text = '\n\n'.join(paragraphs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        print(f"Successfully extracted text to {output_path}")
        
    except Exception as e:
        print(f"Error extracting docx: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    extract_text_from_docx(args.file, args.output)
