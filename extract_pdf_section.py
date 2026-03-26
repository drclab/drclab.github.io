# PDF Section Extractor

import sys
import re
from pathlib import Path
from typing import Optional

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 is required. Install with: pip install PyPDF2")
    sys.exit(1)

def extract_section(pdf_path: str, section_pattern: str, next_section_pattern: Optional[str] = None):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or '' for page in reader.pages)
    # Find section start
    match = re.search(section_pattern, text)
    if not match:
        print(f"Section start pattern '{section_pattern}' not found.")
        return
    start = match.start()
    if next_section_pattern:
        next_match = re.search(next_section_pattern, text[start+1:])
        end = start + 1 + next_match.start() if next_match else None
    else:
        end = None
    section_text = text[start:end]
    print(section_text.strip())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf_section.py <pdf_path> <section_number> [<next_section_number>]")
        sys.exit(1)
    # Hardcode the correct file path for testing
    pdf_path = "/home/cjduan/drclab.github.io/content/pdf/$$_Causal_Inference_Mkting.pdf"
    section = sys.argv[2] if len(sys.argv) > 2 else "5.11"
    next_section = sys.argv[3] if len(sys.argv) > 3 else "5.12"
    print(f"DEBUG: Using PDF path: {pdf_path}")
    # Patterns like '5.11 ' and '5.12 ' (with space to avoid false matches)
    section_pattern = rf"\b{re.escape(section)}[ .]"
    next_section_pattern = rf"\b{re.escape(next_section)}[ .]" if next_section else None
    extract_section(pdf_path, section_pattern, next_section_pattern)
