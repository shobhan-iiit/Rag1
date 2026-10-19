"""
PDF Ingestion and Cleaning for IFAB Laws of the Game 2026/27.
Extracts pages 45 to 150 (Laws 1-17), strips running artifacts, heals hyphens and drop caps.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

LAW_TITLES = {
    1: "The Field of Play",
    2: "The Ball",
    3: "The Players",
    4: "The Players' Equipment",
    5: "The Referee",
    6: "The Other Match Officials",
    7: "The Duration of the Match",
    8: "The Start and Restart of Play",
    9: "The Ball In and Out of Play",
    10: "Determining the Outcome of a Match",
    11: "Offside",
    12: "Fouls and Misconduct",
    13: "Free Kicks",
    14: "The Penalty Kick",
    15: "The Throw-in",
    16: "The Goal Kick",
    17: "The Corner Kick"
}

def clean_page_text(raw_text: str, page_num: int) -> str:
    """
    Cleans raw text extracted from a single PDF page.
    Removes headers, footers, control characters, and heals hyphenation.
    """
    # Remove control characters except standard whitespace
    text = ''.join(ch for ch in raw_text if ord(ch) >= 32 or ch in '\n\t')

    # Remove running headers/footers
    # E.g. "Laws of the Game 2026/27 | Law 11 | Offside" or "L aw s o f t he Ga m e 20 2 6/27 ..."
    text = re.sub(r'(?i)l\s*aw\s*s\s*o\s*f\s*t\s*he\s*ga\s*m\s*e\s*20\s*2\s*6\s*/\s*27.*?\n', '\n', text)
    text = re.sub(r'(?i)laws\s+of\s+the\s+game\s+2026/27.*?\n', '\n', text)

    # Remove standalone page numbers on single lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Heal hyphenation across linebreaks (e.g. "inter-\nference" -> "interference")
    text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\g<1>\g<2>', text)

    # Heal drop-caps common in InDesign exports:
    # E.g. "\nI\n f " -> "\nIf ", "\nT\nhis" -> "\nThis", "\nA\nplayer" -> "\nA player"
    text = re.sub(r'\n([A-Z])\s*\n\s*([a-z]+)', r'\n\g<1>\g<2>', text)
    # E.g. "I f " -> "If ", "T his " -> "This ", "A n " -> "An "
    text = re.sub(r'\bI\s+f\b', 'If', text)
    text = re.sub(r'\bT\s+his\b', 'This', text)
    text = re.sub(r'\bA\s+n\b', 'An', text)
    text = re.sub(r'\bT\s+he\b', 'The', text)
    text = re.sub(r'\bW\s+hen\b', 'When', text)
    text = re.sub(r'\bA\s+fter\b', 'After', text)

    # Normalize typographic quotes and dashes
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('—', '-')

    # Replace multiple spaces with a single space, maintaining paragraphs
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    cleaned = '\n'.join(line for line in lines if line)
    return cleaned

def extract_laws_corpus(pdf_path: Path, start_page: int = 45, end_page: int = 150) -> List[Dict[str, Any]]:
    """
    Extracts text from PDF across specified pages and tags pages with Law number and title.
    Returns list of page records.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required. Run 'pip install pymupdf'.")

    doc = fitz.open(str(pdf_path))
    pages_data = []
    current_law = 1

    for page_idx in range(start_page - 1, min(end_page, len(doc))):
        page_num = page_idx + 1
        page = doc.load_page(page_idx)
        raw_text = page.get_text("text")

        # Detect Law headings (e.g. "Law 11 Offside" or "11 Offside")
        law_match = re.search(r'\bLaw\s+([1-9]|1[0-7])\b', raw_text, re.IGNORECASE)
        if law_match:
            detected_law = int(law_match.group(1))
            if detected_law >= current_law:
                current_law = detected_law

        cleaned_text = clean_page_text(raw_text, page_num)
        if len(cleaned_text.strip()) > 30:  # Ignore blank or near-blank separator pages
            pages_data.append({
                "page": page_num,
                "law_number": current_law,
                "law_title": LAW_TITLES.get(current_law, f"Law {current_law}"),
                "text": cleaned_text
            })

    doc.close()
    return pages_data
