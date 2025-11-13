# src/preprocess.py
import re
import unicodedata
from ftfy import fix_text
from unidecode import unidecode

COMMON_FIXES = {
    r'\bsrra\b': 'sierra',
    r'\bcollede\b': 'college',
    r'\bcollede\b': 'college',
    r'\bbuid\b': 'blvd',
    r'\bsute\b': 'suite',
    r'\bpumgiana\b': 'pompano',
}

ZIP_RE = re.compile(r'\b\d{5}(?:-\d{4})?\b')

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = fix_text(s)
    s = unidecode(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'[\u200b\u00A0]', ' ', s)

    for p, r in COMMON_FIXES.items():
        s = re.sub(p, r, s, flags=re.IGNORECASE)

    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

# fallback tracking number extractor
TRACK_PATTERNS = [
    re.compile(r'\b9\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{2}\b'),
    re.compile(r'\b1Z[0-9A-Z]{16}\b'),
    re.compile(r'\b\d{12,22}\b'),
]

def extract_tracking_from_text(text: str):
    text = text.replace(" ", "")
    for pat in TRACK_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group()
    return ""
