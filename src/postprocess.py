# src/postprocess.py
import re
from preprocess import normalize_text

TRACK_PATTERNS = [
    re.compile(r'9\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}'),
    re.compile(r'1Z[0-9A-Z]{16}', re.IGNORECASE),
    re.compile(r'\b\d{12,22}\b'),
    re.compile(r'\bTBA\d+\b', re.IGNORECASE)
]

def extract_tracking_from_text(text):
    for p in TRACK_PATTERNS:
        m = p.search(text)
        if m:
            return m.group().replace(' ','')
    return None

def merge_tokens_to_fields(tokens, tags):
    out = {}
    cur_label = None
    cur_tokens = []
    for t,l in zip(tokens,tags):
        if l == 'O' or l is None:
            if cur_label:
                out.setdefault(cur_label,[]).append(" ".join(cur_tokens))
                cur_label=None
                cur_tokens=[]
            continue
        if l.startswith('B-'):
            if cur_label:
                out.setdefault(cur_label,[]).append(" ".join(cur_tokens))
            cur_label = l[2:].lower()
            cur_tokens = [t]
        elif l.startswith('I-') and cur_label:
            cur_tokens.append(t)
    if cur_label:
        out.setdefault(cur_label,[]).append(" ".join(cur_tokens))
    return out
