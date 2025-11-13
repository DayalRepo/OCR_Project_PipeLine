# src/auto_label.py
import re, json
from preprocess import normalize_text

# regex patterns
USPS_RE = re.compile(r'9\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}')
UPS_RE = re.compile(r'1Z[0-9A-Z]{16}', re.IGNORECASE)
FEDEX_RE = re.compile(r'\b\d{12,22}\b')
AMZ_RE = re.compile(r'\bTBA\d+\b', re.IGNORECASE)
ZIP_RE = re.compile(r'\b\d{5}(?:-\d{4})?\b')

# small US state abbreviations set
US_STATES = set(["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
                 "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
                 "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
                 "WI","WY","DC"])

def find_tracking(text):
    lst = []
    for r in (USPS_RE, UPS_RE, FEDEX_RE, AMZ_RE):
        for m in r.finditer(text):
            lst.append((m.start(), m.end(), m.group()))
    # sort and dedupe
    lst = sorted(lst, key=lambda x: x[0])
    return lst

def tokenize_simple(text):
    # whitespace & punctuation split but keep tokens used for addresses/numbers
    tokens = re.findall(r"[A-Za-z0-9\-']+|[,\.#]", text)
    return tokens

def auto_label_text(raw_text):
    text = normalize_text(raw_text)
    tokens = tokenize_simple(text)
    tags = ['O'] * len(tokens)

    # map character indices to token indices
    idx = 0
    char_offsets = []
    for i,t in enumerate(tokens):
        # find t in text from idx
        m = re.search(re.escape(t), text[idx:], re.IGNORECASE)
        if not m:
            char_offsets.append((-1,-1))
            continue
        start = idx + m.start()
        end = idx + m.end()
        char_offsets.append((start, end))
        idx = end

    # label tracking
    for s,e,val in find_tracking(text):
        # find tokens overlapping s..e
        for i,(ts,te) in enumerate(char_offsets):
            if ts < 0: continue
            if not (te <= s or ts >= e):
                tags[i] = 'B-TRACKING' if tags[i]=='O' else tags[i]

    # heuristics for address: tag tokens near ZIP or state codes
    for i,(ts,te) in enumerate(char_offsets):
        if ts < 0: continue
        token = tokens[i]
        if ZIP_RE.match(token):
            # backfill previous tokens until start of line or punctuation
            j = i-1
            tags[i] = 'B-ZIP'
            while j>=0 and tokens[j] not in [',', '.']:
                if tags[j] == 'O':
                    tags[j] = 'I-ADDRESS' if tags[j].startswith('I') else 'B-ADDRESS'
                j-=1
            # mark city/state around j
        # state candidate
        if token.upper() in US_STATES:
            tags[i] = 'B-STATE'
            # mark previous token as CITY maybe
            if i-1>=0 and tags[i-1]=='O':
                tags[i-1] = 'B-CITY'

    # naive name detection: tokens before address or after "SHIP" or "TO:" etc.
    for i,tok in enumerate(tokens):
        if tok.lower() in ('ship','to','shipper','recipient','ship:','to:'):
            # look ahead small window for capitalized tokens (names)
            for j in range(i+1, min(len(tokens), i+8)):
                if tokens[j][0].isalpha() and tokens[j][0].isupper():
                    tags[j] = 'B-NAME' if tags[j]=='O' else tags[j]
    # fix BIO formatting: convert sequences to B- and I-
    # (Simplified - more rules can be added)
    for i in range(len(tags)):
        if tags[i].startswith('I-'):
            # if previous is not same type, convert to B-
            if i==0 or tags[i-1][2:] != tags[i][2:]:
                tags[i] = 'B-'+tags[i][2:]

    return {'tokens': tokens, 'tags': tags, 'raw': raw_text}

if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1],'r',encoding='utf-8').read().splitlines()
    out=[]
    for line in txt:
        if not line.strip(): continue
        obj = auto_label_text(line)
        out.append(obj)
    with open('data/auto_labels.jsonl','w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    print("Wrote data/auto_labels.jsonl")
