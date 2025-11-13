# src/evaluate.py
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.preprocess import normalize_text, extract_tracking_from_text

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label

def clean_token(tok: str) -> str:
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("Ċ", "")
    return tok.strip()

def collapse_tokens(enc, preds):
    words = []
    labels = []

    ids = enc.word_ids()
    toks = enc.tokens()

    last = None
    buf = ""
    last_label = None

    for i, wid in enumerate(ids):
        if wid is None:
            continue

        tok = clean_token(toks[i])

        if wid != last:
            if buf:
                words.append(buf)
                labels.append(last_label)
            buf = tok
            last_label = preds[i]
            last = wid
        else:
            buf += tok

    if buf:
        words.append(buf)
        labels.append(last_label)

    return words, labels

def decode_bio(words, labels, id2label):
    out = {
        "NAME": [],
        "ADDRESS": [],
        "CITY": [],
        "STATE": [],
        "ZIP": [],
        "TRACKING": [],
        "COMPANY": []
    }

    cur = None
    buf = []

    for w, lid in zip(words, labels):
        tag = id2label[lid]

        if tag == "O":
            if cur:
                out[cur].append(" ".join(buf))
                buf = []
                cur = None
            continue

        p, typ = tag.split("-")

        if p == "B":
            if cur:
                out[cur].append(" ".join(buf))
            cur = typ
            buf = [w]

        elif p == "I" and cur == typ:
            buf.append(w)
        else:
            if cur:
                out[cur].append(" ".join(buf))
            cur = typ
            buf = [w]

    if cur:
        out[cur].append(" ".join(buf))

    return out

def choose_best(items):
    if not items:
        return ""
    items = [i.strip(" ,.-") for i in items if len(i.strip()) > 0]
    if not items:
        return ""
    return max(items, key=len)

def evaluate_file(path, tokenizer, model, id2label):
    raw = open(path, "r", encoding="utf8").read()
    text = normalize_text(raw)

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits.argmax(-1).squeeze().tolist()

    words, labels = collapse_tokens(enc, logits)
    ents = decode_bio(words, labels, id2label)

    result = {
        "name": choose_best(ents["NAME"]),
        "address": choose_best(ents["ADDRESS"]),
        "city": choose_best(ents["CITY"]),
        "state": choose_best(ents["STATE"]),
        "zip": choose_best(ents["ZIP"]),
        "tracking": choose_best(ents["TRACKING"]),
        "company": choose_best(ents["COMPANY"]),
    }

    # fallback tracking extraction
    if not result["tracking"]:
        result["tracking"] = extract_tracking_from_text(text)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", default="models/roberta_ner")
    args = parser.parse_args()

    tokenizer, model, id2label = load_model(args.model)
    model.eval()

    p = Path(args.path)

    if p.is_file():
        res = evaluate_file(str(p), tokenizer, model, id2label)
        print(json.dumps(res, indent=2))
        return

    files = sorted(p.glob("*.txt"))
    for f in files:
        out = evaluate_file(str(f), tokenizer, model, id2label)
        print("\n----", f.name, "----")
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
