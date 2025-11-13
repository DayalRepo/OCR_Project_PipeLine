# src/convert_to_hf.py

import json
from datasets import Dataset
from pathlib import Path

LABELS = [
    "O",

    "B-NAME","I-NAME",

    "B-ADDRESS","I-ADDRESS",

    "B-CITY","I-CITY",

    "B-STATE","I-STATE",

    "B-ZIP","I-ZIP",

    "B-TRACKING","I-TRACKING",

    "B-COMPANY","I-COMPANY"
]

def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def make_dataset():
    all_data = []
    all_data += load_jsonl("data/labels/bootstrap.jsonl")
    all_data += load_jsonl("data/labels/manual_labels.jsonl")
    all_data += load_jsonl("data/auto_labels.jsonl")

    rows = []
    for e in all_data:
        rows.append({
            "tokens": e["tokens"],
            "ner_tags": [LABELS.index(t) for t in e["tags"]]
        })

    return Dataset.from_list(rows)

if __name__ == "__main__":
    ds = make_dataset()
    ds = ds.train_test_split(test_size=0.1)
    ds.save_to_disk("data/hf_dataset")
    print("Saved dataset to data/hf_dataset")
