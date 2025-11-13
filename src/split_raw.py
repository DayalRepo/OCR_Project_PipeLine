# src/split_raw.py
import re
from pathlib import Path

def split_raw(infile="data/raw.txt", outdir="data/single"):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    raw = open(infile, "r", encoding="utf8").read()

    parts = re.split(r"Raw Text:|\n\[", raw)
    cleaned = []

    for p in parts:
        p = p.strip().strip("[]").strip(",").strip()
        if len(p) > 5:
            cleaned.append(p)

    for i, p in enumerate(cleaned):
        with open(f"{outdir}/{i}.txt", "w", encoding="utf8") as f:
            f.write(p)

    print(f"Wrote {len(cleaned)} files to {outdir}")

if __name__ == "__main__":
    split_raw()
