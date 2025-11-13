# src/annotate_cli.py
import json
from pathlib import Path
import sys

input_file = "data/auto_labels.jsonl"
output_file = "data/labels/manual_labels.jsonl"

start_at = 0
if len(sys.argv) > 1:
    start_at = int(sys.argv[1])

arr = [json.loads(l) for l in Path(input_file).read_text(encoding='utf8').splitlines()]
out = []

# Load previous annotations if output exists
prev_out = []
if Path(output_file).exists():
    prev_out = [json.loads(l) for l in Path(output_file).read_text().splitlines()]

# Skip already processed entries
if start_at == 0 and prev_out:
    start_at = len(prev_out)
    print(f"Resuming at entry {start_at}")

for i, entry in enumerate(arr[start_at:], start=start_at):
    print("="*40)
    print("Entry", i)
    print("RAW:", entry['raw'])

    for j, (t, tag) in enumerate(zip(entry['tokens'], entry['tags'])):
        print(f"{j:03d}: {t:<15} {tag}")

    s = input("Corrections (e.g. 5:B-NAME,6:I-NAME) or ENTER to accept: ").strip()

    if s:
        try:
            corrections = s.split(',')
            for c in corrections:
                idx, lab = c.split(':')
                idx = int(idx)
                if idx < 0 or idx >= len(entry['tags']):
                    print(f"ERROR: Index {idx} out of range. Skipping correction.")
                    continue
                entry['tags'][idx] = lab
        except:
            print("INVALID FORMAT — Skipping corrections")
            pass

    prev_out.append(entry)

    # Save progress continuously so no data is lost if script stops
    Path(output_file).write_text(
        "\n".join(json.dumps(x) for x in prev_out),
        encoding='utf8'
    )

print("Done!")
