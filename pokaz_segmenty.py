from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Pokazuje segmenty z JSON w wybranym zakresie czasu.")
    parser.add_argument("json_path")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=999999.0)
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    for segment in data["segments"]:
        if segment["end"] < args.start or segment["start"] > args.end:
            continue
        take = segment.get("take", "-")
        print(f"{take}: {segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
