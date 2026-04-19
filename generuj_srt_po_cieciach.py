from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_interval(value: str) -> tuple[float, float]:
    start_text, end_text = value.split("-", 1)
    return float(start_text.replace(",", ".")), float(end_text.replace(",", "."))


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def words_for_interval(segments: list[dict], start: float, end: float) -> list[dict]:
    words: list[dict] = []
    for segment in segments:
        for word in segment.get("words", []):
            word_start = word.get("start")
            word_end = word.get("end")
            if word_start is None or word_end is None:
                continue
            if word_end >= start and word_start <= end:
                words.append(word)
    return words


def add_word_blocks(
    blocks: list[str],
    index: int,
    words: list[dict],
    keep_start: float,
    output_offset: float,
) -> int:
    current: list[dict] = []

    def flush() -> None:
        nonlocal index, current
        if not current:
            return
        text = " ".join(word["word"].strip() for word in current).strip()
        start = output_offset + current[0]["start"] - keep_start
        end = output_offset + current[-1]["end"] - keep_start
        blocks.append(f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}")
        index += 1
        current = []

    for word in words:
        if current:
            gap = word["start"] - current[-1]["end"]
            too_long = len(current) >= 7
            if gap > 0.65 or too_long:
                flush()
        current.append(word)

    flush()
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Tworzy SRT dopasowane do filmu po cięciach.")
    parser.add_argument("json_path")
    parser.add_argument("srt_output")
    parser.add_argument("--keep", action="append", required=True, type=parse_interval)
    args = parser.parse_args()

    segments = json.loads(Path(args.json_path).read_text(encoding="utf-8"))["segments"]
    blocks: list[str] = []
    output_offset = 0.0
    index = 1

    for keep_start, keep_end in args.keep:
        words = words_for_interval(segments, keep_start, keep_end)
        if words:
            index = add_word_blocks(blocks, index, words, keep_start, output_offset)
            output_offset += keep_end - keep_start
            continue

        for segment in segments:
            start = max(segment["start"], keep_start)
            end = min(segment["end"], keep_end)
            if end - start <= 0.15:
                continue

            blocks.append(
                f"{index}\n"
                f"{timestamp(output_offset + start - keep_start)} --> "
                f"{timestamp(output_offset + end - keep_start)}\n"
                f"{segment['text'].strip()}"
            )
            index += 1

        output_offset += keep_end - keep_start

    Path(args.srt_output).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Zapisano: {args.srt_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
