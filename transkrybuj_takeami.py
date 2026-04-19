from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
from av.audio.resampler import AudioResampler
from faster_whisper import WhisperModel


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def merge_intervals(intervals: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    container = av.open(str(path))
    audio_stream = container.streams.audio[0]
    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)

    chunks: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32))

    container.close()
    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks)
    return audio / 32768.0


def detect_takes(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    split_pause: float,
    padding: float,
    threshold_db: float | None,
) -> list[tuple[float, float]]:
    window_size = max(1, int(sample_rate * window_seconds))
    windows: list[tuple[float, float]] = []
    levels: list[float] = []

    for index in range(0, len(audio), window_size):
        chunk = audio[index : index + window_size]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        db = 20 * math.log10(max(rms, 1e-6))
        start = index / sample_rate
        end = min((index + len(chunk)) / sample_rate, len(audio) / sample_rate)
        windows.append((start, end))
        levels.append(db)

    if threshold_db is None:
        loud = float(np.percentile(levels, 95))
        quiet = float(np.percentile(levels, 20))
        threshold_db = max(quiet + 8.0, loud - 32.0, -48.0)

    speech = [
        (start, end)
        for (start, end), db in zip(windows, levels)
        if db >= threshold_db
    ]

    padded = [
        (max(0.0, start - padding), min(len(audio) / sample_rate, end + padding))
        for start, end in speech
    ]
    return merge_intervals(padded, split_pause)


def write_srt(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text']}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transkrybuje plik po osobnych take'ach wykrytych z pauz w audio."
    )
    parser.add_argument("plik")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--split-pause",
        type=float,
        default=0.55,
        help="Pauza w sekundach, ktora rozdziela take'i. Domyslnie: 0.55.",
    )
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--window", type=float, default=0.05)
    parser.add_argument("--threshold-db", type=float, default=None)
    parser.add_argument("--word-timestamps", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()

    input_path = Path(args.plik).expanduser().resolve()
    if not input_path.exists():
        print(f"Nie znaleziono pliku: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output).expanduser().resolve() if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem + "_takeami"

    print(f"Czytam audio: {input_path}")
    sample_rate = 16000
    audio = load_audio(input_path, sample_rate=sample_rate)
    if len(audio) == 0:
        print("Nie znaleziono audio w pliku.", file=sys.stderr)
        return 1

    takes = detect_takes(
        audio,
        sample_rate=sample_rate,
        window_seconds=args.window,
        split_pause=args.split_pause,
        padding=args.padding,
        threshold_db=args.threshold_db,
    )
    print(f"Wykryte take'i: {len(takes)}")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    all_segments: list[dict] = []

    for take_index, (take_start, take_end) in enumerate(takes, start=1):
        start_sample = max(0, int(take_start * sample_rate))
        end_sample = min(len(audio), int(take_end * sample_rate))
        take_audio = audio[start_sample:end_sample]
        if len(take_audio) < sample_rate * 0.15:
            continue

        print(f"Take {take_index}/{len(takes)}: {take_start:.2f}-{take_end:.2f}s")
        raw_segments, _ = model.transcribe(
            take_audio,
            language="pl",
            beam_size=5,
            vad_filter=False,
            word_timestamps=args.word_timestamps,
            condition_on_previous_text=False,
        )

        for segment in raw_segments:
            item = {
                "take": take_index,
                "take_start": take_start,
                "take_end": take_end,
                "start": take_start + float(segment.start),
                "end": take_start + float(segment.end),
                "text": segment.text.strip(),
            }
            if args.word_timestamps and segment.words:
                item["words"] = [
                    {
                        "start": take_start + float(word.start),
                        "end": take_start + float(word.end),
                        "word": word.word.strip(),
                        "probability": float(word.probability),
                    }
                    for word in segment.words
                ]
            if item["text"]:
                all_segments.append(item)

    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    json_path = output_dir / f"{stem}.json"
    takes_path = output_dir / f"{stem}_takei.txt"

    txt_path.write_text("\n".join(segment["text"] for segment in all_segments) + "\n", encoding="utf-8")
    write_srt(srt_path, all_segments)
    json_path.write_text(
        json.dumps({"language": "pl", "takes": takes, "segments": all_segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    takes_path.write_text("\n".join(f"{start:.2f}-{end:.2f}" for start, end in takes) + "\n", encoding="utf-8")

    print(f"Zapisano tekst: {txt_path}")
    print(f"Zapisano SRT: {srt_path}")
    print(f"Zapisano JSON: {json_path}")
    print(f"Zapisano take'i: {takes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
