from __future__ import annotations

import argparse
import math
from pathlib import Path

import av
import numpy as np
from av.audio.resampler import AudioResampler


def parse_interval(value: str) -> tuple[float, float]:
    start_text, end_text = value.split("-", 1)
    return float(start_text.replace(",", ".")), float(end_text.replace(",", "."))


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


def intersect_intervals(
    speech: list[tuple[float, float]],
    allowed: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for speech_start, speech_end in speech:
        for allowed_start, allowed_end in allowed:
            start = max(speech_start, allowed_start)
            end = min(speech_end, allowed_end)
            if end > start:
                result.append((start, end))
    return sorted(result)


def detect_speech(
    input_path: Path,
    window_seconds: float,
    max_silence: float,
    padding: float,
    threshold_db: float | None,
) -> list[tuple[float, float]]:
    container = av.open(str(input_path))
    audio_stream = container.streams.audio[0]
    resampler = AudioResampler(format="s16", layout="mono", rate=16000)

    samples: list[np.ndarray] = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            array = resampled.to_ndarray().reshape(-1).astype(np.float32)
            samples.append(array)

    container.close()
    if not samples:
        return []

    audio = np.concatenate(samples)
    sample_rate = 16000
    window_size = max(1, int(sample_rate * window_seconds))
    windows: list[tuple[float, float]] = []
    levels: list[float] = []

    for index in range(0, len(audio), window_size):
        chunk = audio[index : index + window_size]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        db = 20 * math.log10(max(rms, 1.0) / 32768.0)
        start = index / sample_rate
        end = min((index + len(chunk)) / sample_rate, len(audio) / sample_rate)
        windows.append((start, end))
        levels.append(db)

    if threshold_db is None:
        loud = float(np.percentile(levels, 95))
        quiet = float(np.percentile(levels, 20))
        threshold_db = max(quiet + 8.0, loud - 32.0, -48.0)

    raw_intervals = [
        (start, end)
        for (start, end), db in zip(windows, levels)
        if db >= threshold_db
    ]

    padded = [
        (max(0.0, start - padding), min(len(audio) / sample_rate, end + padding))
        for start, end in raw_intervals
    ]
    return merge_intervals(padded, max_silence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wykrywa dynamiczne przedziały mowy do montażu.")
    parser.add_argument("input")
    parser.add_argument("--allow", action="append", required=True, type=parse_interval)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=float, default=0.08)
    parser.add_argument("--max-silence", type=float, default=0.50)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--threshold-db", type=float, default=None)
    args = parser.parse_args()

    speech = detect_speech(
        Path(args.input),
        window_seconds=args.window,
        max_silence=args.max_silence,
        padding=args.padding,
        threshold_db=args.threshold_db,
    )
    dynamic = merge_intervals(intersect_intervals(speech, sorted(args.allow)), args.max_silence)

    output_path = Path(args.output)
    output_path.write_text(
        "\n".join(f"{start:.2f}-{end:.2f}" for start, end in dynamic) + "\n",
        encoding="utf-8",
    )

    total = sum(end - start for start, end in dynamic)
    print(f"Zapisano: {output_path}")
    print(f"Przedzialy: {len(dynamic)}")
    print(f"Laczny czas: {total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
