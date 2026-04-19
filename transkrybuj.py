from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def format_timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def write_txt(path: Path, segments: list[dict]) -> None:
    text = " ".join(segment["text"].strip() for segment in segments).strip()
    path.write_text(text + "\n", encoding="utf-8")


def write_srt(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_json(path: Path, segments: list[dict]) -> None:
    path.write_text(
        json.dumps({"language": "pl", "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zamienia polski komentarz z audio/wideo na tekst i napisy SRT."
    )
    parser.add_argument("plik", help="Ścieżka do pliku audio albo wideo, np. nagranie.mp4")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Folder wynikowy. Domyślnie ten sam folder co plik wejściowy.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Model Whisper: tiny, base, small, medium, large-v3. Domyślnie: medium.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="cpu albo cuda, jeśli masz kartę NVIDIA i działające CUDA. Domyślnie: cpu.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Typ obliczeń, np. int8 dla CPU albo float16 dla CUDA. Domyślnie: int8.",
    )
    parser.add_argument(
        "--bez-srt",
        action="store_true",
        help="Nie zapisuj pliku SRT z timestampami.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Zapisz dodatkowy plik JSON z segmentami i timestampami.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Dodaj timestampy slow do pliku JSON. Przydatne do dokladniejszego montazu.",
    )
    parser.add_argument(
        "--bez-vad",
        action="store_true",
        help="Wylacz filtrowanie ciszy Whispera. Przydatne, gdy model skleja powtarzane take'i.",
    )
    parser.add_argument(
        "--bez-kontekstu",
        action="store_true",
        help="Nie przenos poprzedniego tekstu jako kontekstu dla kolejnych fragmentow.",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()

    if WhisperModel is None:
        print(
            "Brakuje biblioteki faster-whisper.\n"
            "Zainstaluj ją poleceniem:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.plik).expanduser().resolve()
    if not input_path.exists():
        print(f"Nie znaleziono pliku: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output).expanduser().resolve() if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    json_path = output_dir / f"{stem}.json"

    print(f"Ładuję model: {args.model}")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(f"Transkrybuję po polsku: {input_path}")
    try:
        raw_segments, info = model.transcribe(
            str(input_path),
            language="pl",
            beam_size=5,
            vad_filter=not args.bez_vad,
            word_timestamps=args.word_timestamps,
            condition_on_previous_text=not args.bez_kontekstu,
        )
        segments = []
        for segment in raw_segments:
            item = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            if args.word_timestamps and segment.words:
                item["words"] = [
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": word.word.strip(),
                        "probability": float(word.probability),
                    }
                    for word in segment.words
                ]
            segments.append(item)
    except Exception as exc:
        print(
            "Nie udało się odczytać albo przetworzyć pliku.\n"
            "Jeśli to plik wideo, zainstaluj FFmpeg albo wyeksportuj sam dźwięk do MP3/WAV.\n"
            f"Szczegóły błędu: {exc}",
            file=sys.stderr,
        )
        return 1

    if not segments:
        print("Nie wykryłem mowy w pliku.", file=sys.stderr)
        return 1

    write_txt(txt_path, segments)
    print(f"Zapisano tekst: {txt_path}")

    if not args.bez_srt:
        write_srt(srt_path, segments)
        print(f"Zapisano napisy z timestampami: {srt_path}")

    if args.json:
        write_json(json_path, segments)
        print(f"Zapisano JSON: {json_path}")

    print(f"Wykryty język: {info.language}, pewność: {info.language_probability:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
