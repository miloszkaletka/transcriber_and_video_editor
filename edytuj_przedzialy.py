from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
from av.audio.resampler import AudioResampler


def parse_interval(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split("-", 1)
        start = float(start_text.replace(",", "."))
        end = float(end_text.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Nieprawidłowy przedział '{value}'. Użyj formatu start-koniec, np. 25.1-43.9"
        ) from exc

    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(f"Nieprawidłowy przedział '{value}'.")
    return start, end


def is_kept(time_seconds: float | None, intervals: list[tuple[float, float]]) -> bool:
    if time_seconds is None:
        return False
    return any(start <= time_seconds <= end for start, end in intervals)


def render_intervals(
    input_path: Path,
    output_path: Path,
    intervals: list[tuple[float, float]],
    width: int | None = None,
    height: int | None = None,
    crf: str = "21",
    preset: str = "veryfast",
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intervals = sorted(intervals)
    input_container = av.open(str(input_path))
    output_container = av.open(str(output_path), mode="w")

    video_in = input_container.streams.video[0]
    audio_in = input_container.streams.audio[0] if input_container.streams.audio else None
    video_in.thread_type = "AUTO"

    fps = video_in.average_rate or video_in.base_rate or 30
    video_time_base = Fraction(fps.denominator, fps.numerator) if hasattr(fps, "denominator") else Fraction(1, int(fps))
    video_out = output_container.add_stream("libx264", rate=fps)
    video_out.width = width or video_in.codec_context.width
    video_out.height = height or video_in.codec_context.height
    video_out.pix_fmt = "yuv420p"
    video_out.options = {"crf": str(crf), "preset": preset, "movflags": "+faststart"}

    audio_out = None
    audio_resampler = None
    if audio_in is not None:
        audio_rate = audio_in.codec_context.sample_rate or 48000
        audio_time_base = Fraction(1, audio_rate)
        audio_out = output_container.add_stream("aac", rate=audio_rate)
        audio_out.layout = "stereo"
        audio_resampler = AudioResampler(format="fltp", layout="stereo", rate=audio_rate)
    else:
        audio_time_base = None

    kept_video_frames = 0
    kept_audio_frames = 0
    audio_samples = 0

    streams = [video_in]
    if audio_in is not None:
        streams.append(audio_in)

    for frame in input_container.decode(*streams):
        if not is_kept(frame.time, intervals):
            continue

        if isinstance(frame, av.VideoFrame):
            if width and height:
                frame = frame.reformat(width=width, height=height, format="yuv420p")
            frame.pts = kept_video_frames
            frame.time_base = video_time_base
            for packet in video_out.encode(frame):
                output_container.mux(packet)
            kept_video_frames += 1
        elif isinstance(frame, av.AudioFrame) and audio_out is not None:
            for resampled_frame in audio_resampler.resample(frame):
                resampled_frame.pts = audio_samples
                resampled_frame.time_base = audio_time_base
                for packet in audio_out.encode(resampled_frame):
                    output_container.mux(packet)
                audio_samples += resampled_frame.samples
                kept_audio_frames += 1

    for packet in video_out.encode():
        output_container.mux(packet)

    if audio_out is not None:
        for packet in audio_out.encode():
            output_container.mux(packet)

    output_container.close()
    input_container.close()
    return kept_video_frames, kept_audio_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Renderuje MP4 z wybranych przedziałów czasu.")
    parser.add_argument("input", help="Plik wejściowy MP4")
    parser.add_argument("output", help="Plik wynikowy MP4")
    parser.add_argument(
        "--keep",
        action="append",
        required=True,
        type=parse_interval,
        help="Przedział do zostawienia w sekundach, np. --keep 25.1-43.9. Można podać wiele razy.",
    )
    parser.add_argument("--crf", default="20", help="Jakość H.264. Niżej = lepiej/większy plik.")
    parser.add_argument("--preset", default="veryfast", help="Preset H.264, np. veryfast, medium.")
    parser.add_argument("--width", type=int, default=None, help="Szerokość wyniku, np. 1920.")
    parser.add_argument("--height", type=int, default=None, help="Wysokość wyniku, np. 1080.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    kept_video_frames, kept_audio_frames = render_intervals(
        input_path=input_path,
        output_path=output_path,
        intervals=args.keep,
        width=args.width,
        height=args.height,
        crf=args.crf,
        preset=args.preset,
    )

    print(f"Zapisano: {output_path}")
    print(f"Klatki wideo: {kept_video_frames}")
    print(f"Klatki audio: {kept_audio_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
