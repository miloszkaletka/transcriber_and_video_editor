from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel

from edytuj_przedzialy import render_intervals
from transkrybuj_takeami import detect_takes, format_timestamp, load_audio


APP_NAME = "TranscriberVideoEditor"
CREDIT = "Program powstał dzięki Miłosz Kaletka"
MEDIA_TYPES = [
    ("Pliki audio/wideo", "*.mp4 *.mov *.mkv *.avi *.webm *.mp3 *.wav *.m4a"),
    ("Wszystkie pliki", "*.*"),
]


def write_srt(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text']}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def transcribe_by_takes(
    input_path: Path,
    output_dir: Path,
    model: WhisperModel,
    log,
    split_pause: float = 0.15,
    padding: float = 0.08,
    window: float = 0.04,
) -> tuple[list[dict], list[tuple[float, float]], Path]:
    sample_rate = 16000
    log(f"Czytam audio: {input_path.name}")
    audio = load_audio(input_path, sample_rate=sample_rate)
    if len(audio) == 0:
        raise RuntimeError("Nie znaleziono audio w pliku.")

    takes = detect_takes(
        audio,
        sample_rate=sample_rate,
        window_seconds=window,
        split_pause=split_pause,
        padding=padding,
        threshold_db=None,
    )

    log(f"Wykryte take'i: {len(takes)}")
    segments: list[dict] = []

    for take_index, (take_start, take_end) in enumerate(takes, start=1):
        start_sample = max(0, int(take_start * sample_rate))
        end_sample = min(len(audio), int(take_end * sample_rate))
        take_audio = audio[start_sample:end_sample]
        if len(take_audio) < sample_rate * 0.15:
            continue

        log(f"  Take {take_index}/{len(takes)}: {take_start:.2f}-{take_end:.2f}s")
        raw_segments, _ = model.transcribe(
            take_audio,
            language="pl",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        for segment in raw_segments:
            text = segment.text.strip()
            if not text:
                continue

            item = {
                "take": take_index,
                "take_start": take_start,
                "take_end": take_end,
                "start": take_start + float(segment.start),
                "end": take_start + float(segment.end),
                "text": text,
            }
            if segment.words:
                item["words"] = [
                    {
                        "start": take_start + float(word.start),
                        "end": take_start + float(word.end),
                        "word": word.word.strip(),
                        "probability": float(word.probability),
                    }
                    for word in segment.words
                ]
            segments.append(item)

    stem = input_path.stem
    txt_path = output_dir / f"{stem}_takeami.txt"
    srt_path = output_dir / f"{stem}_takeami.srt"
    json_path = output_dir / f"{stem}_takeami.json"
    takes_path = output_dir / f"{stem}_takeami_takei.txt"

    txt_path.write_text("\n".join(segment["text"] for segment in segments) + "\n", encoding="utf-8")
    write_srt(srt_path, segments)
    json_path.write_text(
        json.dumps({"language": "pl", "takes": takes, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    takes_path.write_text("\n".join(f"{start:.2f}-{end:.2f}" for start, end in takes) + "\n", encoding="utf-8")

    log(f"Zapisano: {txt_path.name}, {srt_path.name}, {json_path.name}")
    return segments, takes, json_path


def detect_dynamic_intervals(input_path: Path, log) -> list[tuple[float, float]]:
    sample_rate = 16000
    log("Wykrywam dynamiczne fragmenty mowy do renderu 1080p...")
    audio = load_audio(input_path, sample_rate=sample_rate)
    intervals = detect_takes(
        audio,
        sample_rate=sample_rate,
        window_seconds=0.08,
        split_pause=0.50,
        padding=0.18,
        threshold_db=None,
    )
    return [(start, end) for start, end in intervals if end - start >= 0.25]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("820x620")
        self.minsize(760, 560)

        self.files: list[Path] = []
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "wyniki"))
        self.model_name = tk.StringVar(value="medium")
        self.render_dynamic = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Gotowy.")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.create_widgets()
        self.after(150, self.consume_logs)

    def create_widgets(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text=APP_NAME, font=("Segoe UI", 18, "bold"))
        title.pack(anchor=tk.W)

        credit = ttk.Label(frame, text=CREDIT, font=("Segoe UI", 10))
        credit.pack(anchor=tk.W, pady=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)

        ttk.Button(buttons, text="Wybierz filmy", command=self.pick_files).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Wyczyść listę", command=self.clear_files).pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="Folder wyników", command=self.pick_output_dir).pack(side=tk.LEFT)

        options = ttk.Frame(frame)
        options.pack(fill=tk.X, pady=10)

        ttk.Label(options, text="Model Whisper:").pack(side=tk.LEFT)
        ttk.OptionMenu(options, self.model_name, self.model_name.get(), "small", "medium", "large-v3").pack(side=tk.LEFT, padx=(6, 18))
        ttk.Checkbutton(
            options,
            text="Po transkrypcji wygeneruj automatyczny dynamiczny render 1080p",
            variable=self.render_dynamic,
        ).pack(side=tk.LEFT)

        ttk.Label(frame, text="Wybrane pliki:").pack(anchor=tk.W)
        self.file_list = tk.Listbox(frame, height=8)
        self.file_list.pack(fill=tk.X, pady=(2, 10))

        ttk.Label(frame, text="Folder wyników:").pack(anchor=tk.W)
        output_entry = ttk.Entry(frame, textvariable=self.output_dir)
        output_entry.pack(fill=tk.X, pady=(2, 10))

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.pack(anchor=tk.W)

        ttk.Label(frame, textvariable=self.status).pack(anchor=tk.W, pady=(10, 2))

        self.log_box = tk.Text(frame, height=16, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def pick_files(self) -> None:
        selected = filedialog.askopenfilenames(title="Wybierz filmy lub audio", filetypes=MEDIA_TYPES)
        for item in selected:
            path = Path(item)
            if path not in self.files:
                self.files.append(path)
                self.file_list.insert(tk.END, str(path))

    def clear_files(self) -> None:
        self.files.clear()
        self.file_list.delete(0, tk.END)

    def pick_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Wybierz folder wyników")
        if selected:
            self.output_dir.set(selected)

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def consume_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_box.insert(tk.END, message + "\n")
            self.log_box.see(tk.END)
        self.after(150, self.consume_logs)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.files:
            messagebox.showwarning(APP_NAME, "Najpierw wybierz jeden lub kilka plików.")
            return

        self.start_button.configure(state=tk.DISABLED)
        self.status.set("Przetwarzanie...")
        self.worker = threading.Thread(target=self.process_files, daemon=True)
        self.worker.start()

    def process_files(self) -> None:
        try:
            output_dir = Path(self.output_dir.get()).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            self.log(f"Ładuję model: {self.model_name.get()}")
            model = WhisperModel(self.model_name.get(), device="cpu", compute_type="int8")

            for index, input_path in enumerate(self.files, start=1):
                self.log("")
                self.log(f"[{index}/{len(self.files)}] {input_path.name}")
                transcribe_by_takes(input_path, output_dir, model, self.log)

                if self.render_dynamic.get():
                    intervals = detect_dynamic_intervals(input_path, self.log)
                    intervals_path = output_dir / f"{input_path.stem}_dynamic_intervals.txt"
                    intervals_path.write_text(
                        "\n".join(f"{start:.2f}-{end:.2f}" for start, end in intervals) + "\n",
                        encoding="utf-8",
                    )
                    output_video = output_dir / f"{input_path.stem}_DYNAMIC_1080p.mp4"
                    render_intervals(
                        input_path=input_path,
                        output_path=output_video,
                        intervals=intervals,
                        width=1920,
                        height=1080,
                        crf="21",
                        preset="veryfast",
                    )
                    self.log(f"Zapisano render: {output_video.name}")

            self.status.set("Gotowe.")
            self.log("")
            self.log("Gotowe. Pliki znajdziesz w folderze wyników.")
        except Exception as exc:
            self.status.set("Błąd.")
            self.log(f"BŁĄD: {exc}")
            messagebox.showerror(APP_NAME, str(exc))
        finally:
            self.start_button.configure(state=tk.NORMAL)


if __name__ == "__main__":
    App().mainloop()
