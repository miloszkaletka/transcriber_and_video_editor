from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel


APP_NAME = "TranscriberVideoEditor"
CREDIT = "Program powstał dzięki Miłosz Kaletka"
MEDIA_TYPES = [
    ("Pliki audio/wideo", "*.mp4 *.mov *.mkv *.avi *.webm *.mp3 *.wav *.m4a"),
    ("Wszystkie pliki", "*.*"),
]


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text']}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def transcribe_whole_file(input_path: Path, output_dir: Path, model: WhisperModel, log) -> None:
    log(f"Transkrybuję jako jeden plik: {input_path.name}")
    raw_segments, info = model.transcribe(
        str(input_path),
        language="pl",
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    segments: list[dict] = []
    for segment in raw_segments:
        item = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        if segment.words:
            item["words"] = [
                {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word.strip(),
                    "probability": float(word.probability),
                }
                for word in segment.words
            ]
        if item["text"]:
            segments.append(item)

    stem = input_path.stem
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    json_path = output_dir / f"{stem}.json"

    text = " ".join(segment["text"] for segment in segments).strip()
    txt_path.write_text(text + "\n", encoding="utf-8")
    write_srt(srt_path, segments)
    json_path.write_text(
        json.dumps(
            {
                "language": info.language,
                "language_probability": float(info.language_probability),
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"Zapisano: {txt_path.name}, {srt_path.name}, {json_path.name}")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("800x560")
        self.minsize(720, 520)

        self.files: list[Path] = []
        self.output_dir = tk.StringVar(value=str(app_base_dir() / "wyniki"))
        self.model_name = tk.StringVar(value="medium")
        self.status = tk.StringVar(value="Gotowy.")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.create_widgets()
        self.after(150, self.consume_logs)

    def create_widgets(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=APP_NAME, font=("Segoe UI", 18, "bold")).pack(anchor=tk.W)
        ttk.Label(frame, text=CREDIT, font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)

        ttk.Button(buttons, text="Wybierz filmy", command=self.pick_files).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Wyczyść listę", command=self.clear_files).pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="Folder wyników", command=self.pick_output_dir).pack(side=tk.LEFT)

        options = ttk.Frame(frame)
        options.pack(fill=tk.X, pady=10)

        ttk.Label(options, text="Model Whisper:").pack(side=tk.LEFT)
        ttk.OptionMenu(options, self.model_name, self.model_name.get(), "small", "medium", "large-v3").pack(
            side=tk.LEFT,
            padx=(6, 18),
        )
        ttk.Label(options, text="Program transkrybuje każdy film jako jeden plik.").pack(side=tk.LEFT)

        ttk.Label(frame, text="Wybrane pliki:").pack(anchor=tk.W)
        self.file_list = tk.Listbox(frame, height=8)
        self.file_list.pack(fill=tk.X, pady=(2, 10))

        ttk.Label(frame, text="Folder wyników:").pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self.output_dir).pack(fill=tk.X, pady=(2, 10))

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.pack(anchor=tk.W)

        ttk.Label(frame, textvariable=self.status).pack(anchor=tk.W, pady=(10, 2))

        self.log_box = tk.Text(frame, height=15, wrap=tk.WORD)
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
                transcribe_whole_file(input_path, output_dir, model, self.log)

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
