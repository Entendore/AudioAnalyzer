#!/usr/bin/env python3
"""
worker.py — Background worker for loading and analysing audio files.

Uses QThread + signals to keep the UI responsive during heavy I/O and
computation. Reports progress phases so the UI can show a loading indicator.
"""

from PySide6.QtCore import QThread, Signal

from audio_data import AudioData


class AnalysisWorker(QThread):
    """
    Load and analyse an audio file in a background thread.

    Signals:
      progress(str)   — phase name ("loading", "analysing", "done")
      finished(AudioData) — the loaded + analysed AudioData on success
      error(str)      — error message on failure
    """

    progress = Signal(str)
    finished = Signal(object)      # AudioData
    error = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._file_path = file_path

    def run(self):
        try:
            self.progress.emit("loading")
            audio = AudioData()
            audio.load_file(self._file_path)
            self.progress.emit("done")
            self.finished.emit(audio)
        except Exception as exc:
            self.error.emit(str(exc))
