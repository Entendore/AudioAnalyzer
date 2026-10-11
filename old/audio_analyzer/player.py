#!/usr/bin/env python3
"""
player.py — Encapsulated audio playback controller.

Features:
  - Play / Pause / Stop
  - A-B loop (set loop in/out points)
  - Playback speed control (0.25× – 4.0×)
  - Position tracking via QMediaPlayer
  - Integration with waveform seek and live-spectrum canvas
"""

from PySide6.QtCore import QObject, Signal, QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


class PlaybackController(QObject):
    """Centralised playback logic separated from the UI."""

    position_changed  = Signal(int)       # ms
    duration_changed  = Signal(int)       # ms
    state_changed     = Signal(object)    # QMediaPlayer.PlaybackState
    loop_toggled      = Signal(bool)
    speed_changed     = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer()
        self.output = QAudioOutput()
        self.output.setVolume(0.8)
        self.player.setAudioOutput(self.output)

        # Loop points (ms); None = disabled
        self._loop_a: int | None = None
        self._loop_b: int | None = None
        self._loop_enabled: bool = False

        # Speed
        self._speed: float = 1.0

        # Connect signals
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self.duration_changed.emit)
        self.player.playbackStateChanged.connect(self.state_changed.emit)

    # ── basic transport ──────────────────────────────────────────────────

    def load(self, file_path: str):
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self._loop_a = None
        self._loop_b = None
        self._loop_enabled = False

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

    def toggle_play_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def seek(self, ms: int):
        self.player.setPosition(max(0, ms))

    # ── volume ───────────────────────────────────────────────────────────

    def set_volume(self, vol: float):
        """*vol* in [0, 1]."""
        self.output.setVolume(max(0.0, min(1.0, vol)))

    def volume(self) -> float:
        return self.output.volume()

    # ── speed ────────────────────────────────────────────────────────────

    def set_speed(self, speed: float):
        self._speed = max(0.25, min(4.0, speed))
        self.player.setPlaybackRate(self._speed)
        self.speed_changed.emit(self._speed)

    def speed(self) -> float:
        return self._speed

    # ── A-B loop ─────────────────────────────────────────────────────────

    def set_loop_a(self, ms: int | None = None):
        """Set loop start point.  None = current position."""
        self._loop_a = ms if ms is not None else self.player.position()

    def set_loop_b(self, ms: int | None = None):
        """Set loop end point.  None = current position."""
        self._loop_b = ms if ms is not None else self.player.position()

    def enable_loop(self, enabled: bool):
        self._loop_enabled = enabled
        self.loop_toggled.emit(enabled)

    def loop_enabled(self) -> bool:
        return self._loop_enabled

    def loop_a(self) -> int | None:
        return self._loop_a

    def loop_b(self) -> int | None:
        return self._loop_b

    # ── state queries ────────────────────────────────────────────────────

    def position(self) -> int:
        return self.player.position()

    def duration(self) -> int:
        return self.player.duration()

    def state(self):
        return self.player.playbackState()

    # ── internal ─────────────────────────────────────────────────────────

    def _on_position(self, ms: int):
        # Handle A-B loop
        if (
            self._loop_enabled
            and self._loop_a is not None
            and self._loop_b is not None
            and ms >= self._loop_b
        ):
            self.player.setPosition(self._loop_a)
            return
        self.position_changed.emit(ms)
