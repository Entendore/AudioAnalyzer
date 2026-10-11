#!/usr/bin/env python3
"""
canvases.py — All Matplotlib plot canvases for the Audio Analyzer.

Performance-optimized with:
  - Min/max envelope for waveform (preserves peaks, fewer points)
  - Blitted live spectrum (no full redraw per frame)
  - Pre-allocated line objects for live updates
  - Reused spectrogram computation
"""

import numpy as np
from PySide6.QtCore import QTimer

from base_canvas import BaseCanvas
from audio_data import AudioData
from audio_processing import (
    compute_fft_spectrum, compute_spectrogram,
    compute_waveform_envelope,
)
from config import (
    DARK_AXIS, SURFACE0, LIGHT_TXT, SUBTXT0, GRID_CLR,
    BLUE, GREEN, RED, YELLOW, MAUVE, PEACH, TEAL, SKY, OVERLAY1,
    MAX_WAVEFORM_PTS, MAX_SPECTROGRAM_SEC,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Waveform (min/max envelope downsampling)
# ──────────────────────────────────────────────────────────────────────────────

class WaveformCanvas(BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, height_ratio=3)

    def plot(self, audio: AudioData, channel: int = 0):
        self.clear_plot()
        self._style_ax("Time (s)", "Amplitude", "Waveform")
        if audio is None or audio.samples is None:
            self.draw()
            return

        samples = audio.get_channel(channel)

        # Use min/max envelope for accurate peak-preserving downsampling
        time_norm, min_vals, max_vals = compute_waveform_envelope(
            samples, MAX_WAVEFORM_PTS
        )
        time = time_norm * audio.duration

        if len(time) == len(samples):
            # Short file: plot directly
            self.ax.fill_between(time, samples, alpha=0.08, color=BLUE)
            self.ax.plot(time, samples, color=BLUE, linewidth=0.4, alpha=0.9)
        else:
            # Downsampled: draw envelope as fill between min/max
            self.ax.fill_between(time, min_vals, max_vals, alpha=0.25, color=BLUE)
            # Center line (midpoint of min/max)
            mid = (min_vals + max_vals) / 2
            self.ax.plot(time, mid, color=BLUE, linewidth=0.3, alpha=0.7)

        self.ax.set_xlim(0, audio.duration)
        self.ax.set_ylim(-1.05, 1.05)

        # Mark clips with subtle red spans
        if audio.clips:
            for start_s, end_s in audio.clips[:50]:
                t0 = start_s / audio.sample_rate
                t1 = end_s / audio.sample_rate
                self.ax.axvspan(t0, t1, color=RED, alpha=0.18, zorder=0)

        # dB right axis
        ax2 = self.ax.twinx()
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_yticks([-1, -0.707, 0, 0.707, 1])
        ax2.set_yticklabels(["0 dB", "-3 dB", "-inf", "-3 dB", "0 dB"])
        ax2.tick_params(colors=SUBTXT0, labelsize=6, direction="in")
        for spine in ax2.spines.values():
            spine.set_color(SURFACE0)
            spine.set_linewidth(0.8)

        self.fig.tight_layout(pad=1.5)
        self._playback_line = None
        self.draw()

        # Enable blitting for fast playback cursor updates
        self.enable_blitting()


# ──────────────────────────────────────────────────────────────────────────────
#  Frequency Spectrum
# ──────────────────────────────────────────────────────────────────────────────

class SpectrumCanvas(BaseCanvas):
    def plot(self, audio: AudioData, fft_size: int = 4096, channel: int = 0):
        self.clear_plot()
        self._style_ax("Frequency (Hz)", "Magnitude (dB)", "Frequency Spectrum")
        if audio is None or audio.samples is None:
            self.draw()
            return

        samples = audio.get_channel(channel)
        freqs, mag_db = compute_fft_spectrum(samples, audio.sample_rate, fft_size)

        self.ax.fill_between(freqs, mag_db, np.min(mag_db), alpha=0.08, color=GREEN)
        self.ax.plot(freqs, mag_db, color=GREEN, linewidth=0.6, alpha=0.9)
        self.ax.set_xlim(20, audio.sample_rate / 2)
        self.ax.set_xscale("log")
        y_lo = max(-120, np.min(mag_db))
        self.ax.set_ylim(y_lo, np.max(mag_db) + 5)

        for f in (100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            if f <= audio.sample_rate / 2:
                self.ax.axvline(f, color=SURFACE0, lw=0.5, ls=":", alpha=0.6)

        for f, label in [(100, "100"), (1000, "1k"), (10000, "10k")]:
            if f <= audio.sample_rate / 2:
                self.ax.text(
                    f, np.max(mag_db) + 3, label,
                    color=OVERLAY1, fontsize=6, ha="center", va="bottom",
                )

        self.fig.tight_layout(pad=1.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
#  Spectrogram
# ──────────────────────────────────────────────────────────────────────────────

class SpectrogramCanvas(BaseCanvas):
    def plot(
        self,
        audio: AudioData,
        nperseg: int = 1024,
        channel: int = 0,
        log_freq: bool = True,
        cmap: str = "magma",
    ):
        self.clear_plot()
        self._style_ax("Time (s)", "Frequency (Hz)", "Spectrogram")
        if audio is None or audio.samples is None:
            self.draw()
            return

        samples = audio.get_channel(channel)
        max_len = MAX_SPECTROGRAM_SEC * audio.sample_rate
        if len(samples) > max_len:
            samples = samples[:max_len]

        f, t, Sxx_db = compute_spectrogram(samples, audio.sample_rate, nperseg)
        vmin, vmax = np.percentile(Sxx_db, 10), np.percentile(Sxx_db, 99)

        im = self.ax.pcolormesh(
            t, f, Sxx_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
        )

        self._remove_cbar()
        self._cbar = self.fig.colorbar(im, ax=self.ax, pad=0.02, aspect=30)
        self._cbar.set_label("Power (dB)", color=SUBTXT0, fontsize=7, labelpad=4)
        self._cbar.ax.tick_params(colors=SUBTXT0, labelsize=6, direction="in")
        self._cbar.outline.set_edgecolor(SURFACE0)
        self._cbar.outline.set_linewidth(0.8)

        if log_freq:
            self.ax.set_yscale("log")
            self.ax.set_ylim(20, audio.sample_rate / 2)
        else:
            self.ax.set_ylim(0, min(audio.sample_rate / 2, 22050))

        self.fig.tight_layout(pad=1.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
#  RMS / Loudness History
# ──────────────────────────────────────────────────────────────────────────────

class RMSHistoryCanvas(BaseCanvas):
    def plot(self, audio: AudioData):
        self.clear_plot()
        self._style_ax("Time (s)", "RMS Level (dB)", "Loudness History (RMS)")
        if audio is None or audio.samples is None or len(audio.rms_history) == 0:
            self.draw()
            return

        rms_db = 20.0 * np.log10(audio.rms_history + 1e-10)
        self.ax.fill_between(audio.rms_history_times, rms_db, -120, alpha=0.10, color=MAUVE)
        self.ax.plot(audio.rms_history_times, rms_db, color=MAUVE, linewidth=0.8, alpha=0.9)

        self.ax.set_xlim(0, audio.duration)
        self.ax.set_ylim(max(-80, np.min(rms_db) - 5), max(rms_db) + 5)

        self.ax.axhline(-14, color=GREEN, ls="--", lw=0.7, alpha=0.5)
        self.ax.text(audio.duration * 0.01, -13, "Target -14 LUFS",
                     color=GREEN, fontsize=6, alpha=0.7, va="bottom")
        self.ax.axhline(-1, color=RED, ls="--", lw=0.7, alpha=0.5)
        self.ax.text(audio.duration * 0.01, 0, "0 dBFS Peak",
                     color=RED, fontsize=6, alpha=0.7, va="bottom")
        if audio.lufs > -np.inf:
            self.ax.axhline(audio.lufs, color=SKY, ls="-.", lw=0.7, alpha=0.6)
            self.ax.text(audio.duration * 0.99, audio.lufs + 1,
                         f"Integrated {audio.lufs:.1f} LUFS",
                         color=SKY, fontsize=6, alpha=0.8, ha="right", va="bottom")

        self.fig.tight_layout(pad=1.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
#  Phase / Correlation
# ──────────────────────────────────────────────────────────────────────────────

class PhaseCorrelationCanvas(BaseCanvas):
    """Two-panel view: Lissajous (top) + correlation-over-time (bottom)."""

    def __init__(self, parent=None):
        self.fig.clear()
        self.ax_liss = self.fig.add_subplot(211)
        self.ax_corr = self.fig.add_subplot(212)
        super().__init__(parent, height_ratio=4)
        self.fig.clear()
        self.ax_liss = self.fig.add_subplot(211)
        self.ax_corr = self.fig.add_subplot(212)

    def _style_both(self):
        for ax in (self.ax_liss, self.ax_corr):
            ax.set_facecolor(DARK_AXIS)
            ax.tick_params(colors=SUBTXT0, labelsize=6, direction="in")
            for sp in ax.spines.values():
                sp.set_color(SURFACE0)
                sp.set_linewidth(0.8)
            ax.grid(True, alpha=0.1, color=GRID_CLR, linewidth=0.5)

    def plot(self, audio: AudioData, channel: int = 0):
        self.fig.clear()
        self.ax_liss = self.fig.add_subplot(211)
        self.ax_corr = self.fig.add_subplot(212)
        self._style_both()

        if audio is None or audio.samples is None or audio.channels < 2:
            self.ax_liss.set_title(
                "Stereo Phase (requires stereo file)", color=LIGHT_TXT,
                fontsize=10, loc="left", pad=10,
            )
            self.ax_corr.set_visible(False)
            self.fig.tight_layout(pad=1.5)
            self.draw()
            return

        left = audio.get_channel(0)
        right = audio.get_channel(1)

        # ── Lissajous ────────────────────────────────────────────────────
        step = max(1, len(left) // 20_000)
        l, r = left[::step], right[::step]
        self.ax_liss.scatter(r, l, s=0.4, alpha=0.3, c=TEAL, edgecolors="none")
        self.ax_liss.set_xlim(-1.05, 1.05)
        self.ax_liss.set_ylim(-1.05, 1.05)
        self.ax_liss.set_xlabel("Right", color=SUBTXT0, fontsize=7, labelpad=4)
        self.ax_liss.set_ylabel("Left", color=SUBTXT0, fontsize=7, labelpad=4)
        self.ax_liss.set_title(
            f"Stereo Phase  |  Correlation: {audio.stereo_correlation:.2f}",
            color=LIGHT_TXT, fontsize=10, fontweight="bold", loc="left", pad=10,
        )
        self.ax_liss.set_aspect("equal")
        self.ax_liss.axhline(0, color=SURFACE0, lw=0.5, alpha=0.5)
        self.ax_liss.axvline(0, color=SURFACE0, lw=0.5, alpha=0.5)

        # ── Correlation over time ────────────────────────────────────────
        if len(audio.correlation_history) > 0:
            self.ax_corr.fill_between(
                audio.correlation_history_times,
                audio.correlation_history, 0, alpha=0.08, color=PEACH,
            )
            self.ax_corr.plot(
                audio.correlation_history_times,
                audio.correlation_history, color=PEACH, linewidth=0.7, alpha=0.9,
            )
            self.ax_corr.axhline(1, color=GREEN, ls="--", lw=0.5, alpha=0.4)
            self.ax_corr.axhline(0, color=YELLOW, ls="--", lw=0.5, alpha=0.4)
            self.ax_corr.axhline(-1, color=RED, ls="--", lw=0.5, alpha=0.4)
            self.ax_corr.set_ylim(-1.05, 1.05)
            self.ax_corr.set_xlim(0, audio.duration)
            self.ax_corr.set_xlabel("Time (s)", color=SUBTXT0, fontsize=7, labelpad=4)
            self.ax_corr.set_ylabel("Correlation", color=SUBTXT0, fontsize=7, labelpad=4)
            self.ax_corr.set_title(
                "Correlation Over Time", color=LIGHT_TXT, fontsize=10,
                fontweight="bold", loc="left", pad=10,
            )

        self.fig.tight_layout(pad=1.5)
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
#  Live Spectrum (blitted real-time FFT)
# ──────────────────────────────────────────────────────────────────────────────

class LiveSpectrumCanvas(BaseCanvas):
    """FFT spectrum that updates in real time using blitting for performance."""

    def __init__(self, parent=None):
        super().__init__(parent, height_ratio=3)
        self._audio: AudioData | None = None
        self._channel: int = 0
        self._fft_size: int = 4096
        self._timer = QTimer()
        self._timer.setInterval(80)     # ~12 FPS
        self._timer.timeout.connect(self._tick)
        self._last_pos_ms: int = 0

        # Pre-allocated line object for blitting
        self._line = None
        self._fill = None
        self._y_lo: float = -120.0
        self._y_hi: float = 0.0

    def set_source(self, audio: AudioData, channel: int = 0, fft_size: int = 4096):
        self._audio = audio
        self._channel = channel
        self._fft_size = fft_size

    def start_live(self, position_ms: int = 0):
        self._last_pos_ms = position_ms
        self._timer.start()

    def stop_live(self):
        self._timer.stop()

    def update_position(self, position_ms: int):
        self._last_pos_ms = position_ms

    def _init_line(self, freqs, mag_db):
        """Create line and fill artists once for subsequent in-place updates."""
        self.ax.clear()
        self._style_ax("Frequency (Hz)", "Magnitude (dB)", "Live Spectrum")
        self._y_lo = max(-120, np.min(mag_db))
        self._y_hi = np.max(mag_db) + 5
        self._fill = self.ax.fill_between(
            freqs, mag_db, self._y_lo, alpha=0.06, color=SKY,
        )
        self._line, = self.ax.plot(
            freqs, mag_db, color=SKY, linewidth=0.6, alpha=0.9,
        )
        self.ax.set_xlim(20, self._audio.sample_rate / 2)
        self.ax.set_xscale("log")
        self.ax.set_ylim(self._y_lo, self._y_hi)
        self.fig.tight_layout(pad=1.5)
        self.draw()
        self.enable_blitting()

    def _tick(self):
        if self._audio is None or self._audio.samples is None:
            return
        pos_s = self._last_pos_ms / 1000.0
        samples = self._audio.get_channel(self._channel)
        start = int(pos_s * self._audio.sample_rate)
        chunk = samples[start:start + self._fft_size]
        if len(chunk) < self._fft_size:
            chunk = np.pad(chunk, (0, self._fft_size - len(chunk)))

        from audio_processing import _get_hann_window, _get_rfftfreq
        windowed = chunk * _get_hann_window(len(chunk))
        fft = np.fft.rfft(windowed)
        mag_db = 20.0 * np.log10(np.abs(fft) / len(chunk) + 1e-10)
        freqs = _get_rfftfreq(len(chunk), 1.0 / self._audio.sample_rate)

        if self._line is None:
            self._init_line(freqs, mag_db)
        else:
            # In-place update: no clear/redraw
            self._line.set_ydata(mag_db)
            # Update fill: remove and recreate (fill_between can't be updated in-place)
            if self._fill is not None:
                try:
                    self._fill.remove()
                except Exception:
                    pass
            self._fill = self.ax.fill_between(
                freqs, mag_db, self._y_lo, alpha=0.06, color=SKY,
            )
            # Re-save background and blit
            self._save_blit_bg()
            if self._playback_line is not None:
                try:
                    self.ax.draw_artist(self._playback_line)
                except Exception:
                    pass
            self.blit(self.fig.bbox)
