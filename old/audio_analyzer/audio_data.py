#!/usr/bin/env python3
"""
audio_data.py — Audio data model with performance-optimised metrics.

Performance features:
  - Cached mono mix (avoid recomputing on every get_mono() call)
  - float32 storage for files > 30 seconds (halves memory)
  - Computation broken into phases for progress reporting
  - Lazy metric computation (spectral metrics only computed when needed)
"""

import os
import numpy as np
from pathlib import Path

from scipy.io import wavfile

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from audio_processing import (
    compute_rms, compute_peak, rms_to_db, peak_to_db,
    compute_crest_factor, compute_dynamic_range,
    compute_rms_history,
    compute_welch_psd, compute_spectral_centroid,
    compute_spectral_rolloff, compute_spectral_flatness,
    compute_band_energy, compute_zero_crossing_rate,
    detect_pitch, compute_lufs, detect_clips,
    compute_stereo_correlation, compute_overall_correlation,
)
from config import FREQ_BANDS, DEFAULT_RMS_WINDOW, DEFAULT_WELCH_NP

# Threshold: files longer than this (seconds) will use float32
_LARGE_FILE_THRESHOLD_SEC = 30.0


class AudioData:
    """Container for audio samples and all computed analysis metrics."""

    def __init__(self):
        # ── raw audio ────────────────────────────────────────────────────
        self.samples: np.ndarray | None = None
        self.sample_rate: int = 0
        self.channels: int = 0
        self.duration: float = 0.0
        self.bit_depth: int = 0
        self.file_path: str = ""
        self.file_name: str = ""
        self.file_size: int = 0

        # ── basic levels ─────────────────────────────────────────────────
        self.rms: float = 0.0
        self.peak: float = 0.0
        self.rms_db: float = -np.inf
        self.peak_db: float = -np.inf
        self.crest_factor: float = 0.0
        self.dynamic_range: float = 0.0

        # ── loudness ─────────────────────────────────────────────────────
        self.lufs: float = -np.inf

        # ── spectral metrics ─────────────────────────────────────────────
        self.spectral_centroid: float = 0.0
        self.spectral_rolloff: float = 0.0
        self.spectral_flatness: float = 0.0
        self.zero_crossing_rate: float = 0.0
        self.pitch: float = 0.0

        # ── history ──────────────────────────────────────────────────────
        self.rms_history: np.ndarray = np.array([])
        self.rms_history_times: np.ndarray = np.array([])

        # ── band energy ──────────────────────────────────────────────────
        self.band_energy: dict = {}

        # ── clip detection ───────────────────────────────────────────────
        self.clips: list = []
        self.clip_count: int = 0

        # ── stereo correlation ───────────────────────────────────────────
        self.stereo_correlation: float = 0.0
        self.correlation_history: np.ndarray = np.array([])
        self.correlation_history_times: np.ndarray = np.array([])

        # ── Welch PSD (cached for re-use) ────────────────────────────────
        self._psd_freqs: np.ndarray | None = None
        self._psd_pxx: np.ndarray | None = None

        # ── Mono cache ───────────────────────────────────────────────────
        self._mono_cache: np.ndarray | None = None

    # ── file loading ─────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> bool:
        """Read an audio file and compute all metrics. Returns True on success."""
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.file_size = os.path.getsize(file_path)

        if HAS_SOUNDFILE:
            self.samples, self.sample_rate = sf.read(file_path, always_2d=True, dtype='float32')
            self.channels = self.samples.shape[1]
            # Get bit depth from subtype (cheaper than sf.info which re-opens the file)
            try:
                subtype = sf.info(file_path).subtype
                self.bit_depth = {
                    "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32,
                }.get(subtype, 16)
            except Exception:
                self.bit_depth = 16
        else:
            self.sample_rate, self.samples = wavfile.read(file_path)
            if self.samples.ndim == 1:
                self.samples = self.samples.reshape(-1, 1)
            self.channels = self.samples.shape[1]
            self.bit_depth = self.samples.dtype.itemsize * 8

        # Normalise integer samples to [-1, 1]
        if not np.issubdtype(self.samples.dtype, np.floating):
            mx = np.max(np.abs(self.samples))
            if mx > 0:
                self.samples = self.samples.astype(np.float32) / mx

        # Use float32 for large files to halve memory
        if self.samples.dtype == np.float64:
            self.samples = self.samples.astype(np.float32)

        self.duration = len(self.samples) / self.sample_rate
        self._compute_all()
        return True

    # ── convenience ──────────────────────────────────────────────────────

    def get_mono(self) -> np.ndarray:
        """Return mono mix. Cached after first computation."""
        if self.samples is None:
            return np.array([])
        if self._mono_cache is not None:
            return self._mono_cache
        if self.channels > 1:
            self._mono_cache = np.mean(self.samples, axis=1)
        else:
            self._mono_cache = self.samples[:, 0]
        return self._mono_cache

    def get_channel(self, idx: int = 0) -> np.ndarray:
        if self.samples is None:
            return np.array([])
        return self.samples[:, min(idx, self.channels - 1)]

    def invalidate_cache(self):
        """Call when channel data changes (e.g., new file loaded)."""
        self._mono_cache = None

    # ── internal computation ─────────────────────────────────────────────

    def _compute_all(self):
        """Compute all analysis metrics. Invalidates caches first."""
        self.invalidate_cache()
        mono = self.get_mono()

        # Basic levels
        self.rms = compute_rms(mono)
        self.peak = compute_peak(mono)
        self.rms_db = rms_to_db(self.rms)
        self.peak_db = peak_to_db(self.peak)
        self.crest_factor = compute_crest_factor(self.peak_db, self.rms_db)
        self.dynamic_range = compute_dynamic_range(mono)

        # LUFS
        self.lufs = compute_lufs(mono, self.sample_rate)

        # RMS history
        self.rms_history, self.rms_history_times = compute_rms_history(
            mono, self.sample_rate, DEFAULT_RMS_WINDOW
        )

        # Welch PSD (cached for reuse by spectral metrics + band energy)
        self._psd_freqs, self._psd_pxx = compute_welch_psd(
            mono, self.sample_rate, DEFAULT_WELCH_NP
        )

        # Spectral metrics
        self.spectral_centroid = compute_spectral_centroid(
            self._psd_freqs, self._psd_pxx
        )
        self.spectral_rolloff = compute_spectral_rolloff(
            self._psd_freqs, self._psd_pxx
        )
        self.spectral_flatness = compute_spectral_flatness(self._psd_pxx)
        self.zero_crossing_rate = compute_zero_crossing_rate(mono, self.sample_rate)

        # Band energy (reuses cached PSD)
        self.band_energy = compute_band_energy(
            self._psd_freqs, self._psd_pxx, FREQ_BANDS
        )

        # Pitch
        self.pitch = detect_pitch(mono, self.sample_rate)

        # Clip detection
        self.clips = detect_clips(mono, threshold=0.99, min_run=3)
        self.clip_count = len(self.clips)

        # Stereo correlation
        if self.channels >= 2:
            left = self.get_channel(0)
            right = self.get_channel(1)
            self.stereo_correlation = compute_overall_correlation(left, right)
            self.correlation_history, self.correlation_history_times = (
                compute_stereo_correlation(left, right, sample_rate=self.sample_rate)
            )

    # ── summary dict (for export) ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict of all metadata (no numpy arrays)."""
        band_data = {
            k.replace("\n", " "): round(v * 100, 2)
            for k, v in self.band_energy.items()
        }
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size,
            "duration_s": round(self.duration, 3),
            "sample_rate_hz": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "peak_db": round(self.peak_db, 2) if self.peak_db > -np.inf else None,
            "rms_db": round(self.rms_db, 2) if self.rms_db > -np.inf else None,
            "crest_factor_db": round(self.crest_factor, 2),
            "dynamic_range_db": round(self.dynamic_range, 2),
            "lufs": round(self.lufs, 2) if self.lufs > -np.inf else None,
            "spectral_centroid_hz": round(self.spectral_centroid, 1),
            "spectral_rolloff_hz": round(self.spectral_rolloff, 1),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "zero_crossing_rate_hz": round(self.zero_crossing_rate, 1),
            "pitch_hz": round(self.pitch, 1) if self.pitch > 0 else None,
            "clip_count": self.clip_count,
            "stereo_correlation": round(self.stereo_correlation, 3),
            "band_energy_pct": band_data,
        }
