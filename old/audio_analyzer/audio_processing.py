#!/usr/bin/env python3
"""
audio_processing.py — Standalone audio analysis functions.

Performance-optimized with:
  - FFT-based autocorrelation for pitch detection (O(n log n) vs O(n²))
  - Vectorized clip detection (no Python loop)
  - Vectorized stereo correlation (batched Pearson r)
  - Pre-allocated arrays, minimal allocations in hot paths
  - All heavy computation extracted for offloading to worker threads
"""

import numpy as np
from scipy import signal as scipy_signal

# ──────────────────────────────────────────────────────────────────────────────
#  Basic Level Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_rms(samples: np.ndarray) -> float:
    """Root-mean-square of a 1-D sample array."""
    return float(np.sqrt(np.mean(samples * samples)))    # avoid **2 overhead


def compute_peak(samples: np.ndarray) -> float:
    """Peak absolute value."""
    return float(np.max(np.abs(samples)))


def rms_to_db(rms: float) -> float:
    """Convert RMS to dBFS. Returns -inf for zero/negative input."""
    return 20.0 * np.log10(rms) if rms > 0 else -np.inf


def peak_to_db(peak: float) -> float:
    """Convert peak amplitude to dBFS."""
    return 20.0 * np.log10(peak) if peak > 0 else -np.inf


def compute_crest_factor(peak_db: float, rms_db: float) -> float:
    """Crest factor in dB (peak – RMS)."""
    return peak_db - rms_db if rms_db > -np.inf else 0.0


def compute_dynamic_range(samples: np.ndarray) -> float:
    """Dynamic range estimate via 95th/5th percentile of absolute values."""
    abs_s = np.abs(samples)
    nonzero = abs_s[abs_s > 0]
    if len(nonzero) == 0:
        return 0.0
    # Single percentile call is faster than two separate calls
    p5, p95 = np.percentile(nonzero, [5, 95])
    return 20.0 * np.log10(p95 / p5) if p5 > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  RMS History (windowed loudness envelope)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rms_history(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (rms_values, time_stamps) arrays for a mono signal.
    Vectorised: reshapes into (num_wins, win_size) and computes along axis=1.
    """
    win_size = max(1, int(window_sec * sample_rate))
    if len(samples) <= win_size:
        rms_val = compute_rms(samples)
        return np.array([rms_val]), np.array([0.0])

    num_wins = len(samples) // win_size
    trimmed = samples[: num_wins * win_size].reshape(num_wins, win_size)
    rms_vals = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    times = np.linspace(0, num_wins * win_size / sample_rate, num_wins)
    return rms_vals, times


# ──────────────────────────────────────────────────────────────────────────────
#  Spectral Metrics (via Welch PSD)
# ──────────────────────────────────────────────────────────────────────────────

def compute_welch_psd(
    samples: np.ndarray,
    sample_rate: int,
    nperseg: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, Pxx) via Welch's method."""
    return scipy_signal.welch(samples, fs=sample_rate, nperseg=nperseg)


def compute_spectral_centroid(freqs: np.ndarray, Pxx: np.ndarray) -> float:
    """Weighted mean frequency (spectral centroid)."""
    total = np.sum(Pxx)
    if total == 0:
        return 0.0
    return float(np.dot(freqs, Pxx) / total)       # dot product faster than sum(freqs*Pxx)


def compute_spectral_rolloff(freqs: np.ndarray, Pxx: np.ndarray, threshold: float = 0.85) -> float:
    """Frequency below which *threshold* of total power resides."""
    cumsum = np.cumsum(Pxx)
    idx = np.searchsorted(cumsum, threshold * cumsum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def compute_spectral_flatness(Pxx: np.ndarray) -> float:
    """
    Wiener entropy: geometric mean / arithmetic mean of PSD.
    Computed via log-mean for numerical stability.
    """
    Pxx_pos = Pxx[Pxx > 0]
    if len(Pxx_pos) == 0:
        return 0.0
    log_mean = np.mean(np.log(Pxx_pos))
    arith_mean = np.mean(Pxx_pos)
    if arith_mean <= 0:
        return 0.0
    return float(np.exp(log_mean) / arith_mean)


def compute_band_energy(freqs: np.ndarray, Pxx: np.ndarray, bands: list) -> dict:
    """
    Compute normalised energy per frequency band.
    Uses searchsorted for O(n log n) instead of per-band boolean masking.
    """
    total = np.sum(Pxx)
    if total == 0:
        return {label: 0.0 for label, _, _ in bands}

    cumsum = np.cumsum(Pxx)
    result = {}
    for label, f_lo, f_hi in bands:
        idx_lo = np.searchsorted(freqs, f_lo)
        idx_hi = np.searchsorted(freqs, f_hi)
        result[label] = float((cumsum[min(idx_hi, len(cumsum) - 1)] - (cumsum[idx_lo - 1] if idx_lo > 0 else 0)) / total)
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Zero-Crossing Rate
# ──────────────────────────────────────────────────────────────────────────────

def compute_zero_crossing_rate(samples: np.ndarray, sample_rate: int) -> float:
    """Zero-crossing rate in Hz. Uses diff of sign bits for speed."""
    if len(samples) < 2:
        return 0.0
    signs = np.signbit(samples)            # faster than np.sign for this use case
    crossings = np.count_nonzero(np.diff(signs))
    return float(crossings / (2 * len(samples)) * sample_rate)


# ──────────────────────────────────────────────────────────────────────────────
#  Pitch Detection — FFT-based autocorrelation (O(n log n))
# ──────────────────────────────────────────────────────────────────────────────

def detect_pitch(
    samples: np.ndarray,
    sample_rate: int,
    f_min: float = 50.0,
    f_max: float = 2000.0,
) -> float:
    """
    Estimate the fundamental frequency (Hz) using FFT-based autocorrelation.
    Complexity O(n log n) vs O(n²) for np.correlate.
    Returns 0.0 if no clear pitch is found.
    """
    n = len(samples)
    lag_min = max(2, int(sample_rate / f_max))
    lag_max = min(n // 2, int(sample_rate / f_min))
    if lag_max <= lag_min:
        return 0.0

    # Use only a segment for very long files (pitch is usually short-range)
    max_analysis = int(sample_rate * 2)   # 2 seconds max for pitch
    if n > max_analysis:
        # Use the middle segment
        start = (n - max_analysis) // 2
        samples = samples[start:start + max_analysis]
        n = len(samples)

    # FFT-based autocorrelation
    # Zero-pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * n:
        fft_size <<= 1

    fft = np.fft.rfft(samples, n=fft_size)
    power_spectrum = fft * np.conj(fft)
    corr = np.fft.irfft(power_spectrum)[:n]

    # Normalise
    corr = corr / (corr[0] if corr[0] > 0 else 1.0)

    # Restrict to lag range
    corr_region = corr[lag_min:lag_max + 1]

    # Find first dip then first peak after dip
    diff = np.diff(corr_region)
    zero_crossings = np.where(np.diff(np.signbit(diff)))[0]
    if len(zero_crossings) == 0:
        return 0.0

    start_search = zero_crossings[0]

    # Find peak after dip
    search = corr_region[start_search:]
    if len(search) == 0:
        return 0.0
    peak_local = np.argmax(search)
    peak_idx = peak_local + start_search + lag_min

    if peak_idx < 1 or peak_idx >= n - 1:
        return float(sample_rate / peak_idx) if peak_idx > 0 else 0.0

    # Parabolic interpolation for sub-sample accuracy
    alpha = corr[peak_idx - 1]
    beta = corr[peak_idx]
    gamma = corr[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-12:
        refined = float(peak_idx)
    else:
        p = 0.5 * (alpha - gamma) / denom
        refined = peak_idx + p

    if refined <= 0:
        return 0.0
    pitch = sample_rate / refined
    return pitch if f_min <= pitch <= f_max else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  LUFS Estimation (simplified ITU-R BS.1770)
# ──────────────────────────────────────────────────────────────────────────────

# Cache filter coefficients (they only depend on sample rate, not audio data)
_lufs_filter_cache: dict[int, tuple] = {}

def compute_lufs(samples: np.ndarray, sample_rate: int) -> float:
    """
    Simplified integrated loudness estimate (LUFS).
    Caches filter coefficients per sample rate to avoid repeated computation.
    """
    if len(samples) == 0:
        return -np.inf

    # Use cached filters if available
    if sample_rate in _lufs_filter_cache:
        b_hp, a_hp, b_hs, a_hs = _lufs_filter_cache[sample_rate]
    else:
        b_hp, a_hp = scipy_signal.butter(1, 60.0 / (sample_rate / 2), btype="high")
        b_hs, a_hs = scipy_signal.butter(1, 1000.0 / (sample_rate / 2), btype="high")
        _lufs_filter_cache[sample_rate] = (b_hp, a_hp, b_hs, a_hs)

    # Pre-filter: high-pass at ~60 Hz
    filtered = scipy_signal.filtfilt(b_hp, a_hp, samples)

    # High-shelf boost ~4 kHz
    low = scipy_signal.filtfilt(b_hs, a_hs, filtered)
    high = filtered - low
    high *= 10 ** (4.0 / 20.0)       # +4 dB
    weighted = low + high

    mean_sq = np.mean(weighted * weighted)
    if mean_sq <= 0:
        return -np.inf
    return -0.691 + 10.0 * np.log10(mean_sq)


# ──────────────────────────────────────────────────────────────────────────────
#  Clip Detection — Vectorized
# ──────────────────────────────────────────────────────────────────────────────

def detect_clips(
    samples: np.ndarray,
    threshold: float = 0.99,
    min_run: int = 3,
) -> list[tuple[int, int]]:
    """
    Find contiguous runs of samples at or above *threshold*.
    Fully vectorized: uses np.diff on boolean array instead of Python loop.
    """
    above = np.abs(samples) >= threshold
    if not np.any(above):
        return []

    # Find run boundaries via diff
    diff = np.diff(above.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1    # rising edges
    ends = np.where(diff == -1)[0] + 1      # falling edges

    # Handle edge cases: starts at 0 or ends at len
    if above[0]:
        starts = np.concatenate(([0], starts))
    if above[-1]:
        ends = np.concatenate((ends, [len(above)]))

    # Filter by minimum run length
    lengths = ends - starts
    valid = lengths >= min_run
    runs = list(zip(starts[valid].tolist(), (ends[valid] - 1).tolist()))
    return runs


# ──────────────────────────────────────────────────────────────────────────────
#  Stereo Correlation — Vectorized
# ──────────────────────────────────────────────────────────────────────────────

def compute_stereo_correlation(
    left: np.ndarray,
    right: np.ndarray,
    window_sec: float = 0.05,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-window Pearson correlation between left and right channels.
    Vectorized: reshapes into (num_wins, win_size) and computes along axis=1.
    Returns (correlation_values, time_stamps).
    """
    win = max(1, int(window_sec * sample_rate))
    n = min(len(left), len(right))
    num_wins = n // win
    if num_wins == 0:
        return np.array([]), np.array([])

    # Reshape into windows
    l_wins = left[:num_wins * win].reshape(num_wins, win)
    r_wins = right[:num_wins * win].reshape(num_wins, win)

    # Vectorized Pearson r per window
    l_mean = np.mean(l_wins, axis=1, keepdims=True)
    r_mean = np.mean(r_wins, axis=1, keepdims=True)
    l_centered = l_wins - l_mean
    r_centered = r_wins - r_mean

    numerator = np.sum(l_centered * r_centered, axis=1)
    denom = np.sqrt(np.sum(l_centered * l_centered, axis=1) *
                    np.sum(r_centered * r_centered, axis=1))

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_vals = np.where(denom > 0, numerator / denom, 0.0)

    times = np.linspace(0, num_wins * win / sample_rate, num_wins)
    return corr_vals, times


def compute_overall_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Single scalar: Pearson r between entire left and right arrays."""
    if len(left) == 0 or len(right) == 0:
        return 0.0
    min_len = min(len(left), len(right))
    l, r = left[:min_len], right[:min_len]
    l_centered = l - np.mean(l)
    r_centered = r - np.mean(r)
    denom = np.sqrt(np.sum(l_centered ** 2) * np.sum(r_centered ** 2))
    if denom == 0:
        return 0.0
    return float(np.dot(l_centered, r_centered) / denom)


# ──────────────────────────────────────────────────────────────────────────────
#  FFT for Spectrum Plot
# ──────────────────────────────────────────────────────────────────────────────

# Cache Hann window to avoid reallocation
_hann_cache: dict[int, np.ndarray] = {}

def _get_hann_window(size: int) -> np.ndarray:
    if size not in _hann_cache:
        _hann_cache[size] = np.hanning(size)
    return _hann_cache[size]

# Cache rfftfreq arrays
_rfftfreq_cache: dict[tuple[int, int], np.ndarray] = {}

def _get_rfftfreq(n: int, d: float) -> np.ndarray:
    key = (n, int(1.0 / d) if d > 0 else 0)
    if key not in _rfftfreq_cache:
        _rfftfreq_cache[key] = np.fft.rfftfreq(n, d)
    return _rfftfreq_cache[key]


def compute_fft_spectrum(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (frequencies, magnitude_dB) for the middle chunk of *samples*.
    Uses cached window and rfftfreq for reduced allocation.
    """
    start = max(0, len(samples) // 2 - fft_size // 2)
    chunk = samples[start:start + fft_size]
    if len(chunk) < fft_size:
        chunk = np.pad(chunk, (0, fft_size - len(chunk)))

    window = _get_hann_window(len(chunk))
    windowed = chunk * window
    fft = np.fft.rfft(windowed)
    mag = np.abs(fft) / len(chunk)
    mag_db = 20.0 * np.log10(mag + 1e-10)
    freqs = _get_rfftfreq(len(chunk), 1.0 / sample_rate)
    return freqs, mag_db


# ──────────────────────────────────────────────────────────────────────────────
#  Spectrogram
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequencies, times, Sxx_dB)."""
    f, t, Sxx = scipy_signal.spectrogram(
        samples, fs=sample_rate, window="hann",
        nperseg=nperseg, noverlap=nperseg // 2, scaling="density",
    )
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


# ──────────────────────────────────────────────────────────────────────────────
#  Min/Max Waveform Envelope (for accurate visual downsampling)
# ──────────────────────────────────────────────────────────────────────────────

def compute_waveform_envelope(
    samples: np.ndarray,
    max_points: int = 60_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample waveform using min/max per bucket.
    Returns (time, min_vals, max_vals) — each of length <= max_points * 2.
    Preserves peaks much better than linear subsampling.
    """
    n = len(samples)
    if n <= max_points:
        time = np.linspace(0, 1, n)     # caller will scale by duration
        return time, samples, samples

    bucket_size = max(1, n // max_points)
    num_buckets = n // bucket_size

    trimmed = samples[:num_buckets * bucket_size].reshape(num_buckets, bucket_size)
    min_vals = np.min(trimmed, axis=1)
    max_vals = np.max(trimmed, axis=1)

    # Interleave min/max for proper rendering
    time = np.linspace(0, 1, num_buckets)
    return time, min_vals, max_vals
