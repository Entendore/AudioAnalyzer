#!/usr/bin/env python3
"""
Audio Analyzer — Enhanced PySide6 application for visualizing and analyzing audio files.
Features: Waveform (click-to-seek), FFT Spectrum, Spectrogram, RMS History, 
          Level Meters, Advanced Metrics, Band Energy, Playback, and Export tools.
"""

import sys
import os
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QSlider, QGroupBox, QFormLayout,
    QTabWidget, QStatusBar, QSplitter, QComboBox,
    QMessageBox, QCheckBox, QSpinBox, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QUrl, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QLinearGradient, QPen, QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal as scipy_signal
from scipy.io import wavfile

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

# ──────────────────────────────────────────────────────────────────────────────
#  Theme Constants
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1e1e2e"
DARK_AXIS = "#181825"
LIGHT_TXT = "#cdd6f4"
GRID_CLR  = "#585b70"
BLUE      = "#89b4fa"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
MAUVE     = "#cba6f7"
PEACH     = "#fab387"
TEAL      = "#94e2d5"

# ──────────────────────────────────────────────────────────────────────────────
#  Audio Data Model
# ──────────────────────────────────────────────────────────────────────────────

class AudioData:
    """Container for audio data and computed statistics."""

    def __init__(self):
        self.samples = None
        self.sample_rate: int = 0
        self.channels: int = 0
        self.duration: float = 0.0
        self.bit_depth: int = 0
        self.file_path: str = ""
        self.file_name: str = ""
        
        # Basic stats
        self.rms: float = 0.0
        self.peak: float = 0.0
        self.rms_db: float = -np.inf
        self.peak_db: float = -np.inf
        self.crest_factor: float = 0.0
        self.dynamic_range: float = 0.0
        
        # Advanced stats
        self.spectral_centroid: float = 0.0
        self.spectral_rolloff: float = 0.0
        self.zero_crossing_rate: float = 0.0
        
        # History
        self.rms_history: np.ndarray = np.array([])
        self.rms_history_times: np.ndarray = np.array([])
        
        # Band energy
        self.band_energy: dict = {}

    def load_file(self, file_path: str) -> bool:
        self.file_path = file_path
        self.file_name = Path(file_path).name

        if HAS_SOUNDFILE:
            self.samples, self.sample_rate = sf.read(file_path, always_2d=True)
            self.channels = self.samples.shape[1]
            subtype = sf.info(file_path).subtype
            self.bit_depth = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}.get(subtype, 16)
        else:
            self.sample_rate, self.samples = wavfile.read(file_path)
            if self.samples.ndim == 1:
                self.samples = self.samples.reshape(-1, 1)
            self.channels = self.samples.shape[1]
            self.bit_depth = self.samples.dtype.itemsize * 8

        if not np.issubdtype(self.samples.dtype, np.floating):
            mx = np.max(np.abs(self.samples))
            if mx > 0:
                self.samples = self.samples.astype(np.float64) / mx

        self.duration = len(self.samples) / self.sample_rate
        self._compute_stats()
        return True

    def _compute_stats(self):
        mono = self.get_mono()
        
        # Basic
        self.rms = float(np.sqrt(np.mean(mono ** 2)))
        self.peak = float(np.max(np.abs(mono)))
        self.rms_db = 20 * np.log10(self.rms) if self.rms > 0 else -np.inf
        self.peak_db = 20 * np.log10(self.peak) if self.peak > 0 else -np.inf
        self.crest_factor = self.peak_db - self.rms_db if self.rms_db > -np.inf else 0.0

        abs_s = np.abs(mono)
        nonzero = abs_s[abs_s > 0]
        if len(nonzero) > 0:
            p95, p5 = np.percentile(nonzero, 95), np.percentile(nonzero, 5)
            self.dynamic_range = 20 * np.log10(p95 / p5) if p5 > 0 else 0.0
        else:
            self.dynamic_range = 0.0

        # RMS History (50ms windows)
        win_size = int(0.05 * self.sample_rate)
        if len(mono) > win_size:
            num_wins = len(mono) // win_size
            self.rms_history = np.array([
                np.sqrt(np.mean(mono[i*win_size:(i+1)*win_size]**2)) 
                for i in range(num_wins)
            ])
            self.rms_history_times = np.linspace(0, self.duration, num_wins)
        
        # Advanced Stats
        freqs, Pxx = scipy_signal.welch(mono, fs=self.sample_rate, nperseg=4096)
        pxx_sum = np.sum(Pxx)
        if pxx_sum > 0:
            self.spectral_centroid = float(np.sum(freqs * Pxx) / pxx_sum)
            cumsum = np.cumsum(Pxx)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            self.spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs)-1)])
            
            # Band Energy
            def band_e(f_low, f_high):
                mask = (freqs >= f_low) & (freqs < f_high)
                return float(np.sum(Pxx[mask]) / pxx_sum)
            
            self.band_energy = {
                "Sub Bass\n(20-60Hz)": band_e(20, 60),
                "Bass\n(60-250Hz)": band_e(60, 250),
                "Low Mid\n(250-500Hz)": band_e(250, 500),
                "Mid\n(500-2kHz)": band_e(500, 2000),
                "High Mid\n(2k-4kHz)": band_e(2000, 4000),
                "Presence\n(4k-6kHz)": band_e(4000, 6000),
                "Brilliance\n(6k-20kHz)": band_e(6000, 20000),
            }

        # Zero Crossing Rate
        if len(mono) > 1:
            signs = np.sign(mono)
            self.zero_crossing_rate = float(np.sum(np.abs(np.diff(signs)) > 0) / (2 * len(mono)) * self.sample_rate)

    def get_mono(self) -> np.ndarray:
        if self.samples is None: return np.array([])
        return np.mean(self.samples, axis=1) if self.channels > 1 else self.samples[:, 0]

    def get_channel(self, idx: int = 0) -> np.ndarray:
        if self.samples is None: return np.array([])
        return self.samples[:, min(idx, self.channels - 1)]


# ──────────────────────────────────────────────────────────────────────────────
#  Matplotlib Canvases
# ──────────────────────────────────────────────────────────────────────────────

class _BaseCanvas(FigureCanvas):
    seek_requested = Signal(float)

    def __init__(self, parent=None, height_ratio=3):
        self.fig = Figure(figsize=(10, height_ratio), facecolor=DARK_BG)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cbar = None
        self._playback_line = None

    def _style_ax(self, xlabel="", ylabel="", title=""):
        self.ax.set_facecolor(DARK_AXIS)
        self.ax.tick_params(colors=LIGHT_TXT, labelsize=8)
        for spine in self.ax.spines.values(): spine.set_color(GRID_CLR)
        if xlabel: self.ax.set_xlabel(xlabel, color=LIGHT_TXT, fontsize=9)
        if ylabel: self.ax.set_ylabel(ylabel, color=LIGHT_TXT, fontsize=9)
        if title: self.ax.set_title(title, color=LIGHT_TXT, fontsize=11, fontweight="bold")
        self.ax.grid(True, alpha=0.2, color=GRID_CLR)

    def set_playback_pos(self, time_sec: float, duration: float):
        if self._playback_line is not None:
            try: self._playback_line.remove()
            except Exception: pass
        if 0 <= time_sec <= duration:
            self._playback_line = self.ax.axvline(time_sec, color=RED, linewidth=1.4, alpha=0.85)
        self.draw_idle()


class WaveformCanvas(_BaseCanvas):
    def __init__(self, parent=None):
        super().__init__(parent, height_ratio=3)
        self.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.seek_requested.emit(event.xdata)

    def plot(self, audio: AudioData, channel: int = 0):
        self.ax.clear()
        self._style_ax("Time (s)", "Amplitude", "Waveform (Click to Seek)")
        if audio is None or audio.samples is None: self.draw(); return

        samples = audio.get_channel(channel)
        time = np.linspace(0, audio.duration, len(samples))

        max_pts = 60_000
        if len(samples) > max_pts:
            idx = np.linspace(0, len(samples) - 1, max_pts).astype(int)
            time, samples = time[idx], samples[idx]

        self.ax.plot(time, samples, color=BLUE, linewidth=0.5, alpha=0.85)
        self.ax.fill_between(time, samples, alpha=0.12, color=BLUE)
        self.ax.set_xlim(0, audio.duration)
        self.ax.set_ylim(-1.05, 1.05)

        ax2 = self.ax.twinx()
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_yticks([-1, -0.707, 0, 0.707, 1])
        ax2.set_yticklabels(["0 dB", "-3 dB", "-∞", "-3 dB", "0 dB"])
        ax2.tick_params(colors="#a6adc8", labelsize=7)
        for spine in ax2.spines.values(): spine.set_color(GRID_CLR)

        self.fig.tight_layout()
        self._playback_line = None
        self.draw()


class SpectrumCanvas(_BaseCanvas):
    def plot(self, audio: AudioData, fft_size: int = 4096, channel: int = 0):
        self.ax.clear()
        self._style_ax("Frequency (Hz)", "Magnitude (dB)", "Frequency Spectrum")
        if audio is None or audio.samples is None: self.draw(); return

        samples = audio.get_channel(channel)
        start = max(0, len(samples) // 2 - fft_size // 2)
        chunk = samples[start:start + fft_size]
        if len(chunk) < fft_size: chunk = np.pad(chunk, (0, fft_size - len(chunk)))

        windowed = chunk * np.hanning(len(chunk))
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft) / len(chunk)
        mag_db = 20 * np.log10(mag + 1e-10)
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / audio.sample_rate)

        self.ax.plot(freqs, mag_db, color=GREEN, linewidth=0.8)
        self.ax.fill_between(freqs, mag_db, np.min(mag_db), alpha=0.12, color=GREEN)
        self.ax.set_xlim(20, audio.sample_rate / 2)
        self.ax.set_xscale("log")
        self.ax.set_ylim(max(-120, np.min(mag_db)), np.max(mag_db) + 5)

        for f in (100, 200, 500, 1000, 2000, 5000, 10000, 20000):
            if f <= audio.sample_rate / 2:
                self.ax.axvline(f, color=GRID_CLR, lw=0.5, ls="--", alpha=0.5)

        self.fig.tight_layout()
        self.draw()


class SpectrogramCanvas(_BaseCanvas):
    def plot(self, audio: AudioData, nperseg: int = 1024, channel: int = 0, log_freq: bool = True):
        self.ax.clear()
        self._style_ax("Time (s)", "Frequency (Hz)", "Spectrogram")
        if audio is None or audio.samples is None: self.draw(); return

        samples = audio.get_channel(channel)
        max_len = 60 * audio.sample_rate
        if len(samples) > max_len: samples = samples[:max_len]

        f, t, Sxx = scipy_signal.spectrogram(
            samples, fs=audio.sample_rate, window="hann",
            nperseg=nperseg, noverlap=nperseg // 2, scaling="density"
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmin, vmax = np.percentile(Sxx_db, 10), np.percentile(Sxx_db, 99)

        im = self.ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        
        if self._cbar:
            try: self._cbar.remove()
            except Exception: pass
        self._cbar = self.fig.colorbar(im, ax=self.ax, pad=0.02)
        self._cbar.set_label("Power (dB)", color=LIGHT_TXT, fontsize=8)
        self._cbar.ax.tick_params(colors=LIGHT_TXT, labelsize=7)

        if log_freq:
            self.ax.set_yscale("log")
            self.ax.set_ylim(20, audio.sample_rate / 2)
        else:
            self.ax.set_ylim(0, min(audio.sample_rate / 2, 22050))

        self.fig.tight_layout()
        self.draw()


class RMSHistoryCanvas(_BaseCanvas):
    def plot(self, audio: AudioData):
        self.ax.clear()
        self._style_ax("Time (s)", "RMS Level (dB)", "Loudness History (RMS)")
        if audio is None or audio.samples is None or len(audio.rms_history) == 0:
            self.draw(); return

        rms_db = 20 * np.log10(audio.rms_history + 1e-10)
        self.ax.plot(audio.rms_history_times, rms_db, color=MAUVE, linewidth=1.0)
        self.ax.fill_between(audio.rms_history_times, rms_db, -120, alpha=0.15, color=MAUVE)
        
        self.ax.set_xlim(0, audio.duration)
        self.ax.set_ylim(max(-80, np.min(rms_db) - 5), max(rms_db) + 5)
        
        # Reference lines
        self.ax.axhline(-14, color=GREEN, ls="--", alpha=0.5, label="Target (-14 LUFS approx)")
        self.ax.axhline(-1, color=RED, ls="--", alpha=0.5, label="0 dBFS Peak approx")
        self.ax.legend(loc='lower right', fontsize=7, facecolor=DARK_AXIS, edgecolor=GRID_CLR, labelcolor=LIGHT_TXT)

        self.fig.tight_layout()
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
#  Custom Widgets
# ──────────────────────────────────────────────────────────────────────────────

class LevelMeter(QWidget):
    def __init__(self, label: str = "L", parent=None):
        super().__init__(parent)
        self._label = label
        self._peak = 0.0; self._rms = 0.0
        self.setFixedSize(50, 220)

    def set_levels(self, peak: float, rms: float):
        self._peak = min(peak, 1.0); self._rms = min(rms, 1.0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#313244"))
        p.setPen(QPen(QColor(GRID_CLR), 1)); p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        m = 6; h = self.height() - 2 * m; w = self.width() - 2 * m
        if self._rms > 0:
            bar_h = int(h * self._rms)
            grad = QLinearGradient(0, self.height() - m, 0, m)
            grad.setColorAt(0.0, QColor(GREEN)); grad.setColorAt(0.75, QColor(YELLOW)); grad.setColorAt(1.0, QColor(RED))
            p.fillRect(m, self.height() - m - bar_h, w, bar_h, grad)
        if self._peak > 0:
            py = self.height() - m - int(h * self._peak)
            p.setPen(QPen(QColor(RED), 2)); p.drawLine(m, py, m + w, py)
        p.setPen(QColor(LIGHT_TXT)); p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignBottom, self._label)
        p.end()


class BandEnergyWidget(QWidget):
    """Horizontal bar chart for frequency band energy."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = {}
        self.setMinimumHeight(180)

    def set_data(self, data: dict):
        self._data = data
        self.update()

    def paintEvent(self, event):
        if not self._data: return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        
        n = len(self._data)
        h = self.height(); w = self.width()
        m_left = 80; m_right = 40; m_top = 10; m_bot = 10
        
        draw_w = w - m_left - m_right
        draw_h = h - m_top - m_bot
        bar_h = draw_h / n * 0.7
        spacing = draw_h / n * 0.3
        
        p.setFont(QFont("Segoe UI", 7))
        
        colors = [RED, PEACH, YELLOW, GREEN, TEAL, BLUE, MAUVE]
        
        for i, (label, val) in enumerate(self._data.items()):
            y = m_top + i * (bar_h + spacing)
            bar_w = int(draw_w * val)
            
            # Label
            p.setPen(QColor(LIGHT_TXT))
            p.drawText(0, y, m_left - 5, int(bar_h), Qt.AlignRight | Qt.AlignVCenter, label)
            
            # Bar
            c = colors[i % len(colors)]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c))
            p.drawRoundedRect(m_left, y, bar_w, int(bar_h), 3, 3)
            
            # Percentage text
            p.setPen(QColor(LIGHT_TXT))
            p.drawText(m_left + bar_w + 5, y, m_right - 5, int(bar_h), Qt.AlignLeft | Qt.AlignVCenter, f"{val*100:.1f}%")
        
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio: AudioData | None = None
        self._current_channel: int = 0
        self._setup_ui()
        self._setup_player()
        self._setup_timers()
        self._apply_style()
        self.setWindowTitle("🎵 Advanced Audio Analyzer")
        self.setMinimumSize(1150, 800)
        self.resize(1440, 920)
        self.setAcceptDrops(True)
        self.statusBar().showMessage("Drag & drop an audio file or click Open Audio to begin.")

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path): self._load_file(path); break

    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(8,8,8,8); root.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal); root.addWidget(splitter)

        left = QWidget(); left_lay = QVBoxLayout(left); left_lay.setContentsMargins(0,0,0,0); left_lay.setSpacing(4)
        left_lay.addWidget(self._build_toolbar())

        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)

        def add_tab(canvas, name):
            w = QWidget(); wl = QVBoxLayout(w); wl.setContentsMargins(0,0,0,0); wl.addWidget(canvas); self.tabs.addTab(w, name)

        self.waveform_canvas = WaveformCanvas(); add_tab(self.waveform_canvas, "📊 Waveform")
        self.spectrum_canvas = SpectrumCanvas(); add_tab(self.spectrum_canvas, "📈 Spectrum")
        self.spectrogram_canvas = SpectrogramCanvas(); add_tab(self.spectrogram_canvas, "🔥 Spectrogram")
        self.rms_canvas = RMSHistoryCanvas(); add_tab(self.rms_canvas, "🎚️ Loudness")

        left_lay.addWidget(self.tabs, 1)
        left_lay.addWidget(self._build_transport())
        splitter.addWidget(left)

        # Right Panel Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(310); scroll.setMinimumWidth(250)
        
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(8)

        # File info
        grp = QGroupBox("📋 File Information"); fl = QFormLayout(grp); fl.setSpacing(5)
        self.lbl_file   = self._info_label(BLUE, bold=True, wrap=True)
        self.lbl_dur    = self._info_label(); self.lbl_sr = self._info_label()
        self.lbl_ch     = self._info_label(); self.lbl_bits = self._info_label()
        self.lbl_fsize  = self._info_label()
        fl.addRow("File:", self.lbl_file); fl.addRow("Duration:", self.lbl_dur)
        fl.addRow("Sample Rate:", self.lbl_sr); fl.addRow("Channels:", self.lbl_ch)
        fl.addRow("Bit Depth:", self.lbl_bits); fl.addRow("File Size:", self.lbl_fsize)
        rl.addWidget(grp)

        # Level meters
        grp = QGroupBox("📊 Basic Levels"); vl = QVBoxLayout(grp)
        meters = QHBoxLayout()
        self.meter_l = LevelMeter("L"); self.meter_r = LevelMeter("R")
        meters.addWidget(self.meter_l); meters.addWidget(self.meter_r)
        vl.addLayout(meters)
        fl2 = QFormLayout(); fl2.setSpacing(4)
        self.lbl_peak = self._info_label(); self.lbl_rms = self._info_label()
        self.lbl_crest = self._info_label(); self.lbl_dr = self._info_label()
        fl2.addRow("Peak:", self.lbl_peak); fl2.addRow("RMS:", self.lbl_rms)
        fl2.addRow("Crest Factor:", self.lbl_crest); fl2.addRow("Dynamic Range:", self.lbl_dr)
        vl.addLayout(fl2); rl.addWidget(grp)

        # Advanced Metrics
        grp = QGroupBox("🔬 Advanced Metrics"); fl3 = QFormLayout(grp); fl3.setSpacing(5)
        self.lbl_centroid = self._info_label(PEACH)
        self.lbl_rolloff = self._info_label(YELLOW)
        self.lbl_zcr = self._info_label(TEAL)
        fl3.addRow("Spectral Centroid:", self.lbl_centroid)
        fl3.addRow("Spectral Rolloff:", self.lbl_rolloff)
        fl3.addRow("Zero Crossing Rate:", self.lbl_zcr)
        rl.addWidget(grp)

        # Band Energy
        grp = QGroupBox("🎵 Band Energy"); vl2 = QVBoxLayout(grp)
        self.band_energy_widget = BandEnergyWidget()
        vl2.addWidget(self.band_energy_widget)
        rl.addWidget(grp)

        # Channel & Settings
        grp = QGroupBox("⚙️ Settings"); fl4 = QFormLayout(grp); fl4.setSpacing(6)
        self.combo_ch = QComboBox(); self.combo_ch.addItem("Mono")
        self.combo_ch.currentIndexChanged.connect(self._on_channel)
        fl4.addRow("Channel:", self.combo_ch)
        self.spin_fft = QSpinBox(); self.spin_fft.setRange(256, 16384); self.spin_fft.setValue(4096); self.spin_fft.setSingleStep(512); self.spin_fft.setSuffix(" smpl")
        fl4.addRow("FFT Size:", self.spin_fft)
        self.spin_nperseg = QSpinBox(); self.spin_nperseg.setRange(128, 8192); self.spin_nperseg.setValue(1024); self.spin_nperseg.setSingleStep(128); self.spin_nperseg.setSuffix(" smpl")
        fl4.addRow("Spectr. Win:", self.spin_nperseg)
        self.chk_log = QCheckBox("Log Frequency"); self.chk_log.setChecked(True)
        fl4.addRow(self.chk_log)
        self.btn_refresh = QPushButton("🔄 Refresh Analysis"); self.btn_refresh.setObjectName("accentBtn"); self.btn_refresh.clicked.connect(self._refresh)
        fl4.addRow(self.btn_refresh)
        rl.addWidget(grp)

        rl.addStretch()
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([1000, 300])

    @staticmethod
    def _info_label(color=LIGHT_TXT, bold=False, wrap=False):
        lbl = QLabel("—"); lbl.setStyleSheet(f"color:{color};{'font-weight:bold;' if bold else ''}")
        if wrap: lbl.setWordWrap(True)
        return lbl

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(); bar.setObjectName("toolBar"); lay = QHBoxLayout(bar); lay.setContentsMargins(10,6,10,6)
        btn_open = QPushButton("📂  Open Audio"); btn_open.setObjectName("accentBtn"); btn_open.clicked.connect(self._open_file)
        lay.addWidget(btn_open); lay.addSpacing(15)
        
        self.btn_play  = QPushButton("▶"); self.btn_play.clicked.connect(self._play); self.btn_play.setEnabled(False)
        self.btn_pause = QPushButton("⏸"); self.btn_pause.clicked.connect(self._pause); self.btn_pause.setEnabled(False)
        self.btn_stop  = QPushButton("⏹"); self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
        
        lay.addWidget(self.btn_play); lay.addWidget(self.btn_pause); lay.addWidget(self.btn_stop); lay.addSpacing(15)
        
        btn_export_img = QPushButton("🖼️ Export Plot"); btn_export_img.clicked.connect(self._export_plot)
        btn_export_rpt = QPushButton("📝 Export Report"); btn_export_rpt.clicked.connect(self._export_report)
        lay.addWidget(btn_export_img); lay.addWidget(btn_export_rpt)
        
        lay.addStretch(); return bar

    def _build_transport(self) -> QWidget:
        bar = QWidget(); bar.setObjectName("transportBar"); lay = QHBoxLayout(bar); lay.setContentsMargins(10,5,10,5)
        self.lbl_time = QLabel("0:00.00"); self.lbl_time.setStyleSheet(f"color:{LIGHT_TXT};font-family:monospace;font-size:12px;")
        lay.addWidget(self.lbl_time)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0,1000); self.slider.setValue(0); self.slider.sliderMoved.connect(self._on_slider)
        lay.addWidget(self.slider, 1)
        self.lbl_total = QLabel("0:00.00"); self.lbl_total.setStyleSheet(f"color:{LIGHT_TXT};font-family:monospace;font-size:12px;")
        lay.addWidget(self.lbl_total)
        lay.addSpacing(14); lay.addWidget(QLabel("🔊"))
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0,100); self.vol_slider.setValue(80); self.vol_slider.setMaximumWidth(100); self.vol_slider.valueChanged.connect(self._on_volume)
        lay.addWidget(self.vol_slider); return bar

    def _setup_player(self):
        self.player = QMediaPlayer(); self.output = QAudioOutput(); self.output.setVolume(0.8)
        self.player.setAudioOutput(self.output)
        self.player.positionChanged.connect(self._on_position)
        self.player.playbackStateChanged.connect(self._on_state)
        self.waveform_canvas.seek_requested.connect(self._seek_to)

    def _setup_timers(self):
        self.meter_timer = QTimer(); self.meter_timer.setInterval(50); self.meter_timer.timeout.connect(self._tick_meters)

    # ── actions ───────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.au);;All Files (*)")
        if path: self._load_file(path)

    def _load_file(self, path: str):
        try:
            self._stop(); self.audio = AudioData(); self.audio.load_file(path)
            self._update_info(); self._update_channels(); self._refresh()
            self.btn_play.setEnabled(True); self.btn_pause.setEnabled(True); self.btn_stop.setEnabled(True)
            self.player.setSource(QUrl.fromLocalFile(path))
            self.statusBar().showMessage(f"Loaded: {self.audio.file_name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{exc}")

    def _update_info(self):
        a = self.audio
        if a is None: return
        self.lbl_file.setText(a.file_name)
        m, s = int(a.duration)//60, a.duration%60; self.lbl_dur.setText(f"{m}:{s:05.2f}"); self.lbl_total.setText(f"{m}:{s:05.2f}")
        self.lbl_sr.setText(f"{a.sample_rate:,} Hz"); self.lbl_ch.setText(str(a.channels)); self.lbl_bits.setText(f"{a.bit_depth} bit")
        sz = os.path.getsize(a.file_path); self.lbl_fsize.setText(f"{sz/(1<<20):.1f} MB" if sz > 1<<20 else f"{sz/1024:.1f} KB")
        self.lbl_peak.setText(f"{a.peak_db:.1f} dB"); self.lbl_rms.setText(f"{a.rms_db:.1f} dB")
        self.lbl_crest.setText(f"{a.crest_factor:.1f} dB"); self.lbl_dr.setText(f"{a.dynamic_range:.1f} dB")
        
        # Advanced
        self.lbl_centroid.setText(f"{a.spectral_centroid:.0f} Hz")
        self.lbl_rolloff.setText(f"{a.spectral_rolloff:.0f} Hz")
        self.lbl_zcr.setText(f"{a.zero_crossing_rate:.0f} Hz")
        
        # Band Energy
        self.band_energy_widget.set_data(a.band_energy)

        # Meters
        self.meter_l.set_levels(a.peak, a.rms)
        if a.channels > 1:
            ch1 = a.get_channel(1); self.meter_r.set_levels(float(np.max(np.abs(ch1))), float(np.sqrt(np.mean(ch1**2))))
        else: self.meter_r.set_levels(a.peak, a.rms)

    def _update_channels(self):
        self.combo_ch.blockSignals(True); self.combo_ch.clear()
        if self.audio is None: self.combo_ch.addItem("Mono")
        elif self.audio.channels == 1: self.combo_ch.addItem("Mono")
        elif self.audio.channels == 2: self.combo_ch.addItems(["Left", "Right", "Mono Mix"])
        else:
            for i in range(self.audio.channels): self.combo_ch.addItem(f"Channel {i+1}")
            self.combo_ch.addItem("Mono Mix")
        self.combo_ch.blockSignals(False); self._current_channel = 0

    def _on_channel(self, idx):
        if self.audio is None: return
        self._current_channel = min(idx, self.audio.channels - 1) if idx < self.combo_ch.count() - 1 else 0
        self._refresh()

    def _refresh(self):
        if self.audio is None: return
        ch = self._current_channel if self._current_channel >= 0 else 0
        self.waveform_canvas.plot(self.audio, channel=ch)
        self.spectrum_canvas.plot(self.audio, fft_size=self.spin_fft.value(), channel=ch)
        self.spectrogram_canvas.plot(self.audio, nperseg=self.spin_nperseg.value(), channel=ch, log_freq=self.chk_log.isChecked())
        self.rms_canvas.plot(self.audio)

    # ── playback ──────────────────────────────────────────────────────────

    def _play(self): self.player.play(); self.meter_timer.start()
    def _pause(self): self.player.pause(); self.meter_timer.stop()
    def _stop(self): 
        self.player.stop(); self.meter_timer.stop(); self.slider.setValue(0); self.lbl_time.setText("0:00.00")
        self.waveform_canvas.set_playback_pos(-1, 1)
        if self.audio:
            self.meter_l.set_levels(self.audio.peak, self.audio.rms)
            if self.audio.channels > 1:
                ch1 = self.audio.get_channel(1); self.meter_r.set_levels(float(np.max(np.abs(ch1))), float(np.sqrt(np.mean(ch1**2))))
            else: self.meter_r.set_levels(self.audio.peak, self.audio.rms)

    def _seek_to(self, seconds: float):
        self.player.setPosition(int(seconds * 1000))
        if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self._play()

    def _on_position(self, ms):
        dur = self.player.duration()
        if dur > 0: self.slider.blockSignals(True); self.slider.setValue(int(ms/dur*1000)); self.slider.blockSignals(False)
        s = ms/1000; self.lbl_time.setText(f"{int(s)//60}:{s%60:05.2f}")
        if self.audio: self.waveform_canvas.set_playback_pos(s, self.audio.duration)

    def _on_state(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState: self.meter_timer.stop(); self.waveform_canvas.set_playback_pos(-1, 1)

    def _on_slider(self, v): 
        dur = self.player.duration()
        if dur > 0: self.player.setPosition(int(v/1000*dur))

    def _on_volume(self, v): self.output.setVolume(v/100.0)

    def _tick_meters(self):
        if self.audio is None: return
        pos_s = self.player.position()/1000.0; idx = int(pos_s * self.audio.sample_rate); win = int(0.05 * self.audio.sample_rate)
        if idx + win < len(self.audio.samples):
            ch0 = self.audio.get_channel(0); w0 = ch0[idx:idx+win]
            pk, rms = float(np.max(np.abs(w0))), float(np.sqrt(np.mean(w0**2))); self.meter_l.set_levels(pk, rms)
            if self.audio.channels > 1:
                ch1 = self.audio.get_channel(1); w1 = ch1[idx:idx+win]; self.meter_r.set_levels(float(np.max(np.abs(w1))), float(np.sqrt(np.mean(w1**2))))
            else: self.meter_r.set_levels(pk, rms)

    # ── export ────────────────────────────────────────────────────────────

    def _export_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", f"{self.audio.file_name}_plot.png", "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Vector (*.svg);;PDF Document (*.pdf)")
        if path:
            current_canvas = self.tabs.currentWidget().findChild(FigureCanvas)
            if current_canvas:
                current_canvas.fig.savefig(path, dpi=200, facecolor=current_canvas.fig.get_facecolor(), bbox_inches='tight')
                self.statusBar().showMessage(f"Plot exported to {path}", 5000)

    def _export_report(self):
        if not self.audio: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", f"{self.audio.file_name}_report.txt", "Text File (*.txt);;JSON File (*.json)")
        if path:
            a = self.audio
            with open(path, 'w') as f:
                f.write(f"--- Audio Analysis Report ---\n")
                f.write(f"File: {a.file_name}\nPath: {a.file_path}\n\n")
                f.write(f"[Basic Info]\nDuration: {a.duration:.2f}s\nSample Rate: {a.sample_rate} Hz\nChannels: {a.channels}\nBit Depth: {a.bit_depth}\n\n")
                f.write(f"[Levels]\nPeak: {a.peak_db:.2f} dB\nRMS: {a.rms_db:.2f} dB\nCrest Factor: {a.crest_factor:.2f} dB\nDynamic Range: {a.dynamic_range:.2f} dB\n\n")
                f.write(f"[Advanced Metrics]\nSpectral Centroid: {a.spectral_centroid:.2f} Hz\nSpectral Rolloff (85%): {a.spectral_rolloff:.2f} Hz\nZero Crossing Rate: {a.zero_crossing_rate:.2f} Hz\n\n")
                f.write(f"[Band Energy]\n")
                for band, energy in a.band_energy.items():
                    f.write(f"{band.replace(chr(10), ' ')}: {energy*100:.2f}%\n")
            self.statusBar().showMessage(f"Report exported to {path}", 5000)

    # ── stylesheet ────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {DARK_BG}; }}
            QWidget       {{ background-color: {DARK_BG}; color: {LIGHT_TXT}; font-family: 'Segoe UI', sans-serif; }}
            QScrollArea   {{ background-color: {DARK_BG}; border: none; }}
            QGroupBox     {{ background-color: {DARK_AXIS}; border:1px solid #313244; border-radius:8px; margin-top:12px; padding:12px; padding-top:24px; font-weight:bold; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; }}
            QPushButton   {{ background-color:#313244; color:{LIGHT_TXT}; border:1px solid #45475a; border-radius:4px; padding:6px 12px; font-size:12px; }}
            QPushButton:hover   {{ background-color:#45475a; }}
            QPushButton:pressed {{ background-color:{GRID_CLR}; }}
            QPushButton:disabled{{ background-color:{DARK_BG}; color:#585b70; }}
            QPushButton#accentBtn {{ background-color:{BLUE}; color:#1e1e2e; font-weight:bold; border:none; }}
            QPushButton#accentBtn:hover {{ background-color:#b4befe; }}
            QSlider::groove:horizontal {{ border:1px solid #313244; height:6px; background:#313244; border-radius:3px; }}
            QSlider::handle:horizontal {{ background:{BLUE}; border:1px solid {BLUE}; width:14px; margin:-5px 0; border-radius:7px; }}
            QComboBox     {{ background-color:#313244; color:{LIGHT_TXT}; border:1px solid #45475a; border-radius:4px; padding:4px 8px; }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{ background-color:#313244; color:{LIGHT_TXT}; selection-background-color:{BLUE}; }}
            QSpinBox      {{ background-color:#313244; color:{LIGHT_TXT}; border:1px solid #45475a; border-radius:4px; padding:4px; }}
            QTabWidget::pane {{ border:1px solid #313244; border-radius:4px; background-color:{DARK_AXIS}; }}
            QTabBar::tab  {{ background-color:#313244; color:#a6adc8; padding:8px 14px; margin-right:2px; border-top-left-radius:6px; border-top-right-radius:6px; }}
            QTabBar::tab:selected {{ background-color:{DARK_AXIS}; color:{BLUE}; font-weight:bold; }}
            QTabBar::tab:hover {{ background-color:#45475a; }}
            QCheckBox          {{ color:{LIGHT_TXT}; spacing:6px; }}
            QCheckBox::indicator {{ width:16px; height:16px; border-radius:3px; border:1px solid {GRID_CLR}; background-color:#313244; }}
            QCheckBox::indicator:checked {{ background-color:{BLUE}; border-color:{BLUE}; }}
            QLabel {{ color:{LIGHT_TXT}; }}
            QStatusBar {{ background-color:{DARK_AXIS}; color:#a6adc8; }}
            QSplitter::handle {{ background-color:#313244; width:2px; }}
            QWidget#toolBar, QWidget#transportBar {{ background-color:{DARK_AXIS}; border-radius:6px; }}
        """)

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()