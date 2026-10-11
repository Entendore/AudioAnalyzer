#!/usr/bin/env python3
"""
main_window.py — Main application window with professional layout.

Professional layout features:
  - Branded header bar with app title and version
  - Grouped toolbar with visual separators between button groups
  - Large transport bar with monospaced time display, loop controls
  - Right panel with section headers, metric cards, and visual dividers
  - Collapsible-style grouped sections in the side panel
  - Status indicators row (clips, pitch, LUFS) as card widgets
  - Tab bar with accent underline on selected tab
"""

import os
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QSlider, QGroupBox,
    QFormLayout, QTabWidget, QStatusBar, QSplitter,
    QComboBox, QMessageBox, QCheckBox, QSpinBox,
    QSizePolicy, QScrollArea, QDoubleSpinBox, QMenu,
    QMenuBar, QFrame,
)
from PySide6.QtCore import Qt, QUrl, QTimer, Signal
from PySide6.QtGui import (
    QFont, QColor, QAction, QKeySequence, QShortcut,
    QDragEnterEvent, QDropEvent,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from audio_data import AudioData
from worker import AnalysisWorker
from canvases import (
    WaveformCanvas, SpectrumCanvas, SpectrogramCanvas,
    RMSHistoryCanvas, PhaseCorrelationCanvas, LiveSpectrumCanvas,
)
from widgets import (
    LevelMeter, BandEnergyWidget, CorrelationMeter,
    ClipIndicator, PitchDisplay, LUFSMeter,
    MetricCard, SectionHeader,
)
from player import PlaybackController
from exporter import Exporter
from config import (
    DARK_BG, DARK_AXIS, MANTLE, SURFACE0, SURFACE1, LIGHT_TXT, GRID_CLR,
    BLUE, BLUE_DIM, GREEN, RED, YELLOW, MAUVE, PEACH, TEAL, SKY, OVERLAY1,
    APP_NAME, APP_VERSION, SHORTCUTS, SPECTROGRAM_CMAPS, DEFAULT_PLAYBACK_VOL,
    build_stylesheet, AppSettings,
    RADIUS_SM, RADIUS_MD, RADIUS_LG, RADIUS_XL,
    MARGIN_SM, MARGIN_MD, MARGIN_LG, MARGIN_XL,
    FONT_SIZE_XS, FONT_SIZE_SM, FONT_SIZE_MD, FONT_SIZE_LG, FONT_SIZE_XL,
)


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None):
        super().__init__()
        self.audio: AudioData | None = None
        self._current_channel: int = 0
        self._settings = settings or AppSettings()

        self._setup_ui()
        self._setup_player()
        self._setup_timers()
        self._setup_menu()
        self._setup_shortcuts()
        self._apply_style()

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1250, 880)
        self.resize(1540, 980)
        self.setAcceptDrops(True)
        self.statusBar().showMessage(
            "Drag & drop an audio file or click Open Audio to begin."
        )

        # Restore geometry
        geom = self._settings.get("window_geometry")
        if geom:
            self.restoreGeometry(geom)

    # ── drag & drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self._load_file(path)
                break

    # ── UI construction ──────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────
        root.addWidget(self._build_header())

        # ── Main content (splitter) ──────────────────────────────────────
        content = QHBoxLayout()
        content.setContentsMargins(MARGIN_MD, MARGIN_MD, MARGIN_MD, MARGIN_MD)
        content.setSpacing(MARGIN_MD)

        splitter = QSplitter(Qt.Horizontal)
        content.addWidget(splitter)

        # Left: toolbar + tabs + transport
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(MARGIN_SM)
        left_lay.addWidget(self._build_toolbar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        def add_tab(canvas, name, icon=""):
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.addWidget(canvas)
            self.tabs.addTab(w, f"  {icon}  {name}  " if icon else name)

        self.waveform_canvas = WaveformCanvas()
        add_tab(self.waveform_canvas, "Waveform")
        self.spectrum_canvas = SpectrumCanvas()
        add_tab(self.spectrum_canvas, "Spectrum")
        self.spectrogram_canvas = SpectrogramCanvas()
        add_tab(self.spectrogram_canvas, "Spectrogram")
        self.rms_canvas = RMSHistoryCanvas()
        add_tab(self.rms_canvas, "Loudness")
        self.phase_canvas = PhaseCorrelationCanvas()
        add_tab(self.phase_canvas, "Phase / Stereo")
        self.live_canvas = LiveSpectrumCanvas()
        add_tab(self.live_canvas, "Live Spectrum")

        left_lay.addWidget(self.tabs, 1)
        left_lay.addWidget(self._build_transport())
        splitter.addWidget(left)

        # ── Right: scrollable info panel ─────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(320)
        scroll.setMinimumWidth(270)

        right = QWidget()
        right.setObjectName("rightPanel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(MARGIN_SM, MARGIN_MD, MARGIN_SM, MARGIN_MD)
        rl.setSpacing(MARGIN_SM)

        # ── File Information ─────────────────────────────────────────────
        rl.addWidget(SectionHeader("FILE INFORMATION"))
        grp = QGroupBox("")
        fl = QFormLayout(grp)
        fl.setSpacing(4)
        fl.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)

        self.lbl_file = self._info_label(BLUE, bold=True, wrap=True)
        self.lbl_dur = self._info_label()
        self.lbl_sr = self._info_label()
        self.lbl_ch = self._info_label()
        self.lbl_bits = self._info_label()
        self.lbl_fsize = self._info_label()
        fl.addRow(self._form_label("File:"), self.lbl_file)
        fl.addRow(self._form_label("Duration:"), self.lbl_dur)
        fl.addRow(self._form_label("Sample Rate:"), self.lbl_sr)
        fl.addRow(self._form_label("Channels:"), self.lbl_ch)
        fl.addRow(self._form_label("Bit Depth:"), self.lbl_bits)
        fl.addRow(self._form_label("File Size:"), self.lbl_fsize)
        rl.addWidget(grp)

        # ── Levels & Loudness ────────────────────────────────────────────
        rl.addWidget(SectionHeader("LEVELS & LOUDNESS"))
        grp = QGroupBox("")
        vl = QVBoxLayout(grp)
        vl.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)
        vl.setSpacing(MARGIN_MD)

        # Level meters side by side
        meters = QHBoxLayout()
        meters.setSpacing(MARGIN_LG)
        self.meter_l = LevelMeter("L")
        self.meter_r = LevelMeter("R")
        meters.addWidget(self.meter_l)
        meters.addWidget(self.meter_r)

        # Basic level metrics as cards
        levels_row = QHBoxLayout()
        levels_row.setSpacing(MARGIN_SM)
        self.card_peak = MetricCard("Peak", RED)
        self.card_rms = MetricCard("RMS", GREEN)
        self.card_crest = MetricCard("Crest", YELLOW)
        self.card_dr = MetricCard("Dyn Range", TEAL)
        for card in (self.card_peak, self.card_rms, self.card_crest, self.card_dr):
            levels_row.addWidget(card)

        vl.addLayout(meters)
        vl.addLayout(levels_row)

        # LUFS bar
        self.lufs_meter = LUFSMeter()
        vl.addWidget(self.lufs_meter)

        # Pitch + Clip indicators
        indicators = QHBoxLayout()
        indicators.setSpacing(MARGIN_SM)
        self.pitch_display = PitchDisplay()
        self.clip_indicator = ClipIndicator()
        indicators.addWidget(self.pitch_display, 1)
        indicators.addWidget(self.clip_indicator, 1)
        vl.addLayout(indicators)

        rl.addWidget(grp)

        # ── Spectral Metrics ─────────────────────────────────────────────
        rl.addWidget(SectionHeader("SPECTRAL METRICS"))
        grp = QGroupBox("")
        fl3 = QFormLayout(grp)
        fl3.setSpacing(4)
        fl3.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)
        self.lbl_centroid = self._info_label(PEACH)
        self.lbl_rolloff = self._info_label(YELLOW)
        self.lbl_flatness = self._info_label(SKY)
        self.lbl_zcr = self._info_label(TEAL)
        fl3.addRow(self._form_label("Spectral Centroid:"), self.lbl_centroid)
        fl3.addRow(self._form_label("Spectral Rolloff:"), self.lbl_rolloff)
        fl3.addRow(self._form_label("Spectral Flatness:"), self.lbl_flatness)
        fl3.addRow(self._form_label("Zero Crossing Rate:"), self.lbl_zcr)
        rl.addWidget(grp)

        # ── Stereo Correlation ───────────────────────────────────────────
        rl.addWidget(SectionHeader("STEREO CORRELATION"))
        grp = QGroupBox("")
        vl_stereo = QVBoxLayout(grp)
        vl_stereo.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)
        self.correlation_meter = CorrelationMeter()
        vl_stereo.addWidget(self.correlation_meter, alignment=Qt.AlignCenter)
        rl.addWidget(grp)

        # ── Band Energy ──────────────────────────────────────────────────
        rl.addWidget(SectionHeader("BAND ENERGY"))
        grp = QGroupBox("")
        vl2 = QVBoxLayout(grp)
        vl2.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)
        self.band_energy_widget = BandEnergyWidget()
        vl2.addWidget(self.band_energy_widget)
        rl.addWidget(grp)

        # ── Settings ─────────────────────────────────────────────────────
        rl.addWidget(SectionHeader("ANALYSIS SETTINGS"))
        grp = QGroupBox("")
        fl4 = QFormLayout(grp)
        fl4.setSpacing(6)
        fl4.setContentsMargins(MARGIN_LG, MARGIN_LG, MARGIN_LG, MARGIN_LG)

        self.combo_ch = QComboBox()
        self.combo_ch.addItem("Mono")
        self.combo_ch.currentIndexChanged.connect(self._on_channel)
        fl4.addRow(self._form_label("Channel:"), self.combo_ch)

        self.spin_fft = QSpinBox()
        self.spin_fft.setRange(256, 16384)
        self.spin_fft.setValue(4096)
        self.spin_fft.setSingleStep(512)
        self.spin_fft.setSuffix(" smpl")
        fl4.addRow(self._form_label("FFT Size:"), self.spin_fft)

        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(128, 8192)
        self.spin_nperseg.setValue(1024)
        self.spin_nperseg.setSingleStep(128)
        self.spin_nperseg.setSuffix(" smpl")
        fl4.addRow(self._form_label("Spectr. Win:"), self.spin_nperseg)

        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(SPECTROGRAM_CMAPS)
        self.combo_cmap.setCurrentText("magma")
        fl4.addRow(self._form_label("Spectr. CMap:"), self.combo_cmap)

        self.chk_log = QCheckBox("Log Frequency")
        self.chk_log.setChecked(True)
        fl4.addRow(self.chk_log)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.25, 4.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSingleStep(0.25)
        self.spin_speed.setSuffix("x")
        self.spin_speed.valueChanged.connect(self._on_speed)
        fl4.addRow(self._form_label("Speed:"), self.spin_speed)

        self.btn_refresh = QPushButton("Refresh Analysis")
        self.btn_refresh.setObjectName("accentBtn")
        self.btn_refresh.clicked.connect(self._refresh)
        fl4.addRow(self.btn_refresh)

        rl.addWidget(grp)
        rl.addStretch()

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([1080, 310])

        root.addLayout(content, 1)

    # ── header bar ───────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerBar")
        header.setFixedHeight(40)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(MARGIN_XL, 0, MARGIN_XL, 0)

        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        lay.addWidget(title)

        version = QLabel(f"v{APP_VERSION}")
        version.setObjectName("appVersion")
        lay.addWidget(version)

        lay.addStretch()

        # Shortcut hints
        hints = QLabel("Space: Play  |  Arrows: Seek  |  Ctrl+O: Open  |  L: Loop")
        hints.setStyleSheet(f"color: {OVERLAY1}; font-size: {FONT_SIZE_XS}px;")
        lay.addWidget(hints)

        return header

    # ── helper: form label ───────────────────────────────────────────────

    @staticmethod
    def _form_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("metricLabel")
        lbl.setStyleSheet(
            f"color: {OVERLAY1}; font-size: {FONT_SIZE_SM}px; border: none; background: transparent;"
        )
        return lbl

    # ── helper: info label ───────────────────────────────────────────────

    @staticmethod
    def _info_label(color=LIGHT_TXT, bold=False, wrap=False):
        lbl = QLabel("--")
        lbl.setStyleSheet(
            f"color:{color};{'font-weight:bold;' if bold else ''}"
            f"font-size: {FONT_SIZE_MD}px; border: none; background: transparent;"
        )
        if wrap:
            lbl.setWordWrap(True)
        return lbl

    # ── toolbar ──────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(MARGIN_SM, MARGIN_SM, MARGIN_SM, MARGIN_SM)
        lay.setSpacing(MARGIN_SM)

        # ── Group 1: File ────────────────────────────────────────────────
        btn_open = QPushButton("Open Audio")
        btn_open.setObjectName("accentBtn")
        btn_open.clicked.connect(self._open_file)
        lay.addWidget(btn_open)

        lay.addWidget(self._toolbar_separator())

        # ── Group 2: Transport ───────────────────────────────────────────
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("transportBtn")
        self.btn_play.clicked.connect(self._play)
        self.btn_play.setEnabled(False)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("transportBtn")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("transportBtn")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        lay.addWidget(self.btn_play)
        lay.addWidget(self.btn_pause)
        lay.addWidget(self.btn_stop)

        lay.addWidget(self._toolbar_separator())

        # ── Group 3: Loop ────────────────────────────────────────────────
        self.btn_loop = QPushButton("Loop")
        self.btn_loop.setObjectName("loopBtn")
        self.btn_loop.setCheckable(True)
        self.btn_loop.clicked.connect(self._on_loop_toggle)
        self.btn_loop.setEnabled(False)
        self.btn_set_a = QPushButton("A")
        self.btn_set_a.setObjectName("transportBtn")
        self.btn_set_a.setMaximumWidth(28)
        self.btn_set_a.clicked.connect(self._on_set_loop_a)
        self.btn_set_a.setEnabled(False)
        self.btn_set_b = QPushButton("B")
        self.btn_set_b.setObjectName("transportBtn")
        self.btn_set_b.setMaximumWidth(28)
        self.btn_set_b.clicked.connect(self._on_set_loop_b)
        self.btn_set_b.setEnabled(False)
        lay.addWidget(self.btn_loop)
        lay.addWidget(self.btn_set_a)
        lay.addWidget(self.btn_set_b)

        lay.addWidget(self._toolbar_separator())

        # ── Group 4: Export ──────────────────────────────────────────────
        btn_export_img = QPushButton("Export Plot")
        btn_export_img.setObjectName("sectionBtn")
        btn_export_img.clicked.connect(self._export_plot)
        btn_export_rpt = QPushButton("Report")
        btn_export_rpt.setObjectName("sectionBtn")
        btn_export_rpt.clicked.connect(self._export_report)
        btn_export_csv = QPushButton("CSV")
        btn_export_csv.setObjectName("sectionBtn")
        btn_export_csv.clicked.connect(self._export_csv)
        btn_export_json = QPushButton("JSON")
        btn_export_json.setObjectName("sectionBtn")
        btn_export_json.clicked.connect(self._export_json)
        btn_export_all = QPushButton("All Tabs")
        btn_export_all.setObjectName("sectionBtn")
        btn_export_all.clicked.connect(self._export_all)
        lay.addWidget(btn_export_img)
        lay.addWidget(btn_export_rpt)
        lay.addWidget(btn_export_csv)
        lay.addWidget(btn_export_json)
        lay.addWidget(btn_export_all)

        lay.addStretch()
        return bar

    @staticmethod
    def _toolbar_separator() -> QWidget:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet(
            f"color: {SURFACE1}; max-width: 1px; margin: 4px 2px;"
        )
        return sep

    # ── transport bar ────────────────────────────────────────────────────

    def _build_transport(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("transportBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(MARGIN_LG, MARGIN_SM, MARGIN_LG, MARGIN_SM)
        lay.setSpacing(MARGIN_MD)

        # Current time
        self.lbl_time = QLabel("0:00.000")
        self.lbl_time.setObjectName("timeDisplay")
        self.lbl_time.setMinimumWidth(80)
        lay.addWidget(self.lbl_time)

        # Seek slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.sliderMoved.connect(self._on_slider)
        lay.addWidget(self.slider, 1)

        # Total time
        self.lbl_total = QLabel("0:00.000")
        self.lbl_total.setObjectName("timeDisplay")
        self.lbl_total.setMinimumWidth(80)
        lay.addWidget(self.lbl_total)

        # Loop markers
        self.lbl_loop = QLabel("")
        self.lbl_loop.setStyleSheet(
            f"color: {MAUVE}; font-size: {FONT_SIZE_XS}px; "
            f"font-family: 'Consolas', monospace; background: transparent; border: none;"
        )
        self.lbl_loop.setMaximumWidth(180)
        lay.addWidget(self.lbl_loop)

        # Volume
        vol_label = QLabel("Vol")
        vol_label.setStyleSheet(
            f"color: {OVERLAY1}; font-size: {FONT_SIZE_XS}px; background: transparent; border: none;"
        )
        lay.addWidget(vol_label)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setMaximumWidth(100)
        self.vol_slider.valueChanged.connect(self._on_volume)
        lay.addWidget(self.vol_slider)

        # Speed
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setStyleSheet(
            f"color: {SKY}; font-size: {FONT_SIZE_XS}px; "
            f"font-family: 'Consolas', monospace; background: transparent; border: none;"
        )
        self.lbl_speed.setMaximumWidth(36)
        lay.addWidget(self.lbl_speed)

        return bar

    # ── player ───────────────────────────────────────────────────────────

    def _setup_player(self):
        self.playback = PlaybackController(self)
        self.playback.position_changed.connect(self._on_position)
        self.playback.state_changed.connect(self._on_state)
        self.playback.loop_toggled.connect(self._on_loop_state)
        self.playback.speed_changed.connect(self._on_speed_changed)
        self.waveform_canvas.seek_requested.connect(self._seek_to)

    # ── timers ───────────────────────────────────────────────────────────

    def _setup_timers(self):
        self.meter_timer = QTimer()
        self.meter_timer.setInterval(50)
        self.meter_timer.timeout.connect(self._tick_meters)

        self.live_timer = QTimer()
        self.live_timer.setInterval(80)
        self.live_timer.timeout.connect(self._tick_live)

    # ── menu bar ─────────────────────────────────────────────────────────

    def _setup_menu(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        act_open = QAction("Open Audio...", self)
        act_open.setShortcut(QKeySequence(SHORTCUTS["open_file"]))
        act_open.triggered.connect(self._open_file)
        file_menu.addAction(act_open)

        self._recent_menu = file_menu.addMenu("Recent Files")
        file_menu.addSeparator()
        act_clear_recent = QAction("Clear Recent", self)
        act_clear_recent.triggered.connect(self._clear_recent)
        file_menu.addAction(act_clear_recent)
        file_menu.addSeparator()
        act_close = QAction("Close", self)
        act_close.setShortcut(QKeySequence(SHORTCUTS["close_file"]))
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        # Export menu
        export_menu = menu_bar.addMenu("Export")
        act_plot = QAction("Export Plot...", self)
        act_plot.setShortcut(QKeySequence(SHORTCUTS["export_plot"]))
        act_plot.triggered.connect(self._export_plot)
        export_menu.addAction(act_plot)
        act_rpt = QAction("Export Report...", self)
        act_rpt.setShortcut(QKeySequence(SHORTCUTS["export_report"]))
        act_rpt.triggered.connect(self._export_report)
        export_menu.addAction(act_rpt)
        act_csv = QAction("Export CSV...", self)
        act_csv.triggered.connect(self._export_csv)
        export_menu.addAction(act_csv)
        act_json = QAction("Export JSON...", self)
        act_json.triggered.connect(self._export_json)
        export_menu.addAction(act_json)
        export_menu.addSeparator()
        act_all = QAction("Export All Tabs...", self)
        act_all.triggered.connect(self._export_all)
        export_menu.addAction(act_all)

        # View menu
        view_menu = menu_bar.addMenu("View")
        act_refresh = QAction("Refresh Analysis", self)
        act_refresh.setShortcut(QKeySequence(SHORTCUTS["refresh"]))
        act_refresh.triggered.connect(self._refresh)
        view_menu.addAction(act_refresh)

        # Playback menu
        play_menu = menu_bar.addMenu("Playback")
        act_loop = QAction("Toggle Loop", self)
        act_loop.setShortcut(QKeySequence(SHORTCUTS["toggle_loop"]))
        act_loop.triggered.connect(lambda: self.btn_loop.toggle())
        play_menu.addAction(act_loop)

        self._populate_recent_menu()

    # ── keyboard shortcuts ───────────────────────────────────────────────

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(SHORTCUTS["play_pause"]), self).activated.connect(
            self._toggle_play_pause
        )
        QShortcut(QKeySequence(SHORTCUTS["stop"]), self).activated.connect(self._stop)
        QShortcut(QKeySequence(SHORTCUTS["seek_forward"]), self).activated.connect(
            self._seek_forward
        )
        QShortcut(QKeySequence(SHORTCUTS["seek_backward"]), self).activated.connect(
            self._seek_backward
        )
        QShortcut(QKeySequence(SHORTCUTS["volume_up"]), self).activated.connect(
            self._volume_up
        )
        QShortcut(QKeySequence(SHORTCUTS["volume_down"]), self).activated.connect(
            self._volume_down
        )
        QShortcut(QKeySequence(SHORTCUTS["next_tab"]), self).activated.connect(
            self._next_tab
        )
        QShortcut(QKeySequence(SHORTCUTS["prev_tab"]), self).activated.connect(
            self._prev_tab
        )

    # ── recent files ─────────────────────────────────────────────────────

    def _populate_recent_menu(self):
        self._recent_menu.clear()
        recent = self._settings.recent_files()
        if not recent:
            act = self._recent_menu.addAction("No recent files")
            act.setEnabled(False)
            return
        for path in recent:
            act = self._recent_menu.addAction(os.path.basename(path))
            act.setToolTip(path)
            act.triggered.connect(lambda checked, p=path: self._load_file(p))

    def _clear_recent(self):
        self._settings.clear_recent()
        self._populate_recent_menu()

    # ── file actions ─────────────────────────────────────────────────────

    def _open_file(self):
        last = self._settings.last_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", last,
            "Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.au);;All Files (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        """Load file asynchronously using a background worker thread."""
        self._stop()
        self.statusBar().showMessage(f"Loading: {path}...")

        # Disable controls during load
        for btn in (self.btn_play, self.btn_pause, self.btn_stop,
                    self.btn_loop, self.btn_set_a, self.btn_set_b):
            btn.setEnabled(False)

        # Run analysis in background thread
        self._worker = AnalysisWorker(path)
        self._worker.progress.connect(self._on_load_progress)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    def _on_load_progress(self, phase: str):
        self.statusBar().showMessage(f"{phase.capitalize()} audio...")

    def _on_load_finished(self, audio: AudioData):
        """Called when background loading completes."""
        self.audio = audio

        self._settings.add_recent(audio.file_path)
        self._settings.set_last_dir(audio.file_path)
        self._populate_recent_menu()

        self._update_info()
        self._update_channels()
        self._refresh()

        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_loop.setEnabled(True)
        self.btn_set_a.setEnabled(True)
        self.btn_set_b.setEnabled(True)

        self.playback.load(audio.file_path)
        self.live_canvas.set_source(
            self.audio, channel=self._current_channel,
            fft_size=self.spin_fft.value(),
        )

        self.statusBar().showMessage(f"Loaded: {self.audio.file_name}", 5000)

    def _on_load_error(self, msg: str):
        QMessageBox.critical(self, "Error", f"Could not load file:\n{msg}")

    # ── info panel update ────────────────────────────────────────────────

    def _update_info(self):
        a = self.audio
        if a is None:
            return
        self.lbl_file.setText(a.file_name)
        m, s = int(a.duration) // 60, a.duration % 60
        self.lbl_dur.setText(f"{m}:{s:05.2f}")
        self.lbl_total.setText(f"{m}:{s:05.2f}")
        self.lbl_sr.setText(f"{a.sample_rate:,} Hz")
        self.lbl_ch.setText(str(a.channels))
        self.lbl_bits.setText(f"{a.bit_depth} bit")
        sz = a.file_size
        self.lbl_fsize.setText(
            f"{sz / (1 << 20):.1f} MB" if sz > 1 << 20 else f"{sz / 1024:.1f} KB"
        )

        # Metric cards
        self.card_peak.set_value(f"{a.peak_db:.1f} dB" if a.peak_db > -np.inf else "-inf")
        self.card_rms.set_value(f"{a.rms_db:.1f} dB" if a.rms_db > -np.inf else "-inf")
        self.card_crest.set_value(f"{a.crest_factor:.1f} dB")
        self.card_dr.set_value(f"{a.dynamic_range:.1f} dB")

        # LUFS
        self.lufs_meter.set_lufs(a.lufs)

        # Pitch
        self.pitch_display.set_pitch(a.pitch)

        # Clips
        self.clip_indicator.set_clip_count(a.clip_count)

        # Advanced
        self.lbl_centroid.setText(f"{a.spectral_centroid:.0f} Hz")
        self.lbl_rolloff.setText(f"{a.spectral_rolloff:.0f} Hz")
        self.lbl_flatness.setText(f"{a.spectral_flatness:.4f}")
        self.lbl_zcr.setText(f"{a.zero_crossing_rate:.0f} Hz")

        # Band Energy
        self.band_energy_widget.set_data(a.band_energy)

        # Correlation
        self.correlation_meter.set_value(a.stereo_correlation)

        # Meters
        self.meter_l.set_levels(a.peak, a.rms)
        if a.channels > 1:
            ch1 = a.get_channel(1)
            self.meter_r.set_levels(
                float(np.max(np.abs(ch1))),
                float(np.sqrt(np.mean(ch1 ** 2))),
            )
        else:
            self.meter_r.set_levels(a.peak, a.rms)

    def _update_channels(self):
        self.combo_ch.blockSignals(True)
        self.combo_ch.clear()
        if self.audio is None:
            self.combo_ch.addItem("Mono")
        elif self.audio.channels == 1:
            self.combo_ch.addItem("Mono")
        elif self.audio.channels == 2:
            self.combo_ch.addItems(["Left", "Right", "Mono Mix"])
        else:
            for i in range(self.audio.channels):
                self.combo_ch.addItem(f"Channel {i + 1}")
            self.combo_ch.addItem("Mono Mix")
        self.combo_ch.blockSignals(False)
        self._current_channel = 0

    def _on_channel(self, idx):
        if self.audio is None:
            return
        self._current_channel = (
            min(idx, self.audio.channels - 1)
            if idx < self.combo_ch.count() - 1
            else 0
        )
        self._refresh()

    def _refresh(self):
        if self.audio is None:
            return
        ch = self._current_channel if self._current_channel >= 0 else 0
        self.waveform_canvas.plot(self.audio, channel=ch)
        self.spectrum_canvas.plot(
            self.audio, fft_size=self.spin_fft.value(), channel=ch
        )
        self.spectrogram_canvas.plot(
            self.audio,
            nperseg=self.spin_nperseg.value(),
            channel=ch,
            log_freq=self.chk_log.isChecked(),
            cmap=self.combo_cmap.currentText(),
        )
        self.rms_canvas.plot(self.audio)
        self.phase_canvas.plot(self.audio, channel=ch)
        self.live_canvas.set_source(
            self.audio, channel=ch, fft_size=self.spin_fft.value()
        )

    # ── playback ─────────────────────────────────────────────────────────

    def _play(self):
        self.playback.play()
        self.meter_timer.start()
        self.live_timer.start()

    def _pause(self):
        self.playback.pause()
        self.meter_timer.stop()
        self.live_timer.stop()

    def _stop(self):
        self.playback.stop()
        self.meter_timer.stop()
        self.live_timer.stop()
        self.slider.setValue(0)
        self.lbl_time.setText("0:00.000")
        self.waveform_canvas.set_playback_pos(-1, 1)
        self.live_canvas.stop_live()
        if self.audio:
            self.meter_l.set_levels(self.audio.peak, self.audio.rms)
            if self.audio.channels > 1:
                ch1 = self.audio.get_channel(1)
                self.meter_r.set_levels(
                    float(np.max(np.abs(ch1))),
                    float(np.sqrt(np.mean(ch1 ** 2))),
                )
            else:
                self.meter_r.set_levels(self.audio.peak, self.audio.rms)

    def _toggle_play_pause(self):
        if self.playback.state() == self.playback.player.PlaybackState.PlayingState:
            self._pause()
        else:
            self._play()

    def _seek_to(self, seconds: float):
        self.playback.seek(int(seconds * 1000))
        if self.playback.state() == self.playback.player.PlaybackState.StoppedState:
            self._play()

    def _seek_forward(self):
        self.playback.seek(self.playback.position() + 5000)

    def _seek_backward(self):
        self.playback.seek(max(0, self.playback.position() - 5000))

    def _volume_up(self):
        self.vol_slider.setValue(min(100, self.vol_slider.value() + 5))

    def _volume_down(self):
        self.vol_slider.setValue(max(0, self.vol_slider.value() - 5))

    def _on_position(self, ms):
        dur = self.playback.duration()
        if dur > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(ms / dur * 1000))
            self.slider.blockSignals(False)
        s = ms / 1000
        self.lbl_time.setText(f"{int(s) // 60}:{s % 60:06.3f}")
        if self.audio:
            self.waveform_canvas.set_playback_pos(s, self.audio.duration)
        self.live_canvas.update_position(ms)

    def _on_state(self, state):
        if state == self.playback.player.PlaybackState.StoppedState:
            self.meter_timer.stop()
            self.live_timer.stop()
            self.waveform_canvas.set_playback_pos(-1, 1)

    def _on_slider(self, v):
        dur = self.playback.duration()
        if dur > 0:
            self.playback.seek(int(v / 1000 * dur))

    def _on_volume(self, v):
        self.playback.set_volume(v / 100.0)

    def _on_speed(self, val):
        self.playback.set_speed(val)

    def _on_speed_changed(self, speed):
        self.lbl_speed.setText(f"{speed:.2g}x")

    # ── loop ─────────────────────────────────────────────────────────────

    def _on_loop_toggle(self, checked):
        self.playback.enable_loop(checked)
        if not checked:
            self.lbl_loop.setText("")

    def _on_loop_state(self, enabled):
        if enabled:
            a_ms = self.playback.loop_a()
            b_ms = self.playback.loop_b()
            a_txt = f"{a_ms / 1000:.2f}s" if a_ms is not None else "?"
            b_txt = f"{b_ms / 1000:.2f}s" if b_ms is not None else "?"
            self.lbl_loop.setText(f"Loop {a_txt} - {b_txt}")
        else:
            self.lbl_loop.setText("")

    def _on_set_loop_a(self):
        self.playback.set_loop_a()
        self._on_loop_state(self.playback.loop_enabled())

    def _on_set_loop_b(self):
        self.playback.set_loop_b()
        self._on_loop_state(self.playback.loop_enabled())

    # ── meter ticking ────────────────────────────────────────────────────

    def _tick_meters(self):
        """Optised meter update: pre-fetch channel slices, use dot-product for RMS."""
        if self.audio is None or self.audio.samples is None:
            return
        pos_s = self.playback.position() / 1000.0
        idx = int(pos_s * self.audio.sample_rate)
        win = int(0.05 * self.audio.sample_rate)
        n_samples = len(self.audio.samples)
        if idx + win >= n_samples:
            return

        # Direct array slicing (avoids get_channel function call overhead)
        w0 = self.audio.samples[idx: idx + win, 0]
        abs_w0 = np.abs(w0)
        pk0 = float(np.max(abs_w0))
        rms0 = float(np.sqrt(np.dot(w0, w0) / win))  # dot product faster than mean(w0**2)
        self.meter_l.set_levels(pk0, rms0)

        if self.audio.channels > 1:
            w1 = self.audio.samples[idx: idx + win, 1]
            pk1 = float(np.max(np.abs(w1)))
            rms1 = float(np.sqrt(np.dot(w1, w1) / win))
            self.meter_r.set_levels(pk1, rms1)
        else:
            self.meter_r.set_levels(pk0, rms0)

    def _tick_live(self):
        if self.audio is None:
            return
        self.live_canvas.update_position(self.playback.position())
        self.live_canvas._tick()

    # ── tab navigation ───────────────────────────────────────────────────

    def _next_tab(self):
        idx = self.tabs.currentIndex()
        if idx < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(idx + 1)

    def _prev_tab(self):
        idx = self.tabs.currentIndex()
        if idx > 0:
            self.tabs.setCurrentIndex(idx - 1)

    # ── export ───────────────────────────────────────────────────────────

    def _export_plot(self):
        current_canvas = self.tabs.currentWidget().findChild(FigureCanvas)
        if current_canvas and self.audio:
            Exporter.export_plot(self, current_canvas, self.audio.file_name)

    def _export_report(self):
        if self.audio:
            Exporter.export_text_report(self, self.audio, self.audio.file_name)

    def _export_csv(self):
        if self.audio:
            Exporter.export_csv(self, self.audio, self.audio.file_name)

    def _export_json(self):
        if self.audio:
            Exporter.export_json_report(self, self.audio, self.audio.file_name)

    def _export_all(self):
        if self.audio:
            canvases = {
                "waveform": self.waveform_canvas,
                "spectrum": self.spectrum_canvas,
                "spectrogram": self.spectrogram_canvas,
                "loudness": self.rms_canvas,
                "phase": self.phase_canvas,
            }
            Exporter.export_all_plots(self, canvases, self.audio.file_name)

    # ── stylesheet ───────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(build_stylesheet())

    # ── persistence ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._settings.set("window_geometry", self.saveGeometry())
        super().closeEvent(event)
