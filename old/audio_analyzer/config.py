#!/usr/bin/env python3
"""
config.py — Centralized theme constants, application settings, keyboard shortcuts,
recent-files management, and professional stylesheet for the Audio Analyzer.
"""

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont

# ──────────────────────────────────────────────────────────────────────────────
#  Catppuccin Mocha Theme Palette
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#1e1e2e"
DARK_AXIS  = "#181825"
MANTLE     = "#11111b"     # deepest background
SURFACE0   = "#313244"
SURFACE1   = "#45475a"
SURFACE2   = "#585b70"
OVERLAY0   = "#6c7086"
OVERLAY1   = "#7f849c"
OVERLAY2   = "#9399b2"
LIGHT_TXT  = "#cdd6f4"
SUBTXT1    = "#bac2de"
SUBTXT0    = "#a6adc8"
GRID_CLR   = "#585b70"

BLUE       = "#89b4fa"
BLUE_LIGHT = "#b4befe"
BLUE_DIM   = "#5b7fb8"
GREEN      = "#a6e3a1"
RED        = "#f38ba8"
RED_DIM    = "#b0546a"
YELLOW     = "#f9e2af"
MAUVE      = "#cba6f7"
PEACH      = "#fab387"
TEAL       = "#94e2d5"
PINK       = "#f5c2e7"
SKY        = "#89dceb"
LAVENDER   = "#b4befe"
ROSEWATER  = "#f5e0dc"
FLAMINGO   = "#f2cdcd"

# Derived helpers
ACCENT     = BLUE
SUCCESS    = GREEN
WARNING    = YELLOW
ERROR      = RED

# ──────────────────────────────────────────────────────────────────────────────
#  Layout Constants (NEW — for consistent spacing)
# ──────────────────────────────────────────────────────────────────────────────

MARGIN_XS      = 2
MARGIN_SM      = 4
MARGIN_MD      = 8
MARGIN_LG      = 12
MARGIN_XL      = 16
MARGIN_XXL     = 24

RADIUS_SM      = 4
RADIUS_MD      = 6
RADIUS_LG      = 8
RADIUS_XL      = 12

FONT_SIZE_XS   = 7
FONT_SIZE_SM   = 8
FONT_SIZE_MD   = 10
FONT_SIZE_LG   = 12
FONT_SIZE_XL   = 14
FONT_SIZE_XXL  = 18

# ──────────────────────────────────────────────────────────────────────────────
#  Audio Frequency Band Definitions
# ──────────────────────────────────────────────────────────────────────────────

FREQ_BANDS = [
    ("Sub Bass\n(20-60Hz)",    20,    60),
    ("Bass\n(60-250Hz)",       60,   250),
    ("Low Mid\n(250-500Hz)",  250,   500),
    ("Mid\n(500-2kHz)",       500,  2000),
    ("High Mid\n(2k-4kHz)",  2000,  4000),
    ("Presence\n(4k-6kHz)",  4000,  6000),
    ("Brilliance\n(6k-20kHz)",6000, 20000),
]

BAND_COLORS = [RED, PEACH, YELLOW, GREEN, TEAL, BLUE, MAUVE]

# ──────────────────────────────────────────────────────────────────────────────
#  Spectrogram Colour Maps
# ──────────────────────────────────────────────────────────────────────────────

SPECTROGRAM_CMAPS = [
    "magma", "inferno", "plasma", "viridis",
    "turbo", "hot", "cividis", "coolwarm",
]

# ──────────────────────────────────────────────────────────────────────────────
#  Default Analysis Parameters
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_FFT_SIZE    = 4096
DEFAULT_SPECT_WIN   = 1024
DEFAULT_RMS_WINDOW  = 0.05     # 50 ms
DEFAULT_WELCH_NP    = 4096
DEFAULT_SPECT_CMAP  = "magma"
DEFAULT_PLAYBACK_VOL = 0.8
DEFAULT_PLAYBACK_SPEED = 1.0
MAX_WAVEFORM_PTS    = 60_000
MAX_SPECTROGRAM_SEC = 60

# ──────────────────────────────────────────────────────────────────────────────
#  Keyboard Shortcuts
# ──────────────────────────────────────────────────────────────────────────────

SHORTCUTS = {
    "open_file":       "Ctrl+O",
    "play_pause":      "Space",
    "stop":            "S",
    "seek_forward":    "Right",
    "seek_backward":   "Left",
    "volume_up":       "Up",
    "volume_down":     "Down",
    "refresh":         "F5",
    "export_plot":     "Ctrl+E",
    "export_report":   "Ctrl+Shift+E",
    "toggle_loop":     "L",
    "next_tab":        "Ctrl+Tab",
    "prev_tab":        "Ctrl+Shift+Tab",
    "zoom_in":         "Ctrl+=",
    "zoom_out":        "Ctrl+-",
    "reset_zoom":      "Ctrl+0",
    "close_file":      "Ctrl+W",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Application Metadata
# ──────────────────────────────────────────────────────────────────────────────

APP_NAME    = "Advanced Audio Analyzer"
APP_VERSION = "2.0.0"
APP_ORG     = "AudioAnalyzer"

# ──────────────────────────────────────────────────────────────────────────────
#  Settings Manager (QSettings wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class AppSettings:
    """Persistent application settings via QSettings."""

    _KEYS = {
        "window_geometry":   "main_window/geometry",
        "window_state":      "main_window/state",
        "splitter_sizes":    "main_window/splitter",
        "recent_files":      "history/recent_files",
        "last_dir":          "history/last_directory",
        "fft_size":          "analysis/fft_size",
        "spect_win":         "analysis/spectrogram_window",
        "spect_cmap":        "analysis/spectrogram_cmap",
        "log_freq":          "analysis/log_frequency",
        "volume":            "playback/volume",
        "playback_speed":    "playback/speed",
        "auto_refresh":      "analysis/auto_refresh",
    }

    def __init__(self):
        self._q = QSettings(APP_ORG, APP_NAME)

    # ── generic get / set ────────────────────────────────────────────────

    def get(self, key: str, default=None):
        return self._q.value(self._KEYS.get(key, key), default)

    def set(self, key: str, value):
        self._q.setValue(self._KEYS.get(key, key), value)

    # ── recent files ─────────────────────────────────────────────────────

    def recent_files(self, max_count: int = 10) -> list[str]:
        raw = self.get("recent_files", [])
        return raw[:max_count] if isinstance(raw, list) else []

    def add_recent(self, path: str, max_count: int = 10):
        recent = self.recent_files(max_count)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self.set("recent_files", recent[:max_count])

    def clear_recent(self):
        self.set("recent_files", [])

    # ── last directory ───────────────────────────────────────────────────

    def last_dir(self) -> str:
        return self.get("last_dir", "")

    def set_last_dir(self, path: str):
        import os
        self.set("last_dir", os.path.dirname(path) if os.path.isfile(path) else path)


# ──────────────────────────────────────────────────────────────────────────────
#  Global Stylesheet — Professional Edition
# ──────────────────────────────────────────────────────────────────────────────

def build_stylesheet() -> str:
    """Return a polished, professional dark-mode stylesheet."""
    return f"""
    /* ── Global ──────────────────────────────────────────────────────── */
    QMainWindow {{
        background-color: {DARK_BG};
    }}
    QWidget {{
        background-color: {DARK_BG};
        color: {LIGHT_TXT};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: {FONT_SIZE_MD}px;
    }}
    QScrollArea {{
        background-color: {DARK_BG};
        border: none;
    }}

    /* ── Header Bar ──────────────────────────────────────────────────── */
    QWidget#headerBar {{
        background-color: {MANTLE};
        border-bottom: 1px solid {SURFACE0};
        padding: 0;
    }}
    QLabel#appTitle {{
        color: {BLUE};
        font-size: {FONT_SIZE_XL}px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }}
    QLabel#appVersion {{
        color: {OVERLAY0};
        font-size: {FONT_SIZE_SM}px;
    }}

    /* ── Toolbar ─────────────────────────────────────────────────────── */
    QWidget#toolBar {{
        background-color: {DARK_AXIS};
        border: 1px solid {SURFACE0};
        border-radius: {RADIUS_LG}px;
        padding: 2px;
    }}
    QWidget#toolBarGroup {{
        background-color: transparent;
        border: none;
    }}

    /* ── GroupBox / Section Panels ───────────────────────────────────── */
    QGroupBox {{
        background-color: {DARK_AXIS};
        border: 1px solid {SURFACE0};
        border-radius: {RADIUS_LG}px;
        margin-top: 18px;
        padding: {MARGIN_LG}px;
        padding-top: 28px;
        font-weight: bold;
        font-size: {FONT_SIZE_SM}px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {MARGIN_LG}px;
        padding: 0 8px;
        color: {SUBTXT1};
    }}

    /* ── Buttons ─────────────────────────────────────────────────────── */
    QPushButton {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_SM}px;
        padding: 5px 14px;
        font-size: {FONT_SIZE_MD}px;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {SURFACE1};
        border-color: {SURFACE2};
    }}
    QPushButton:pressed {{
        background-color: {SURFACE2};
    }}
    QPushButton:disabled {{
        background-color: {DARK_BG};
        color: {OVERLAY0};
        border-color: {SURFACE0};
    }}
    QPushButton#accentBtn {{
        background-color: {BLUE};
        color: {DARK_BG};
        font-weight: bold;
        border: none;
        padding: 6px 18px;
    }}
    QPushButton#accentBtn:hover {{
        background-color: {BLUE_LIGHT};
    }}
    QPushButton#accentBtn:pressed {{
        background-color: {BLUE_DIM};
    }}
    QPushButton#loopBtn {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_SM}px;
        padding: 4px 12px;
        font-size: {FONT_SIZE_SM}px;
    }}
    QPushButton#loopBtn:checked {{
        background-color: {MAUVE};
        color: {DARK_BG};
        font-weight: bold;
        border: none;
    }}
    QPushButton#transportBtn {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_MD}px;
        padding: 4px 8px;
        font-size: {FONT_SIZE_LG}px;
        min-width: 32px;
        min-height: 28px;
    }}
    QPushButton#transportBtn:hover {{
        background-color: {SURFACE1};
        color: {BLUE};
    }}
    QPushButton#transportBtn:disabled {{
        color: {OVERLAY0};
    }}
    QPushButton#sectionBtn {{
        background-color: transparent;
        color: {SUBTXT1};
        border: none;
        border-radius: {RADIUS_SM}px;
        padding: 3px 10px;
        font-size: {FONT_SIZE_SM}px;
    }}
    QPushButton#sectionBtn:hover {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
    }}

    /* ── Sliders ─────────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background: {SURFACE0};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {BLUE};
        border: 2px solid {BLUE};
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {BLUE_LIGHT};
        border-color: {BLUE_LIGHT};
    }}
    QSlider::sub-page:horizontal {{
        background: {BLUE_DIM};
        border-radius: 3px;
    }}

    /* ── ComboBox ────────────────────────────────────────────────────── */
    QComboBox {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_SM}px;
        padding: 4px 10px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        border-color: {BLUE_DIM};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {SUBTXT0};
    }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        selection-background-color: {BLUE};
        selection-color: {DARK_BG};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_SM}px;
        padding: 2px;
    }}

    /* ── SpinBox ─────────────────────────────────────────────────────── */
    QSpinBox, QDoubleSpinBox {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_SM}px;
        padding: 4px 6px;
        min-height: 20px;
    }}
    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {BLUE_DIM};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: {SURFACE1};
        border: none;
        width: 16px;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {BLUE_DIM};
    }}

    /* ── Tabs ────────────────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {SURFACE0};
        border-radius: {RADIUS_SM}px;
        background-color: {DARK_AXIS};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {SURFACE0};
        color: {SUBTXT0};
        padding: 8px 18px;
        margin-right: 2px;
        border-top-left-radius: {RADIUS_MD}px;
        border-top-right-radius: {RADIUS_MD}px;
        border: 1px solid {SURFACE0};
        border-bottom: none;
        font-size: {FONT_SIZE_SM}px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        background-color: {DARK_AXIS};
        color: {BLUE};
        font-weight: bold;
        border-color: {SURFACE0};
        border-bottom: 2px solid {BLUE};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {SURFACE1};
        color: {LIGHT_TXT};
    }}

    /* ── CheckBox ────────────────────────────────────────────────────── */
    QCheckBox {{
        color: {LIGHT_TXT};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: {RADIUS_SM}px;
        border: 1px solid {SURFACE2};
        background-color: {SURFACE0};
    }}
    QCheckBox::indicator:checked {{
        background-color: {BLUE};
        border-color: {BLUE};
    }}
    QCheckBox::indicator:hover {{
        border-color: {BLUE_DIM};
    }}

    /* ── Labels ──────────────────────────────────────────────────────── */
    QLabel {{
        color: {LIGHT_TXT};
    }}
    QLabel#metricValue {{
        color: {LIGHT_TXT};
        font-size: {FONT_SIZE_MD}px;
        font-weight: 600;
        font-family: 'Segoe UI Semibold', 'Inter Medium', sans-serif;
    }}
    QLabel#metricLabel {{
        color: {OVERLAY1};
        font-size: {FONT_SIZE_SM}px;
    }}
    QLabel#sectionHeader {{
        color: {SUBTXT1};
        font-size: {FONT_SIZE_SM}px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 0;
    }}
    QLabel#timeDisplay {{
        color: {LIGHT_TXT};
        font-family: 'Consolas', 'Cascadia Code', 'JetBrains Mono', monospace;
        font-size: {FONT_SIZE_LG}px;
        font-weight: bold;
        background-color: {MANTLE};
        border: 1px solid {SURFACE0};
        border-radius: {RADIUS_SM}px;
        padding: 2px 8px;
    }}

    /* ── Status Bar ──────────────────────────────────────────────────── */
    QStatusBar {{
        background-color: {MANTLE};
        color: {SUBTXT0};
        border-top: 1px solid {SURFACE0};
        font-size: {FONT_SIZE_SM}px;
        padding: 2px 8px;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* ── Splitter ────────────────────────────────────────────────────── */
    QSplitter::handle {{
        background-color: {SURFACE0};
        width: 2px;
    }}
    QSplitter::handle:hover {{
        background-color: {BLUE_DIM};
    }}

    /* ── Transport Bar ───────────────────────────────────────────────── */
    QWidget#transportBar {{
        background-color: {MANTLE};
        border: 1px solid {SURFACE0};
        border-radius: {RADIUS_LG}px;
    }}

    /* ── Menus ───────────────────────────────────────────────────────── */
    QMenuBar {{
        background-color: {MANTLE};
        color: {SUBTXT1};
        border-bottom: 1px solid {SURFACE0};
        font-size: {FONT_SIZE_SM}px;
        padding: 1px;
    }}
    QMenuBar::item {{
        padding: 4px 10px;
        border-radius: {RADIUS_SM}px;
    }}
    QMenuBar::item:selected {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
    }}
    QMenu {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        border-radius: {RADIUS_MD}px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px;
        border-radius: {RADIUS_SM}px;
    }}
    QMenu::item:selected {{
        background-color: {BLUE};
        color: {DARK_BG};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {SURFACE1};
        margin: 4px 8px;
    }}

    /* ── Tooltips ────────────────────────────────────────────────────── */
    QToolTip {{
        background-color: {SURFACE0};
        color: {LIGHT_TXT};
        border: 1px solid {SURFACE1};
        padding: 6px 10px;
        border-radius: {RADIUS_SM}px;
        font-size: {FONT_SIZE_SM}px;
    }}

    /* ── Scrollbar ───────────────────────────────────────────────────── */
    QScrollBar:vertical {{
        background: {DARK_AXIS};
        width: 10px;
        border: none;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {SURFACE1};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {SURFACE2};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    /* ── Divider widget ──────────────────────────────────────────────── */
    QWidget#hDivider {{
        background-color: {SURFACE0};
        max-height: 1px;
        min-height: 1px;
        margin: 4px 8px;
    }}
    """
