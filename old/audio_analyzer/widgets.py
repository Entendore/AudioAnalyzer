#!/usr/bin/env python3
"""
widgets.py — Custom Qt widgets for the Audio Analyzer.

Professional dark-mode widgets with consistent visual language:
  - MetricCard         (key-value pair with colored accent)
  - LevelMeter         (peak-hold with decay, gradient bar, dB scale)
  - BandEnergyWidget   (horizontal bars per frequency band)
  - CorrelationMeter   (stereo correlation -1..+1 gauge)
  - ClipIndicator      (red warning when clips detected)
  - PitchDisplay       (shows detected fundamental frequency)
  - LUFSMeter          (simplified loudness bar)
  - SectionHeader      (reusable section divider with label)
  - StatusCardRow      (row of MetricCards)
"""

import math
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QRect, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QLinearGradient,
    QRadialGradient, QPainterPath,
)

from config import (
    DARK_BG, DARK_AXIS, MANTLE, SURFACE0, SURFACE1, SURFACE2,
    OVERLAY0, OVERLAY1, LIGHT_TXT, SUBTXT1, SUBTXT0, GRID_CLR,
    BLUE, BLUE_DIM, BLUE_LIGHT, GREEN, RED, RED_DIM, YELLOW,
    MAUVE, PEACH, TEAL, SKY,
    BAND_COLORS,
    RADIUS_SM, RADIUS_MD, RADIUS_LG,
    FONT_SIZE_XS, FONT_SIZE_SM, FONT_SIZE_MD, FONT_SIZE_LG, FONT_SIZE_XL,
    MARGIN_SM, MARGIN_MD, MARGIN_LG,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Section Header (divider with label)
# ──────────────────────────────────────────────────────────────────────────────

class SectionHeader(QWidget):
    """Horizontal divider with a centered or left-aligned label."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        mid_y = h // 2

        # Left line
        p.setPen(QPen(QColor(SURFACE1), 1))
        p.drawLine(0, mid_y, 4, mid_y)

        # Title
        if self._title:
            p.setFont(QFont("Segoe UI", FONT_SIZE_XS, QFont.Bold))
            p.setPen(QColor(OVERLAY1))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(self._title)
            text_x = 8
            p.drawText(text_x, mid_y + fm.ascent() // 2 - 1, self._title)

            # Right line
            line_start = text_x + tw + 6
            p.setPen(QPen(QColor(SURFACE1), 1))
            p.drawLine(line_start, mid_y, w, mid_y)
        else:
            p.drawLine(4, mid_y, w, mid_y)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Metric Card (key-value display with color accent)
# ──────────────────────────────────────────────────────────────────────────────

class MetricCard(QWidget):
    """Compact metric display: label on top, value below, with left accent stripe."""

    def __init__(self, label: str = "", accent_color: str = BLUE, parent=None):
        super().__init__(parent)
        self._label = label
        self._value = "—"
        self._accent = accent_color
        self.setMinimumSize(80, 44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, text: str):
        self._value = text
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = RADIUS_SM

        # Background card
        p.setBrush(QColor(SURFACE0))
        p.setPen(Qt.NoPen)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), r, r)
        p.drawPath(path)

        # Left accent stripe
        stripe_w = 3
        p.setBrush(QColor(self._accent))
        stripe_path = QPainterPath()
        stripe_path.addRoundedRect(QRectF(0, 0, stripe_w + r, h), r, r)
        p.drawPath(stripe_path)
        # Cover right side of stripe to make it flush
        p.fillRect(QRect(stripe_w, 0, r, h), QColor(self._accent))

        # Label
        label_x = stripe_w + 8
        p.setPen(QColor(OVERLAY1))
        p.setFont(QFont("Segoe UI", FONT_SIZE_XS))
        p.drawText(QRect(label_x, 3, w - label_x - 4, 14), Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # Value
        p.setPen(QColor(LIGHT_TXT))
        p.setFont(QFont("Segoe UI Semibold", FONT_SIZE_MD, QFont.Bold))
        p.drawText(QRect(label_x, 17, w - label_x - 4, 22), Qt.AlignLeft | Qt.AlignVCenter, self._value)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Level Meter (with peak hold & decay)
# ──────────────────────────────────────────────────────────────────────────────

class LevelMeter(QWidget):
    """Vertical level meter with gradient bar, peak-hold indicator, and dB scale."""

    def __init__(self, label: str = "L", parent=None):
        super().__init__(parent)
        self._label = label
        self._peak = 0.0
        self._rms = 0.0
        self._peak_hold = 0.0
        self._hold_timer = 0
        self.setFixedSize(52, 200)

        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(50)
        self._decay_timer.timeout.connect(self._decay_peak)
        self._decay_timer.start()

    def set_levels(self, peak: float, rms: float):
        self._peak = min(peak, 1.0)
        self._rms = min(rms, 1.0)
        if self._peak >= self._peak_hold:
            self._peak_hold = self._peak
            self._hold_timer = 0
        self.update()

    def _decay_peak(self):
        if self._peak_hold > 0:
            self._hold_timer += 1
            if self._hold_timer > 10:
                self._peak_hold = max(self._peak, self._peak_hold - 0.015)
                self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Card background with rounded corners
        w, h = self.width(), self.height()
        p.setBrush(QColor(SURFACE0))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, w, h), RADIUS_SM, RADIUS_SM)

        m = 5
        bar_x = m + 4
        bar_w = w - 2 * m - 4
        bar_area_h = h - 2 * m - 16  # reserve bottom for label

        # dB scale background ticks
        p.setPen(QPen(QColor(SURFACE1), 0.5))
        p.setFont(QFont("Segoe UI", 5))
        for db_val in (0, -3, -6, -12, -18, -24, -36, -48):
            frac = 10 ** (db_val / 20.0)
            y = m + bar_area_h - int(bar_area_h * frac)
            p.drawLine(bar_x - 2, y, bar_x + bar_w + 2, y)

        # RMS bar (gradient)
        if self._rms > 0:
            bar_h = int(bar_area_h * self._rms)
            grad = QLinearGradient(0, m + bar_area_h, 0, m)
            grad.setColorAt(0.0, QColor(GREEN))
            grad.setColorAt(0.55, QColor(GREEN))
            grad.setColorAt(0.72, QColor(YELLOW))
            grad.setColorAt(0.88, QColor(PEACH))
            grad.setColorAt(1.0, QColor(RED))
            p.fillRect(bar_x, m + bar_area_h - bar_h, bar_w, bar_h, grad)

        # Peak-hold line
        if self._peak_hold > 0:
            py = m + bar_area_h - int(bar_area_h * self._peak_hold)
            p.setPen(QPen(QColor(RED), 2))
            p.drawLine(bar_x, py, bar_x + bar_w, py)

        # Border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(SURFACE1), 1))
        p.drawRoundedRect(QRectF(0, 0, w, h), RADIUS_SM, RADIUS_SM)

        # Label at bottom
        p.setPen(QColor(SUBTXT0))
        p.setFont(QFont("Segoe UI", FONT_SIZE_SM, QFont.Bold))
        p.drawText(QRect(0, h - 14, w, 14), Qt.AlignCenter, self._label)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Band Energy Widget
# ──────────────────────────────────────────────────────────────────────────────

class BandEnergyWidget(QWidget):
    """Horizontal bar chart for frequency band energy percentages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict = {}
        self.setMinimumHeight(180)

    def set_data(self, data: dict):
        self._data = data
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        n = len(self._data)
        h = self.height()
        w = self.width()
        m_left, m_right, m_top, m_bot = 85, 50, 4, 4
        draw_w = w - m_left - m_right
        draw_h = h - m_top - m_bot
        bar_h = draw_h / n * 0.65
        spacing = draw_h / n * 0.35

        for i, (label, val) in enumerate(self._data.items()):
            y = m_top + i * (bar_h + spacing)
            bar_w = max(2, int(draw_w * val))

            # Track background
            p.setBrush(QColor(MANTLE))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(m_left, y, draw_w, bar_h), 2, 2)

            # Colored bar
            c = BAND_COLORS[i % len(BAND_COLORS)]
            p.setBrush(QColor(c))
            p.drawRoundedRect(QRectF(m_left, y, bar_w, bar_h), 2, 2)

            # Label
            p.setPen(QColor(SUBTXT0))
            p.setFont(QFont("Segoe UI", FONT_SIZE_XS))
            p.drawText(
                QRect(0, int(y), m_left - 6, int(bar_h)),
                Qt.AlignRight | Qt.AlignVCenter, label.replace("\n", " "),
            )

            # Percentage
            p.setPen(QColor(LIGHT_TXT))
            p.setFont(QFont("Segoe UI", FONT_SIZE_XS, QFont.Bold))
            pct_text = f"{val * 100:.1f}%"
            text_x = m_left + bar_w + 5
            if text_x + 35 > w:
                text_x = m_left + bar_w - 35
                p.setPen(QColor(DARK_BG))
            p.drawText(QRect(text_x, int(y), 40, int(bar_h)), Qt.AlignLeft | Qt.AlignVCenter, pct_text)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Stereo Correlation Meter
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationMeter(QWidget):
    """
    Semicircular gauge showing stereo correlation from -1 (out of phase)
    through 0 (uncorrelated) to +1 (in phase / mono).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value: float = 0.0
        self.setFixedSize(180, 100)

    def set_value(self, correlation: float):
        self._value = max(-1.0, min(1.0, correlation))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h - 12
        radius = min(cx, cy) - 16

        # Background track (thick arc)
        pen_w = 10
        p.setPen(QPen(QColor(SURFACE0), pen_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 180 * 16, 180 * 16)

        # Gradient arc: red → yellow → green
        # Draw as segments
        seg_pen = pen_w - 2
        # Red zone: -1 to 0 (left half)
        p.setPen(QPen(QColor(RED_DIM), seg_pen, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 180 * 16, 60 * 16)
        # Yellow zone
        p.setPen(QPen(QColor(YELLOW), seg_pen, Qt.SolidLine, Qt.FlatCap))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 240 * 16, 30 * 16)
        # Green zone: 0 to +1 (right half)
        p.setPen(QPen(QColor(GREEN), seg_pen, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 60 * 16, 60 * 16)

        # Needle
        angle_deg = 180 - (self._value + 1) / 2 * 180
        angle_rad = math.radians(angle_deg)
        needle_len = radius * 0.82
        nx = cx + int(needle_len * math.cos(angle_rad))
        ny = cy - int(needle_len * math.sin(angle_rad))

        # Needle shadow
        p.setPen(QPen(QColor(DARK_BG), 3))
        p.drawLine(cx + 1, cy + 1, nx + 1, ny + 1)

        # Needle
        p.setPen(QPen(QColor(LIGHT_TXT), 2))
        p.drawLine(cx, cy, nx, ny)

        # Centre pivot
        p.setBrush(QColor(LIGHT_TXT))
        p.setPen(QPen(QColor(DARK_AXIS), 1))
        p.drawEllipse(cx - 4, cy - 4, 8, 8)

        # Scale labels
        p.setPen(QColor(OVERLAY1))
        p.setFont(QFont("Segoe UI", FONT_SIZE_XS))
        p.drawText(cx - radius - 2, cy + 14, "-1")
        p.drawText(cx - 4, cy + 14, "0")
        p.drawText(cx + radius - 6, cy + 14, "+1")

        # Value display
        p.setFont(QFont("Segoe UI", FONT_SIZE_LG, QFont.Bold))
        val_text = f"{self._value:+.2f}" if abs(self._value) > 0.005 else "N/A"
        # Colour code the value
        if self._value > 0.5:
            p.setPen(QColor(GREEN))
        elif self._value > 0:
            p.setPen(QColor(YELLOW))
        else:
            p.setPen(QColor(RED))
        p.drawText(QRect(cx - 40, cy - 28, 80, 20), Qt.AlignCenter, val_text)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Clip Indicator
# ──────────────────────────────────────────────────────────────────────────────

class ClipIndicator(QWidget):
    """Warning badge that lights up red when clips are detected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._count: int = 0
        self.setMinimumSize(80, 28)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_clip_count(self, count: int):
        self._count = count
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = RADIUS_SM

        if self._count > 0:
            # Danger gradient
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(RED_DIM))
            grad.setColorAt(1.0, QColor(RED))
            p.setBrush(grad)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

            # Pulsing indicator dot
            p.setBrush(QColor(FLAMINGO))
            p.drawEllipse(8, h // 2 - 3, 6, 6)

            p.setPen(QColor(DARK_BG))
            p.setFont(QFont("Segoe UI", FONT_SIZE_SM, QFont.Bold))
            p.drawText(QRect(18, 0, w - 22, h), Qt.AlignVCenter | Qt.AlignLeft, f"CLIP x{self._count}")
        else:
            p.setBrush(QColor(SURFACE0))
            p.setPen(QPen(QColor(SURFACE1), 1))
            p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

            p.setPen(QColor(OVERLAY0))
            p.setFont(QFont("Segoe UI", FONT_SIZE_SM))
            p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "No Clips")

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Pitch Display
# ──────────────────────────────────────────────────────────────────────────────

class PitchDisplay(QWidget):
    """Shows the detected fundamental frequency with a note name."""

    _NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pitch: float = 0.0
        self.setMinimumSize(80, 44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_pitch(self, hz: float):
        self._pitch = hz
        self.update()

    @staticmethod
    def _hz_to_note(hz: float) -> str:
        if hz <= 0:
            return "—"
        midi = 69 + 12 * math.log2(hz / 440.0)
        note_idx = int(round(midi)) % 12
        octave = int(round(midi)) // 12 - 1
        return f"{PitchDisplay._NOTE_NAMES[note_idx]}{octave}"

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        r = RADIUS_SM

        # Card background
        p.setBrush(QColor(SURFACE0))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        # Left accent
        accent_w = 3
        p.setBrush(QColor(TEAL))
        p.drawRoundedRect(QRectF(0, 0, accent_w + r, h), r, r)
        p.fillRect(QRect(accent_w, 0, r, h), QColor(TEAL))

        content_x = accent_w + 8

        if self._pitch > 0:
            note = self._hz_to_note(self._pitch)
            # Note name (large)
            p.setPen(QColor(TEAL))
            p.setFont(QFont("Segoe UI", FONT_SIZE_LG, QFont.Bold))
            p.drawText(QRect(content_x, 1, w - content_x - 4, 22), Qt.AlignLeft | Qt.AlignVCenter, note)
            # Frequency
            p.setPen(QColor(OVERLAY1))
            p.setFont(QFont("Segoe UI", FONT_SIZE_XS))
            p.drawText(QRect(content_x, 22, w - content_x - 4, 18), Qt.AlignLeft | Qt.AlignVCenter, f"{self._pitch:.1f} Hz")
        else:
            p.setPen(QColor(OVERLAY0))
            p.setFont(QFont("Segoe UI", FONT_SIZE_SM))
            p.drawText(QRect(content_x, 0, w - content_x - 4, h), Qt.AlignVCenter | Qt.AlignLeft, "No Pitch")

        # Border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(SURFACE1), 1))
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  LUFS Meter Bar
# ──────────────────────────────────────────────────────────────────────────────

class LUFSMeter(QWidget):
    """Horizontal bar meter for integrated LUFS with colour coding and range labels."""

    _MIN_LUFS = -60.0
    _MAX_LUFS = 0.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lufs: float = -np.inf
        self.setMinimumWidth(160)
        self.setFixedHeight(32)

    def set_lufs(self, lufs: float):
        self._lufs = lufs
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        m = 3
        r = RADIUS_SM

        # Track background
        p.setBrush(QColor(MANTLE))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(m, m, w - 2 * m, h - 2 * m), r, r)

        if self._lufs > -np.inf:
            frac = (self._lufs - self._MIN_LUFS) / (self._MAX_LUFS - self._MIN_LUFS)
            frac = max(0.0, min(1.0, frac))
            bar_w = max(4, int((w - 2 * m) * frac))

            # Colour: green if < -16, yellow if -16..-6, red if > -6
            if self._lufs < -16:
                bar_color = QColor(GREEN)
            elif self._lufs < -6:
                bar_color = QColor(YELLOW)
            else:
                bar_color = QColor(RED)

            p.setBrush(bar_color)
            p.drawRoundedRect(QRectF(m, m, bar_w, h - 2 * m), r, r)

            # Value text
            p.setPen(QColor(DARK_BG))
            p.setFont(QFont("Segoe UI Semibold", FONT_SIZE_SM, QFont.Bold))
            p.drawText(QRect(m, m, w - 2 * m, h - 2 * m), Qt.AlignCenter, f"{self._lufs:.1f} LUFS")

            # Range tick marks
            p.setPen(QPen(QColor(SURFACE2), 0.5))
            for db in (-40, -24, -14, -6):
                x = m + int((w - 2 * m) * (db - self._MIN_LUFS) / (self._MAX_LUFS - self._MIN_LUFS))
                p.drawLine(x, h - m - 2, x, h - m)
        else:
            p.setPen(QColor(OVERLAY0))
            p.setFont(QFont("Segoe UI", FONT_SIZE_SM))
            p.drawText(QRect(m, m, w - 2 * m, h - 2 * m), Qt.AlignCenter, "— LUFS")

        # Border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(SURFACE1), 1))
        p.drawRoundedRect(QRectF(m, m, w - 2 * m, h - 2 * m), r, r)

        p.end()
