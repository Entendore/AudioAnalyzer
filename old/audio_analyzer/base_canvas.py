#!/usr/bin/env python3
"""
base_canvas.py — Enhanced Matplotlib canvas base with performance-optimised
playback cursor (blitting), in-place crosshair updates, and professional styling.

Performance features:
  - Blitted playback cursor (no full redraw per frame)
  - In-place crosshair line updates (no create/destroy per mouse move)
  - Throttled crosshair updates to avoid excessive redraws
  - Cached background for blitting
"""

import time

from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import DARK_BG, DARK_AXIS, LIGHT_TXT, SUBTXT0, GRID_CLR, SURFACE0, RED


class BaseCanvas(FigureCanvas):
    """Base class for all Matplotlib-based analysis canvases."""

    seek_requested = Signal(float)   # seconds
    cursor_moved   = Signal(float, float)  # (x, y) in data coords

    def __init__(self, parent=None, height_ratio: float = 3.0):
        self.fig = Figure(figsize=(10, height_ratio), facecolor=DARK_BG, dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.10)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)

        # Internal state
        self._cbar = None
        self._playback_line = None
        self._crosshair_v = None
        self._crosshair_h = None
        self._annotation = None

        # ── Blitting state ──────────────────────────────────────────────
        self._blit_bg = None              # cached background bitmap
        self._blitting_enabled = False

        # ── Crosshair throttle ──────────────────────────────────────────
        self._last_motion_time: float = 0.0
        self._motion_throttle_sec: float = 0.016   # ~60 FPS cap for hover

        # Mouse interaction
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("axes_leave_event", self._on_leave)

    # ── styling ──────────────────────────────────────────────────────────

    def _style_ax(self, xlabel: str = "", ylabel: str = "", title: str = ""):
        """Apply professional dark theme to the main axes."""
        self.ax.set_facecolor(DARK_AXIS)
        self.ax.tick_params(
            colors=SUBTXT0, labelsize=7, direction="in",
            length=3, width=0.5,
        )
        for spine in self.ax.spines.values():
            spine.set_color(SURFACE0)
            spine.set_linewidth(0.8)
        if xlabel:
            self.ax.set_xlabel(xlabel, color=SUBTXT0, fontsize=8, labelpad=6)
        if ylabel:
            self.ax.set_ylabel(ylabel, color=SUBTXT0, fontsize=8, labelpad=6)
        if title:
            self.ax.set_title(
                title, color=LIGHT_TXT, fontsize=10, fontweight="bold",
                pad=10, loc="left",
            )
        self.ax.grid(True, alpha=0.12, color=GRID_CLR, linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(which='minor', length=2, width=0.3, colors=SURFACE0)

    # ── blitting helpers ─────────────────────────────────────────────────

    def enable_blitting(self):
        """Enable blitted rendering for fast updates (playback cursor, live data)."""
        self._blitting_enabled = True
        self._save_blit_bg()

    def disable_blitting(self):
        """Disable blitting and return to normal rendering."""
        self._blitting_enabled = False
        self._blit_bg = None

    def _save_blit_bg(self):
        """Cache the current figure background for blitting."""
        try:
            self._blit_bg = self.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._blit_bg = None

    def _blit_restore(self):
        """Restore cached background and restore animated artists."""
        if self._blit_bg is not None:
            self.restore_region(self._blit_bg)
            # Re-restore animated artists
            if self._playback_line is not None:
                try:
                    self.ax.draw_artist(self._playback_line)
                except Exception:
                    pass
            self.blit(self.fig.bbox)
        else:
            self.draw_idle()

    # ── playback cursor (blitted) ────────────────────────────────────────

    def set_playback_pos(self, time_sec: float, duration: float):
        """Update the playback cursor. Uses blitting when available for speed."""
        if self._playback_line is not None:
            try:
                self._playback_line.remove()
            except Exception:
                pass

        if 0 <= time_sec <= duration:
            self._playback_line = self.ax.axvline(
                time_sec, color=RED, linewidth=1.2, alpha=0.8,
                linestyle="-", animated=self._blitting_enabled,
            )
        else:
            self._playback_line = None

        if self._blitting_enabled and self._blit_bg is not None:
            self._blit_restore()
        else:
            self.draw_idle()

    # ── crosshair & tooltip (in-place update, throttled) ─────────────────

    def _on_press(self, event):
        """Emit seek signal on left-click inside axes."""
        if event.inaxes == self.ax and event.button == 1 and event.xdata is not None:
            self.seek_requested.emit(event.xdata)

    def _on_motion(self, event):
        """Show crosshair and value tooltip. Throttled to ~60 FPS."""
        if event.inaxes != self.ax or event.xdata is None:
            self._clear_crosshair()
            return

        # Throttle: skip if too soon since last update
        now = time.monotonic()
        if now - self._last_motion_time < self._motion_throttle_sec:
            return
        self._last_motion_time = now

        # Update vertical crosshair in-place
        if self._crosshair_v is not None:
            try:
                self._crosshair_v.set_xdata([event.xdata, event.xdata])
            except Exception:
                # Fallback: recreate
                try:
                    self._crosshair_v.remove()
                except Exception:
                    pass
                self._crosshair_v = self.ax.axvline(
                    event.xdata, color=LIGHT_TXT, linewidth=0.4, alpha=0.3, linestyle="--"
                )
        else:
            self._crosshair_v = self.ax.axvline(
                event.xdata, color=LIGHT_TXT, linewidth=0.4, alpha=0.3, linestyle="--"
            )

        # Update horizontal crosshair in-place
        if self._crosshair_h is not None:
            try:
                self._crosshair_h.set_ydata([event.ydata, event.ydata])
            except Exception:
                try:
                    self._crosshair_h.remove()
                except Exception:
                    pass
                self._crosshair_h = self.ax.axhline(
                    event.ydata, color=LIGHT_TXT, linewidth=0.4, alpha=0.3, linestyle="--"
                )
        else:
            self._crosshair_h = self.ax.axhline(
                event.ydata, color=LIGHT_TXT, linewidth=0.4, alpha=0.3, linestyle="--"
            )

        # Update annotation in-place
        if self._annotation is not None:
            try:
                self._annotation.set_text(f"{event.xdata:.3f}, {event.ydata:.2f}")
                self._annotation.xy = (event.xdata, event.ydata)
            except Exception:
                try:
                    self._annotation.remove()
                except Exception:
                    pass
                self._annotation = None

        if self._annotation is None:
            self._annotation = self.ax.annotate(
                f"{event.xdata:.3f}, {event.ydata:.2f}",
                xy=(event.xdata, event.ydata),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=7, color=LIGHT_TXT, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK_AXIS,
                          edgecolor=SURFACE0, alpha=0.9),
            )

        self.cursor_moved.emit(event.xdata, event.ydata)
        self.draw_idle()

    def _on_leave(self, event):
        self._clear_crosshair()

    def _clear_crosshair(self):
        for attr in ("_crosshair_v", "_crosshair_h", "_annotation"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.draw_idle()

    # ── colourbar management ─────────────────────────────────────────────

    def _remove_cbar(self):
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

    # ── convenience for subclasses ───────────────────────────────────────

    def clear_plot(self):
        """Full reset: clear axes, remove cbar, reset playback line."""
        self.disable_blitting()
        self.ax.clear()
        self._remove_cbar()
        self._playback_line = None
        self._clear_crosshair()
