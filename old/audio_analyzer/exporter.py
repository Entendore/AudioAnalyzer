#!/usr/bin/env python3
"""
exporter.py — Export utilities for plots, text reports, JSON, and CSV.

Features:
  - Export current plot as PNG / JPEG / SVG / PDF
  - Export analysis report as plain text or JSON
  - Export raw data (RMS history, band energy, metrics) as CSV
  - Batch-export all tabs as images
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from audio_data import AudioData
from config import APP_NAME, APP_VERSION


class Exporter:
    """Static helper methods for exporting analysis results."""

    # ── single plot image ────────────────────────────────────────────────

    @staticmethod
    def export_plot(parent, canvas: FigureCanvas, default_name: str = "plot"):
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export Plot",
            f"{default_name}_plot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Vector (*.svg);;PDF Document (*.pdf)",
        )
        if not path:
            return
        try:
            canvas.fig.savefig(
                path, dpi=200,
                facecolor=canvas.fig.get_facecolor(),
                bbox_inches="tight",
            )
        except Exception as exc:
            QMessageBox.critical(parent, "Export Error", str(exc))

    # ── batch-export all tabs ────────────────────────────────────────────

    @staticmethod
    def export_all_plots(parent, tab_canvases: dict, default_name: str = "analysis"):
        """Export every canvas in *tab_canvases* {tab_name: canvas} to a folder."""
        folder = QFileDialog.getExistingDirectory(parent, "Select Export Folder")
        if not folder:
            return
        for tab_name, canvas in tab_canvases.items():
            safe = tab_name.replace(" ", "_").replace("/", "_")
            fpath = os.path.join(folder, f"{default_name}_{safe}.png")
            try:
                canvas.fig.savefig(
                    fpath, dpi=200,
                    facecolor=canvas.fig.get_facecolor(),
                    bbox_inches="tight",
                )
            except Exception as exc:
                QMessageBox.warning(parent, "Export Warning", f"Failed to export {tab_name}: {exc}")

    # ── text report ──────────────────────────────────────────────────────

    @staticmethod
    def export_text_report(parent, audio: AudioData, default_name: str = "report"):
        if audio is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export Report",
            f"{default_name}_report.txt",
            "Text File (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{'=' * 60}\n")
                f.write(f"  {APP_NAME} v{APP_VERSION} — Audio Analysis Report\n")
                f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 60}\n\n")

                f.write("[File Information]\n")
                f.write(f"  File:         {audio.file_name}\n")
                f.write(f"  Path:         {audio.file_path}\n")
                f.write(f"  Duration:     {audio.duration:.3f} s\n")
                f.write(f"  Sample Rate:  {audio.sample_rate} Hz\n")
                f.write(f"  Channels:     {audio.channels}\n")
                f.write(f"  Bit Depth:    {audio.bit_depth}\n")
                sz = audio.file_size
                f.write(f"  File Size:    {sz / (1 << 20):.1f} MB\n" if sz > 1 << 20 else f"  File Size:  {sz / 1024:.1f} KB\n")
                f.write("\n")

                f.write("[Level Measurements]\n")
                f.write(f"  Peak:             {audio.peak_db:.2f} dB\n" if audio.peak_db > -np.inf else "  Peak:             -inf dB\n")
                f.write(f"  RMS:              {audio.rms_db:.2f} dB\n" if audio.rms_db > -np.inf else "  RMS:              -inf dB\n")
                f.write(f"  Crest Factor:     {audio.crest_factor:.2f} dB\n")
                f.write(f"  Dynamic Range:    {audio.dynamic_range:.2f} dB\n")
                f.write(f"  Integrated LUFS:  {audio.lufs:.2f}\n" if audio.lufs > -np.inf else "  Integrated LUFS:  N/A\n")
                f.write("\n")

                f.write("[Spectral Metrics]\n")
                f.write(f"  Spectral Centroid:   {audio.spectral_centroid:.1f} Hz\n")
                f.write(f"  Spectral Rolloff:    {audio.spectral_rolloff:.1f} Hz\n")
                f.write(f"  Spectral Flatness:   {audio.spectral_flatness:.4f}\n")
                f.write(f"  Zero Crossing Rate:  {audio.zero_crossing_rate:.1f} Hz\n")
                f.write(f"  Detected Pitch:      {audio.pitch:.1f} Hz\n" if audio.pitch > 0 else "  Detected Pitch:      N/A\n")
                f.write("\n")

                f.write("[Clip Detection]\n")
                f.write(f"  Clip Count:  {audio.clip_count}\n")
                if audio.clips:
                    for i, (s, e) in enumerate(audio.clips[:20]):
                        t0 = s / audio.sample_rate
                        t1 = e / audio.sample_rate
                        f.write(f"  Clip {i + 1}: {t0:.4f}s – {t1:.4f}s\n")
                f.write("\n")

                if audio.channels >= 2:
                    f.write("[Stereo Analysis]\n")
                    f.write(f"  Overall Correlation: {audio.stereo_correlation:.3f}\n")
                    f.write("\n")

                f.write("[Band Energy]\n")
                for band, energy in audio.band_energy.items():
                    f.write(f"  {band.replace(chr(10), ' '):30s}  {energy * 100:.2f}%\n")
        except Exception as exc:
            QMessageBox.critical(parent, "Export Error", str(exc))

    # ── JSON report ──────────────────────────────────────────────────────

    @staticmethod
    def export_json_report(parent, audio: AudioData, default_name: str = "report"):
        if audio is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export JSON",
            f"{default_name}_report.json",
            "JSON File (*.json)",
        )
        if not path:
            return
        try:
            data = audio.to_dict()
            data["generator"] = f"{APP_NAME} v{APP_VERSION}"
            data["timestamp"] = datetime.now().isoformat()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            QMessageBox.critical(parent, "Export Error", str(exc))

    # ── CSV data ─────────────────────────────────────────────────────────

    @staticmethod
    def export_csv(parent, audio: AudioData, default_name: str = "data"):
        if audio is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export CSV",
            f"{default_name}_data.csv",
            "CSV File (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Metadata
                writer.writerow(["Metric", "Value"])
                d = audio.to_dict()
                for k, v in d.items():
                    if k != "band_energy_pct":
                        writer.writerow([k, v])
                writer.writerow([])

                # Band energy
                writer.writerow(["Band", "Energy (%)"])
                for band, pct in d.get("band_energy_pct", {}).items():
                    writer.writerow([band, pct])
                writer.writerow([])

                # RMS history
                if len(audio.rms_history) > 0:
                    writer.writerow(["Time (s)", "RMS Level (dB)"])
                    rms_db = 20.0 * np.log10(audio.rms_history + 1e-10)
                    for t, r in zip(audio.rms_history_times, rms_db):
                        writer.writerow([f"{t:.4f}", f"{r:.2f}"])
                    writer.writerow([])

                # Correlation history
                if len(audio.correlation_history) > 0:
                    writer.writerow(["Time (s)", "Correlation"])
                    for t, c in zip(audio.correlation_history_times, audio.correlation_history):
                        writer.writerow([f"{t:.4f}", f"{c:.3f}"])
        except Exception as exc:
            QMessageBox.critical(parent, "Export Error", str(exc))
