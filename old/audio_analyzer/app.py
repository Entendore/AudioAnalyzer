#!/usr/bin/env python3
"""
app.py — Entry point for the Advanced Audio Analyzer application.
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from config import APP_NAME, APP_ORG, DEFAULT_PLAYBACK_VOL
from main_window import MainWindow
from config import AppSettings


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setFont(QFont("Segoe UI", 10))

    settings = AppSettings()

    # Restore volume
    saved_vol = settings.get("volume", DEFAULT_PLAYBACK_VOL)
    if isinstance(saved_vol, (int, float)):
        # Will be applied after MainWindow reads it
        pass

    win = MainWindow(settings=settings)

    # Apply saved volume
    if isinstance(saved_vol, (int, float)):
        vol_int = int(saved_vol * 100)
        win.vol_slider.setValue(vol_int)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
