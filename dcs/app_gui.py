"""PagerMole DCS — Native Desktop Application.

A high-performance PyQt6 desktop interface for POCSAG and APRS signal encoding,
decoding, and real-time native waveform analysis. Integrates direct in-canvas
Matplotlib visualization for Time-Domain Bit Overlays, Power Spectral Density (PSD),
Decoder Eye Diagrams, Spectrogram Waterfalls, 0x7E Synchronization Overlays, and
Zero-Crossing Histograms.
"""

from __future__ import annotations

import io
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import traceback
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import threading

try:
    import sounddevice as _sd
    _SD_OK = True
except Exception:
    _sd = None  # type: ignore
    _SD_OK = False

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox,
    QTextEdit, QFileDialog, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSplitter, QScrollArea, QFrame,
    QSizePolicy, QMessageBox, QProgressBar, QStatusBar, QButtonGroup,
    QRadioButton, QToolButton,
)

# ── Core DCS imports ────────────────────────────────────────────────
from dcs.generators.pocsag import generate_pocsag_wav, POCSAG_MARK_HZ, POCSAG_SPACE_HZ
from dcs.generators.aprs import generate_aprs_wav
from dcs.analysis.theme import PLOT_THEME
from dcs.analysis.wav_plots import (
    read_wav,
    compute_pocsag_metrics,
    compute_aprs_metrics,
    draw_pocsag_time,
    draw_pocsag_psd,
    draw_pocsag_spectrogram,
    draw_pocsag_all,
    draw_aprs_time,
    draw_aprs_psd,
    draw_aprs_spectrogram,
    draw_aprs_all,
)

# ── Engine binary paths ────────────────────────────────────────────
_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
_BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", _ROOT))


def get_engine_bin(name: str) -> Path:
    """Locate an engine executable across frozen and development directories."""
    candidates = [
        _BUNDLE_ROOT / "engines" / "bin" / name,
        _ROOT / "_internal" / "engines" / "bin" / name,
        _ROOT / "engines" / "bin" / name,
        Path(__file__).resolve().parent.parent / "engines" / "bin" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


MULTIMON_BIN = get_engine_bin("multimon-ng.exe")
DIREWOLF_BIN = get_engine_bin("direwolf.exe")

# ── Pure Monochrome Black & Grey Palette ─────────────────────────────
C_BASE = "#0a0a0a"           # True black base
C_SURFACE = "#111111"        # Cards, side panels
C_PANEL = "#171717"          # Input backgrounds
C_PANEL_ALT = "#1c1c1c"      # Elevated containers
C_HOVER = "#222222"          # Hover state
C_ACTIVE = "#2a2a2a"         # Active / selected
C_BORDER_SUBTLE = "#1a1a1a"  # Hair-line dividers
C_BORDER = "#242424"         # Standard border
C_BORDER_STRONG = "#363636"  # Elevated border on hover

C_TEXT_BRIGHT = "#ebebeb"    # Near-white headings
C_TEXT_NORMAL = "#b8b8b8"    # Body text
C_TEXT_MUTED = "#686868"     # Secondary / captions
C_TEXT_FAINT = "#404040"     # Placeholder ghost text

C_ACCENT_CYAN = "#9a9a9a"    # Repurposed: light grey highlight
C_ACCENT_GREEN = "#888888"   # Repurposed: mid grey
C_ACCENT_AMBER = "#707070"   # Repurposed: dark-mid grey
C_ACCENT_ROSE = "#505050"    # Repurposed: deep grey

GLOBAL_STYLESHEET = f"""
    QMainWindow, QWidget {{
        background-color: {C_BASE};
        color: {C_TEXT_NORMAL};
        font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 13px;
    }}

    /* ── Tabs ── */
    QTabWidget::pane {{
        border: 1px solid {C_BORDER};
        background: {C_BASE};
        border-top: none;
    }}
    QTabBar::tab {{
        background: {C_SURFACE};
        color: {C_TEXT_MUTED};
        padding: 10px 26px;
        margin-right: 2px;
        border: 1px solid {C_BORDER};
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background: {C_BASE};
        color: {C_TEXT_BRIGHT};
        border-bottom: 2px solid {C_BORDER_STRONG};
        font-weight: 700;
    }}
    QTabBar::tab:hover:!selected {{
        background: {C_PANEL};
        color: {C_TEXT_NORMAL};
    }}

    /* ── Cards / GroupBoxes ── */
    QGroupBox {{
        border: 1px solid {C_BORDER};
        border-radius: 6px;
        margin-top: 14px;
        padding: 16px 12px 12px 12px;
        font-weight: 700;
        color: {C_TEXT_BRIGHT};
        background-color: {C_SURFACE};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 10px;
        color: {C_TEXT_NORMAL};
        background-color: transparent;
        font-size: 12px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}

    /* ── Form Inputs ── */
    QLabel {{
        color: {C_TEXT_NORMAL};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {C_PANEL};
        color: {C_TEXT_BRIGHT};
        border: 1px solid {C_BORDER};
        border-radius: 5px;
        padding: 6px 10px;
        selection-background-color: {C_ACTIVE};
        selection-color: {C_TEXT_BRIGHT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {C_BORDER_STRONG};
        background-color: {C_PANEL_ALT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {C_PANEL};
        color: {C_TEXT_BRIGHT};
        border: 1px solid {C_BORDER};
        selection-background-color: {C_HOVER};
        selection-color: {C_TEXT_BRIGHT};
        padding: 4px;
    }}

    /* ── Buttons ── */
    QPushButton {{
        background-color: {C_PANEL};
        color: {C_TEXT_NORMAL};
        border: 1px solid {C_BORDER};
        border-radius: 5px;
        padding: 7px 16px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {C_HOVER};
        border-color: {C_BORDER_STRONG};
        color: {C_TEXT_BRIGHT};
    }}
    QPushButton:pressed {{
        background-color: {C_ACTIVE};
    }}

    /* Primary Generator Action */
    QPushButton#primary {{
        background-color: {C_PANEL_ALT};
        border: 1px solid {C_BORDER_STRONG};
        color: {C_TEXT_BRIGHT};
        font-weight: 700;
    }}
    QPushButton#primary:hover {{
        background-color: {C_HOVER};
        border-color: #4a4a4a;
        color: #ffffff;
    }}
    QPushButton#primary:pressed {{
        background-color: {C_ACTIVE};
    }}

    /* Secondary Action */
    QPushButton#secondary {{
        background-color: {C_SURFACE};
        border: 1px solid {C_BORDER};
        color: {C_TEXT_MUTED};
    }}
    QPushButton#secondary:hover {{
        background-color: {C_PANEL};
        border-color: {C_BORDER_STRONG};
        color: {C_TEXT_NORMAL};
    }}

    /* Graph Segmented Control Buttons */
    QPushButton#view_toggle:checked {{
        background-color: {C_ACTIVE};
        border: 1px solid {C_BORDER_STRONG};
        color: {C_TEXT_BRIGHT};
        font-weight: 700;
    }}
    QPushButton#view_toggle:!checked {{
        background-color: {C_SURFACE};
        border: 1px solid {C_BORDER};
        color: {C_TEXT_MUTED};
        font-weight: 500;
    }}
    QPushButton#view_toggle:!checked:hover {{
        background-color: {C_PANEL};
        border-color: {C_BORDER_STRONG};
        color: {C_TEXT_NORMAL};
    }}

    /* ── Logs & Consoles ── */
    QTextEdit {{
        background-color: {C_BASE};
        color: {C_TEXT_NORMAL};
        border: 1px solid {C_BORDER_SUBTLE};
        border-radius: 5px;
        padding: 8px;
        font-family: 'Consolas', 'Cascadia Code', 'JetBrains Mono', monospace;
        font-size: 12px;
    }}

    /* ── Slim Grey Scrollbars ── */
    QScrollBar:vertical {{
        border: none;
        background: {C_BASE};
        width: 8px;
        margin: 0px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {C_BORDER};
        min-height: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {C_BORDER_STRONG};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        border: none;
        background: {C_BASE};
        height: 8px;
        margin: 0px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {C_BORDER};
        min-width: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {C_BORDER_STRONG};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* ── Splitter & Status Bar ── */
    QSplitter::handle {{
        background: {C_BORDER_SUBTLE};
        width: 2px;
    }}
    QSplitter::handle:hover {{
        background: {C_BORDER_STRONG};
    }}
    QStatusBar {{
        background: {C_SURFACE};
        color: {C_TEXT_MUTED};
        border-top: 1px solid {C_BORDER_SUBTLE};
        font-size: 11px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
"""


# ═══════════════════════════════════════════════════════════════════
# Worker Thread for Non-Blocking DSP & Signal Generation
# ═══════════════════════════════════════════════════════════════════
class WorkerThread(QThread):
    """Generic background worker that emits results or errors safely."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            res = self._fn(*self._args, **self._kwargs)
            self.finished.emit(res)
        except Exception as exc:
            self.error.emit(f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════
# Matplotlib Dark Toolbar Widget
# ═══════════════════════════════════════════════════════════════════
class DarkNavigationToolbar(NavigationToolbar):
    """Monochrome grey Matplotlib navigation toolbar."""
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setStyleSheet(f"""
            QToolBar {{
                background-color: {C_SURFACE};
                border: 1px solid {C_BORDER};
                border-radius: 5px;
                padding: 2px 4px;
                spacing: 3px;
            }}
            QToolButton {{
                background-color: transparent;
                color: {C_TEXT_MUTED};
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: {C_HOVER};
                color: {C_TEXT_BRIGHT};
                border: 1px solid {C_BORDER_STRONG};
            }}
            QToolButton:pressed {{
                background-color: {C_ACTIVE};
            }}
            QLabel {{
                color: {C_TEXT_MUTED};
                font-family: monospace;
                font-size: 11px;
                padding-left: 8px;
            }}
        """)


class SmartCanvas(FigureCanvas):
    """FigureCanvas with always-on left-drag pan and scroll-wheel zoom.

    No toolbar button selection needed — drag anywhere on the plot to pan,
    scroll to zoom in/out centred on the cursor.
    Right-drag or toolbar zoom-box still works normally.
    """

    def __init__(self, fig, coord_callback=None):
        super().__init__(fig)
        self._coord_cb    = coord_callback
        self._drag_active = False
        self._drag_px0    = None   # pixel origin at drag start
        self._drag_xlims  = {}     # ax -> xlim at drag start
        self._drag_ylims  = {}     # ax -> ylim at drag start

        self.mpl_connect("button_press_event",   self._sm_press)
        self.mpl_connect("button_release_event", self._sm_release)
        self.mpl_connect("motion_notify_event",  self._sm_motion)
        self.mpl_connect("scroll_event",         self._sm_scroll)

    # ── drag-to-pan ──────────────────────────────────────────────────
    def _sm_press(self, ev):
        if ev.button != 1 or ev.inaxes is None:
            return
        # Only engage custom pan when toolbar is idle (no zoom/pan mode active)
        try:
            if self._toolbar and self._toolbar.mode != "":
                return
        except AttributeError:
            pass
        self._drag_active = True
        self._drag_px0    = (ev.x, ev.y)
        axes = self.figure.axes
        self._drag_xlims  = {ax: ax.get_xlim() for ax in axes}
        self._drag_ylims  = {ax: ax.get_ylim() for ax in axes}

    def _sm_release(self, ev):
        self._drag_active = False

    def _sm_motion(self, ev):
        # Coordinate readout
        if self._coord_cb and ev.inaxes and ev.xdata is not None:
            self._coord_cb(ev.xdata, ev.ydata)
        elif self._coord_cb:
            self._coord_cb(None, None)

        if not self._drag_active or self._drag_px0 is None:
            return
        dx_px = ev.x - self._drag_px0[0]
        dy_px = ev.y - self._drag_px0[1]

        for ax in self.figure.axes:
            if ax not in self._drag_xlims:
                continue
            x0, x1 = self._drag_xlims[ax]
            y0, y1 = self._drag_ylims[ax]
            bbox = ax.get_window_extent()
            if bbox.width < 1 or bbox.height < 1:
                continue
            xs = (x1 - x0) / bbox.width
            ys = (y1 - y0) / bbox.height
            ax.set_xlim(x0 - dx_px * xs, x1 - dx_px * xs)
            ax.set_ylim(y0 - dy_px * ys, y1 - dy_px * ys)
        self.draw_idle()

    # ── scroll-wheel zoom ────────────────────────────────────────────
    def _sm_scroll(self, ev):
        if ev.inaxes is None:
            return
        ax     = ev.inaxes
        factor = 0.82 if ev.button == "up" else 1.0 / 0.82
        xlim   = ax.get_xlim()
        ylim   = ax.get_ylim()
        xd, yd = ev.xdata, ev.ydata
        ax.set_xlim(xd - (xd - xlim[0]) * factor,
                    xd + (xlim[1] - xd) * factor)
        ax.set_ylim(yd - (yd - ylim[0]) * factor,
                    yd + (ylim[1] - yd) * factor)
        self.draw_idle()


# ═══════════════════════════════════════════════════════════════════
# Tab 1: WAVE GENERATOR & REAL-TIME VISUALIZER
# ═══════════════════════════════════════════════════════════════════
class WaveGeneratorTab(QWidget):
    """Unified Signal Generator & Real-Time Visualization Studio.

    Generates signals, accepts any loaded WAV file, computes actual PCM measurements,
    and visualizes all required POCSAG and APRS graphs directly on native PyQt Matplotlib canvases.
    """

    def __init__(self):
        super().__init__()
        self._active_proto = "POCSAG"   # "POCSAG" or "APRS"
        self._active_wav_bytes: Optional[bytes] = None
        self._active_wav_path: Optional[Path] = None
        self._active_metrics: Optional[Dict[str, Any]] = None
        self._current_view_idx = 3       # Default to Tri-View Overview (All 3)
        self._worker: Optional[WorkerThread] = None

        # Audio playback state
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_stop_evt = threading.Event()
        self._is_playing = False
        self._encoder_expanded = True

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(3)

        # ── Left Panel: Controls & File Management ─────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        ctrl_layout = QVBoxLayout(scroll_content)
        ctrl_layout.setContentsMargins(2, 2, 6, 2)
        ctrl_layout.setSpacing(10)

        # ── ENCODER SECTION HEADER (collapsible) ──────────────────
        encoder_header = QWidget()
        encoder_header.setStyleSheet(f"""
            QWidget {{
                background-color: {C_PANEL_ALT};
                border: 1px solid {C_BORDER};
                border-radius: 5px;
            }}
        """)
        enc_hdr_layout = QHBoxLayout(encoder_header)
        enc_hdr_layout.setContentsMargins(10, 6, 6, 6)
        enc_hdr_layout.setSpacing(6)

        enc_title = QLabel("SIGNAL ENCODING")
        enc_title.setStyleSheet(f"font-weight: 700; font-size: 11px; letter-spacing: 1px; color: {C_TEXT_NORMAL}; background: transparent; border: none;")
        enc_hdr_layout.addWidget(enc_title)
        enc_hdr_layout.addStretch()

        self._enc_collapse_btn = QToolButton()
        self._enc_collapse_btn.setText("▲ Collapse")
        self._enc_collapse_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: none;
                color: {C_TEXT_MUTED};
                font-size: 11px;
                padding: 2px 4px;
            }}
            QToolButton:hover {{ color: {C_TEXT_NORMAL}; }}
        """)
        self._enc_collapse_btn.clicked.connect(self._toggle_encoder_panel)
        enc_hdr_layout.addWidget(self._enc_collapse_btn)
        ctrl_layout.addWidget(encoder_header)

        # ── Encoder container (collapsible body) ───────────────────
        self._encoder_container = QWidget()
        enc_body = QVBoxLayout(self._encoder_container)
        enc_body.setContentsMargins(0, 0, 0, 0)
        enc_body.setSpacing(10)

        # 1. Protocol Selector
        proto_group = QGroupBox("PROTOCOL")
        pg_layout = QVBoxLayout(proto_group)
        self.proto_combo = QComboBox()
        self.proto_combo.addItems(["POCSAG (2-Tone FSK)", "APRS (1200 Baud Bell 202 AFSK)"])
        self.proto_combo.currentIndexChanged.connect(self._on_proto_changed)
        pg_layout.addWidget(self.proto_combo)
        enc_body.addWidget(proto_group)

        # 2. POCSAG Parameters Card
        self.pocsag_group = QGroupBox("POCSAG ENCODER")
        pform = QFormLayout(self.pocsag_group)
        self.pocsag_address = QSpinBox()
        self.pocsag_address.setRange(1, 2097151)
        self.pocsag_address.setValue(1234567)
        pform.addRow("RIC Address:", self.pocsag_address)

        self.pocsag_func = QComboBox()
        self.pocsag_func.addItems(["0 - Tone Only", "1 - Numeric", "2 - Activation", "3 - Alphanumeric"])
        self.pocsag_func.setCurrentIndex(3)
        pform.addRow("Function:", self.pocsag_func)

        self.pocsag_baud = QComboBox()
        self.pocsag_baud.addItems(["512", "1200", "2400"])
        self.pocsag_baud.setCurrentText("1200")
        self.pocsag_baud.currentIndexChanged.connect(self._on_baud_changed)
        pform.addRow("Baud Rate:", self.pocsag_baud)

        self.pocsag_numeric = QCheckBox("Numeric Mode (BCD)")
        pform.addRow("", self.pocsag_numeric)

        self.pocsag_msg = QLineEdit("PagerMole DCS Alert #1")
        pform.addRow("Message:", self.pocsag_msg)
        enc_body.addWidget(self.pocsag_group)

        # 3. APRS Parameters Card
        self.aprs_group = QGroupBox("APRS ENCODER")
        aform = QFormLayout(self.aprs_group)
        self.aprs_callsign = QLineEdit("N0CALL-1")
        aform.addRow("Callsign:", self.aprs_callsign)

        self.aprs_dest = QLineEdit("APRS")
        aform.addRow("Destination:", self.aprs_dest)

        self.aprs_path = QLineEdit("WIDE1-1,WIDE2-1")
        aform.addRow("Digi Path:", self.aprs_path)

        pos_row = QHBoxLayout()
        self.aprs_lat = QDoubleSpinBox()
        self.aprs_lat.setRange(-90.0, 90.0)
        self.aprs_lat.setDecimals(4)
        self.aprs_lat.setValue(37.7749)
        self.aprs_lat.setPrefix("Lat: ")
        pos_row.addWidget(self.aprs_lat)

        self.aprs_lon = QDoubleSpinBox()
        self.aprs_lon.setRange(-180.0, 180.0)
        self.aprs_lon.setDecimals(4)
        self.aprs_lon.setValue(-122.4194)
        self.aprs_lon.setPrefix("Lon: ")
        pos_row.addWidget(self.aprs_lon)

        self.aprs_use_pos = QCheckBox("GPS")
        pos_row.addWidget(self.aprs_use_pos)
        aform.addRow("Position:", pos_row)

        self.aprs_msg = QLineEdit("PagerMole DCS Signal Test")
        aform.addRow("Info:", self.aprs_msg)

        self.aprs_comment = QLineEdit("Real-Time Native DSP")
        aform.addRow("Comment:", self.aprs_comment)

        enc_body.addWidget(self.aprs_group)
        self.aprs_group.hide()

        ctrl_layout.addWidget(self._encoder_container)

        # 4. Action Buttons Card
        actions_group = QGroupBox("SIGNAL ACTIONS & FILES")
        act_layout = QVBoxLayout(actions_group)

        self.gen_btn = QPushButton("⚡  Generate & Visualize")
        self.gen_btn.setObjectName("primary")
        self.gen_btn.setFixedHeight(36)
        self.gen_btn.clicked.connect(self._generate_signal)
        act_layout.addWidget(self.gen_btn)

        # ── Playback Transport Bar ────────────────────────────────
        transport_row = QHBoxLayout()
        transport_row.setSpacing(4)

        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setObjectName("secondary")
        self.play_btn.setFixedHeight(30)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_audio)
        transport_row.addWidget(self.play_btn)

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setObjectName("secondary")
        self.pause_btn.setFixedHeight(30)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._pause_audio)
        transport_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("secondary")
        self.stop_btn.setFixedHeight(30)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_audio)
        transport_row.addWidget(self.stop_btn)

        act_layout.addLayout(transport_row)

        if not _SD_OK:
            no_audio_lbl = QLabel("⚠ sounddevice not available")
            no_audio_lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; padding: 2px 0;")
            act_layout.addWidget(no_audio_lbl)

        file_row = QHBoxLayout()
        self.load_btn = QPushButton("📂  Load WAV...")
        self.load_btn.setObjectName("secondary")
        self.load_btn.clicked.connect(self._browse_load_wav)
        file_row.addWidget(self.load_btn)

        self.save_btn = QPushButton("💾  Save WAV...")
        self.save_btn.clicked.connect(self._save_active_wav)
        self.save_btn.setEnabled(False)
        file_row.addWidget(self.save_btn)
        act_layout.addLayout(file_row)

        quick_row = QHBoxLayout()
        lbl_recent = QLabel("Recent:")
        lbl_recent.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px;")
        quick_row.addWidget(lbl_recent)
        self.recent_combo = QComboBox()
        self.recent_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh_recent_files()
        self.recent_combo.currentIndexChanged.connect(self._on_recent_selected)
        quick_row.addWidget(self.recent_combo)
        act_layout.addLayout(quick_row)

        ctrl_layout.addWidget(actions_group)

        # 5. Signal Characteristics Card
        self.meta_group = QGroupBox("SIGNAL TELEMETRY")
        mform = QFormLayout(self.meta_group)
        self.lbl_meta_proto = QLabel("—")
        self.lbl_meta_proto.setStyleSheet(f"font-weight: 700; color: {C_TEXT_BRIGHT};")
        mform.addRow("Protocol:", self.lbl_meta_proto)

        self.lbl_meta_dur = QLabel("—")
        mform.addRow("Duration:", self.lbl_meta_dur)

        self.lbl_meta_samples = QLabel("—")
        mform.addRow("Samples:", self.lbl_meta_samples)

        self.lbl_meta_peak = QLabel("—")
        mform.addRow("Peak / RMS:", self.lbl_meta_peak)

        self.lbl_meta_source = QLabel("No signal loaded")
        self.lbl_meta_source.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px;")
        mform.addRow("Source:", self.lbl_meta_source)
        ctrl_layout.addWidget(self.meta_group)

        # 6. Activity Console
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(90)
        self.log_text.setPlaceholderText("DSP log stream...")
        ctrl_layout.addWidget(self.log_text)

        ctrl_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(460)

        # ── Right Panel: Graph Studio ────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(5)

        # ── Graph Header Bar ────────────────────────────────────────
        graph_header = QHBoxLayout()
        graph_header.setSpacing(6)

        self.info_banner = QLabel("Generate a signal or load a WAV file to see live graphs.")
        self.info_banner.setStyleSheet(f"""
            background-color: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 5px;
            padding: 6px 12px;
            color: {C_TEXT_MUTED};
            font-size: 11px;
        """)
        self.info_banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        graph_header.addWidget(self.info_banner, 1)

        self.analyze_wav_btn = QPushButton("📂  Analyze WAV File...")
        self.analyze_wav_btn.setObjectName("secondary")
        self.analyze_wav_btn.setFixedHeight(30)
        self.analyze_wav_btn.setToolTip("Load any WAV file from disk and render its graphs")
        self.analyze_wav_btn.clicked.connect(self._browse_load_wav)
        graph_header.addWidget(self.analyze_wav_btn)

        self.export_png_btn = QPushButton("📷  Export PNG")
        self.export_png_btn.setObjectName("secondary")
        self.export_png_btn.setFixedHeight(30)
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.clicked.connect(self._export_plot_png)
        graph_header.addWidget(self.export_png_btn)

        right_layout.addLayout(graph_header)

        # ── Hint strip ──────────────────────────────────────────────
        hint = QLabel(
            "  🖱 Scroll to zoom  ·  🔲 Pan tool to drag  ·  🏠 Home to reset  ·  "
            "Graphs update live on every Generate"
        )
        hint.setStyleSheet(
            f"color: {C_TEXT_FAINT}; font-size: 10px; padding: 2px 4px;"
        )
        right_layout.addWidget(hint)

        # ── Per-graph Tab Widget ─────────────────────────────────────
        self.graph_tabs = QTabWidget()
        self.graph_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {C_BORDER};
                background: {C_BASE};
                border-top: none;
            }}
            QTabBar::tab {{
                background: {C_SURFACE};
                color: {C_TEXT_MUTED};
                padding: 6px 16px;
                margin-right: 2px;
                border: 1px solid {C_BORDER};
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 12px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {C_BASE};
                color: {C_TEXT_BRIGHT};
                border-bottom: 2px solid {C_BORDER_STRONG};
                font-weight: 700;
            }}
            QTabBar::tab:hover:!selected {{
                background: {C_PANEL};
                color: {C_TEXT_NORMAL};
            }}
        """)
        self.graph_tabs.currentChanged.connect(self._on_graph_tab_changed)

        # Build 4 canvas tabs (3 individual + 1 overview)
        self._graph_figs: List[Figure] = []
        self._graph_canvases: List[SmartCanvas] = []
        self._graph_toolbars: List[DarkNavigationToolbar] = []

        # Both protocols share the same 4 tab labels
        self._tab_labels = [
            "1. Time Overlay", "2. PSD Spectrum", "3. Spectrogram", "\u229e All Views"
        ]

        _fig_sizes = [
            (10, 5.5),  # time overlay
            (9,  5.0),  # psd
            (9,  5.0),  # spectrogram
            (11, 7.0),  # tri-view
        ]

        for i in range(4):
            with plt.rc_context(PLOT_THEME):
                fig = Figure(figsize=_fig_sizes[i], dpi=100)
                fig.set_facecolor(PLOT_THEME["figure.facecolor"])

            canvas = SmartCanvas(fig, coord_callback=self._on_coord_update)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumHeight(360)

            toolbar = DarkNavigationToolbar(canvas)
            # Give SmartCanvas a reference to the toolbar so it can check mode
            canvas._toolbar = toolbar

            tab_widget = QWidget()
            tab_vbox = QVBoxLayout(tab_widget)
            tab_vbox.setContentsMargins(0, 0, 0, 0)
            tab_vbox.setSpacing(0)
            tab_vbox.addWidget(toolbar)
            tab_vbox.addWidget(canvas, 1)

            self._graph_figs.append(fig)
            self._graph_canvases.append(canvas)
            self._graph_toolbars.append(toolbar)
            self.graph_tabs.addTab(tab_widget, self._tab_labels[i])

        right_layout.addWidget(self.graph_tabs, 1)

        # Coordinates readout
        self.coord_label = QLabel("Cursor: —")
        self.coord_label.setStyleSheet(
            f"color: {C_TEXT_MUTED}; font-family: monospace; font-size: 11px; padding: 2px 6px;"
        )
        right_layout.addWidget(self.coord_label)

        # ── Assemble Splitter ────────────────────────────────────────
        self._splitter.addWidget(left_widget)
        self._splitter.addWidget(right_widget)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([370, 870])

        main_layout.addWidget(self._splitter)

        # Auto-generate initial default signal so user sees real results immediately
        self._generate_signal(auto_initial=True)

    # ── Graph tab / view helpers ─────────────────────────────────────

    def _on_graph_tab_changed(self, tab_idx: int):
        self._current_view_idx = tab_idx
        self._update_info_banner()

    def _update_tab_labels(self):
        """Both protocols now share the same 4 tab labels."""
        for i, lbl in enumerate(self._tab_labels):
            self.graph_tabs.setTabText(i, lbl)

    def _on_coord_update(self, x, y):
        """Callback from SmartCanvas when mouse moves over axes."""
        if x is not None:
            self.coord_label.setText(f"X: {x:10.3f}  |  Y: {y:10.3f}")
        else:
            self.coord_label.setText("Cursor: —")

    def _update_info_banner(self):
        """Update the description banner for the currently active view."""
        if not self._active_metrics:
            self.info_banner.setText(
                "Generate a signal or load any WAV file to visualize measurements."
            )
            return

        source = self._active_wav_path.name if self._active_wav_path else "Generated Signal"
        idx    = self._current_view_idx
        proto  = self._active_proto

        if proto == "POCSAG":
            baud = self._active_metrics.get("baud", 1200)
            msgs = [
                f"POCSAG • Time Overlay — {baud} Baud CPFSK • "
                f"Full bit stream (lavender) + coloured FSK audio (blue=mark / amber=space) • "
                f"Drag to pan, scroll to zoom • {source}",
                f"POCSAG • PSD Spectrum — Welch FFT • "
                f"Mark lobe at {2400} Hz (blue) and Space lobe at {1400} Hz (amber) • {source}",
                f"POCSAG • Spectrogram — Short-Time FFT Waterfall • "
                f"Visualises FSK frequency switching over the full packet duration • {source}",
                f"POCSAG • All Views — {baud} Baud • "
                f"Time Overlay + PSD Spectrum + Spectrogram • {source}",
            ]
        else:
            msgs = [
                f"APRS • Time Overlay — Bell 202 AFSK 1200 Baud • "
                f"Full audio waveform + instantaneous frequency track (teal) • "
                f"Drag to pan, scroll to zoom • {source}",
                f"APRS • PSD Spectrum — Welch FFT • "
                f"Mark lobe at 1200 Hz (blue) and Space lobe at 2200 Hz (amber) • {source}",
                f"APRS • Spectrogram — Full Packet Waterfall • "
                f"Preamble flags, data payload and FCS visible as frequency shifts • {source}",
                f"APRS • All Views — Bell 202 • "
                f"Time Overlay + PSD Spectrum + Spectrogram • {source}",
            ]

        self.info_banner.setText(msgs[min(idx, 3)])

    def _on_proto_changed(self, idx: int):
        self._active_proto = "POCSAG" if idx == 0 else "APRS"
        if self._active_proto == "POCSAG":
            self.pocsag_group.show()
            self.aprs_group.hide()
        else:
            self.pocsag_group.hide()
            self.aprs_group.show()
        self._update_tab_labels()
        if self._active_wav_bytes:
            self._process_active_wav(self._active_wav_bytes, self._active_wav_path)

    def _on_baud_changed(self):
        if self._active_proto == "POCSAG" and self._active_wav_bytes:
            self._process_active_wav(self._active_wav_bytes, self._active_wav_path)

    def _generate_signal(self, auto_initial: bool = False):
        """Generate a real POCSAG or APRS signal and update visualization."""
        self.gen_btn.setEnabled(False)
        if not auto_initial:
            self.log_text.append(f"⚡ Generating {self._active_proto} signal...")

        if self._active_proto == "POCSAG":
            baud = int(self.pocsag_baud.currentText())
            self._worker = WorkerThread(
                generate_pocsag_wav,
                address=self.pocsag_address.value(),
                message=self.pocsag_msg.text() or "PagerMole DCS",
                function=self.pocsag_func.currentIndex(),
                baud=baud,
                is_numeric=self.pocsag_numeric.isChecked(),
            )
        else:
            paths = [p.strip() for p in self.aprs_path.text().split(",") if p.strip()]
            kwargs = dict(
                source=self.aprs_callsign.text() or "N0CALL-1",
                destination=self.aprs_dest.text() or "APRS",
                path=paths or None,
                info=self.aprs_msg.text() or "PagerMole DCS",
                comment=self.aprs_comment.text() or "",
            )
            if self.aprs_use_pos.isChecked():
                kwargs["latitude"] = self.aprs_lat.value()
                kwargs["longitude"] = self.aprs_lon.value()
            self._worker = WorkerThread(generate_aprs_wav, **kwargs)

        self._worker.finished.connect(lambda wav_bytes: self._on_generate_done(wav_bytes, auto_initial))
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_generate_done(self, wav_bytes: bytes, auto_initial: bool):
        self.gen_btn.setEnabled(True)
        gen_dir = _ROOT / "data" / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self._active_proto.lower()}_{uuid.uuid4().hex[:10]}.wav"
        save_path = gen_dir / filename
        save_path.write_bytes(wav_bytes)

        if not auto_initial:
            self.log_text.append(f"✓ Generated {len(wav_bytes) / 1024:.1f} KB WAV → {filename}")

        self._process_active_wav(wav_bytes, save_path)
        self._refresh_recent_files()

    def _process_active_wav(self, wav_bytes: bytes, wav_path: Optional[Path]):
        """Compute DSP metrics from WAV bytes and trigger native rendering."""
        self._active_wav_bytes = wav_bytes
        self._active_wav_path = wav_path
        self.save_btn.setEnabled(True)
        self.export_png_btn.setEnabled(True)
        # Enable transport once we have a signal
        if _SD_OK:
            self.play_btn.setEnabled(True)

        baud = int(self.pocsag_baud.currentText()) if self._active_proto == "POCSAG" else 1200
        proto = self._active_proto

        def dsp_job():
            sr, samples = read_wav(wav_bytes)
            if proto == "POCSAG":
                metrics = compute_pocsag_metrics(samples, sr, baud)
            else:
                metrics = compute_aprs_metrics(samples, sr)
            return metrics

        self._worker = WorkerThread(dsp_job)
        self._worker.finished.connect(self._on_dsp_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_dsp_done(self, metrics: Dict[str, Any]):
        self._active_metrics = metrics

        dur = metrics["duration"]
        samples_len = len(metrics["audio"])
        peak = metrics["peak"]
        rms = metrics["rms"]
        sr = metrics["sample_rate"]
        source_name = self._active_wav_path.name if self._active_wav_path else "Memory Buffer"

        self.lbl_meta_proto.setText(f"{self._active_proto} ({metrics.get('baud', 1200)} Baud)")
        self.lbl_meta_dur.setText(f"{dur:.3f} s  ({sr} Hz)")
        self.lbl_meta_samples.setText(f"{samples_len:,} samples")
        self.lbl_meta_peak.setText(f"{peak:.3f}  /  {rms:.3f}")
        self.lbl_meta_source.setText(source_name)

        self._update_info_banner()
        self._render_current_view()

    def _render_current_view(self):
        """Render all 4 graph tabs from the active DSP metrics simultaneously."""
        if not self._active_metrics:
            return

        m     = self._active_metrics
        label = self._active_wav_path.name if self._active_wav_path else "Active Signal"
        proto = self._active_proto

        with plt.rc_context(PLOT_THEME):
            if proto == "POCSAG":
                # Tab 0: Time Overlay (bit stream + audio)
                fig0 = self._graph_figs[0]
                fig0.clear()
                gs = fig0.add_gridspec(
                    2, 1, height_ratios=[1.0, 2.2],
                    hspace=0.10, left=0.11, right=0.97, top=0.92, bottom=0.09
                )
                ax_bits  = fig0.add_subplot(gs[0])
                ax_audio = fig0.add_subplot(gs[1], sharex=ax_bits)
                draw_pocsag_time(ax_bits, ax_audio, m, label)
                plt.setp(ax_bits.get_xticklabels(), visible=False)

                # Tab 1: PSD
                fig1 = self._graph_figs[1]
                fig1.clear()
                ax1 = fig1.add_subplot(111)
                draw_pocsag_psd(ax1, m, label)
                try:
                    fig1.tight_layout(pad=1.4)
                except Exception:
                    pass

                # Tab 2: Spectrogram
                fig2 = self._graph_figs[2]
                fig2.clear()
                ax2 = fig2.add_subplot(111)
                draw_pocsag_spectrogram(ax2, m, label, fig=fig2)
                try:
                    fig2.tight_layout(pad=1.4)
                except Exception:
                    pass

                # Tab 3: All Views
                draw_pocsag_all(self._graph_figs[3], m, label)

            else:  # APRS
                # Tab 0: Time Overlay (audio + inst-freq track)
                fig0 = self._graph_figs[0]
                fig0.clear()
                gs = fig0.add_gridspec(
                    2, 1, height_ratios=[1.0, 1.4],
                    hspace=0.10, left=0.11, right=0.97, top=0.92, bottom=0.09
                )
                ax_audio = fig0.add_subplot(gs[0])
                ax_freq  = fig0.add_subplot(gs[1], sharex=ax_audio)
                draw_aprs_time(ax_audio, ax_freq, m, label)
                plt.setp(ax_audio.get_xticklabels(), visible=False)

                # Tab 1: PSD
                fig1 = self._graph_figs[1]
                fig1.clear()
                ax1 = fig1.add_subplot(111)
                draw_aprs_psd(ax1, m, label)
                try:
                    fig1.tight_layout(pad=1.4)
                except Exception:
                    pass

                # Tab 2: Spectrogram
                fig2 = self._graph_figs[2]
                fig2.clear()
                ax2 = fig2.add_subplot(111)
                draw_aprs_spectrogram(ax2, m, label, fig=fig2)
                try:
                    fig2.tight_layout(pad=1.4)
                except Exception:
                    pass

                # Tab 3: All Views
                draw_aprs_all(self._graph_figs[3], m, label)

        # Flush all 4 canvases
        for canvas in self._graph_canvases:
            canvas.draw_idle()

    def _browse_load_wav(self):
        """Load any user-specified WAV file for real-time analysis."""
        init_dir = str(_ROOT / "data" / "generated")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV Audio File", init_dir,
            "WAV Audio (*.wav);;All Files (*)"
        )
        if not path:
            return

        p = Path(path)
        try:
            raw_bytes = p.read_bytes()
            lower = p.name.lower()
            if "aprs" in lower:
                self.proto_combo.setCurrentIndex(1)
            elif "pocsag" in lower:
                self.proto_combo.setCurrentIndex(0)

            self.log_text.append(f"📂 Loaded: {p.name} ({len(raw_bytes) / 1024:.1f} KB)")
            self._process_active_wav(raw_bytes, p)
        except Exception as exc:
            self.log_text.append(f"✗ Failed to open {p.name}: {exc}")

    def _on_recent_selected(self, idx: int):
        if idx <= 0:
            return
        filename = self.recent_combo.currentText()
        filepath = _ROOT / "data" / "generated" / filename
        if filepath.exists():
            try:
                raw_bytes = filepath.read_bytes()
                lower = filename.lower()
                if "aprs" in lower:
                    self.proto_combo.setCurrentIndex(1)
                elif "pocsag" in lower:
                    self.proto_combo.setCurrentIndex(0)
                self.log_text.append(f"📂 Selected: {filename}")
                self._process_active_wav(raw_bytes, filepath)
            except Exception as exc:
                self.log_text.append(f"✗ Error loading {filename}: {exc}")

    def _refresh_recent_files(self):
        """Populate the recent files dropdown with WAVs from data/generated."""
        gen_dir = _ROOT / "data" / "generated"
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        self.recent_combo.addItem("(Select generated WAV...)")
        if gen_dir.exists():
            files = sorted(gen_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
            for f in files[:20]:
                self.recent_combo.addItem(f.name)
        self.recent_combo.blockSignals(False)

    def _save_active_wav(self):
        """Save the currently active WAV audio bytes to disk."""
        if not self._active_wav_bytes:
            return
        proto_slug = self._active_proto.lower()
        default_name = f"pagermole_{proto_slug}_signal.wav"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Signal WAV", default_name, "WAV Audio (*.wav);;All Files (*)"
        )
        if path:
            Path(path).write_bytes(self._active_wav_bytes)
            self.log_text.append(f"✓ Saved WAV: {path}")

    def _export_plot_png(self):
        """Export the active tab's Matplotlib figure to a high-res PNG/SVG/PDF."""
        proto_slug = self._active_proto.lower()
        view_names = (
            ["time_overlay", "psd", "eye", "triview"]
            if self._active_proto == "POCSAG"
            else ["spectrogram", "sync", "zero_crossings", "triview"]
        )
        tab_idx = self.graph_tabs.currentIndex()
        default_name = f"pagermole_{proto_slug}_{view_names[tab_idx]}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph to PNG", default_name,
            "PNG Image (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)"
        )
        if path:
            fig = self._graph_figs[tab_idx]
            fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
            self.log_text.append(f"✓ Graph exported: {path}")

    def _on_worker_error(self, err_msg: str):
        self.gen_btn.setEnabled(True)
        self.log_text.append(f"✗ DSP Error: {err_msg[:120]}")

    # ── Encoder Panel Collapse / Expand ──────────────────────────────

    def _toggle_encoder_panel(self):
        """Collapse or expand the signal encoding section."""
        self._encoder_expanded = not self._encoder_expanded
        self._encoder_container.setVisible(self._encoder_expanded)
        if self._encoder_expanded:
            self._enc_collapse_btn.setText("▲ Collapse")
            # Restore left panel to its previous width
            sizes = self._splitter.sizes()
            if sizes[0] < 300:
                self._splitter.setSizes([370, sum(sizes) - 370])
        else:
            self._enc_collapse_btn.setText("▼ Expand")
            # Shrink left panel significantly when encoder is hidden
            total = sum(self._splitter.sizes())
            self._splitter.setSizes([180, total - 180])

    # ── Audio Playback Transport ──────────────────────────────────────

    def _set_transport_state(self, playing: bool, has_signal: bool):
        """Update transport button enabled/label state."""
        self._is_playing = playing
        self.play_btn.setEnabled(has_signal and not playing and _SD_OK)
        self.pause_btn.setEnabled(playing and _SD_OK)
        self.stop_btn.setEnabled((playing or self._playback_thread is not None) and _SD_OK)
        self.play_btn.setText("▶  Play" if not playing else "▶  Playing...")

    def _play_audio(self):
        """Start playback of the active WAV in a background thread."""
        if not _SD_OK or not self._active_wav_bytes:
            return
        self._stop_audio()   # stop any current playback first

        self._playback_stop_evt.clear()
        self._playback_pause_evt = threading.Event()
        self._playback_pause_evt.set()   # not paused initially

        wav_bytes = self._active_wav_bytes

        def _stream():
            import io, wave as _wave
            import numpy as np
            try:
                with _wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    raw = wf.readframes(wf.getnframes())

                if width == 2:
                    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                elif width == 1:
                    samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

                if channels > 1:
                    samples = samples.reshape(-1, channels)

                chunk = max(sr // 20, 512)   # ~50 ms chunks
                idx = 0
                with _sd.OutputStream(samplerate=sr, channels=channels if channels > 1 else 1,
                                      dtype="float32") as stream:
                    flat = samples.flatten() if channels == 1 else samples
                    total = len(flat) if channels == 1 else len(samples)
                    while idx < total and not self._playback_stop_evt.is_set():
                        self._playback_pause_evt.wait()   # blocks when paused
                        if self._playback_stop_evt.is_set():
                            break
                        end = min(idx + chunk, total)
                        block = flat[idx:end] if channels == 1 else samples[idx:end]
                        stream.write(block.reshape(-1, 1) if channels == 1 else block)
                        idx = end
            except Exception as exc:
                self.log_text.append(f"⚠ Playback error: {exc}")
            finally:
                self._playback_thread = None
                # Update UI from main thread
                from PyQt6.QtCore import QMetaObject, Qt as _Qt
                QMetaObject.invokeMethod(
                    self, "_on_playback_ended", _Qt.ConnectionType.QueuedConnection
                )

        self._playback_thread = threading.Thread(target=_stream, daemon=True)
        self._playback_thread.start()
        self._set_transport_state(True, True)
        self.log_text.append("▶ Playback started")

    def _pause_audio(self):
        """Toggle pause / resume of current playback."""
        if not hasattr(self, "_playback_pause_evt") or self._playback_pause_evt is None:
            return
        if self._playback_pause_evt.is_set():
            self._playback_pause_evt.clear()   # pause
            self.pause_btn.setText("▶  Resume")
            self.log_text.append("⏸ Paused")
        else:
            self._playback_pause_evt.set()     # resume
            self.pause_btn.setText("⏸  Pause")
            self.log_text.append("▶ Resumed")

    def _stop_audio(self):
        """Stop any active playback immediately."""
        self._playback_stop_evt.set()
        if hasattr(self, "_playback_pause_evt") and self._playback_pause_evt is not None:
            self._playback_pause_evt.set()   # unblock paused thread so it can exit
        self._playback_thread = None
        self._on_playback_ended()

    @pyqtSlot()
    def _on_playback_ended(self):
        """Called (possibly from main thread via invokeMethod) when stream finishes."""
        has_signal = self._active_wav_bytes is not None
        self._set_transport_state(False, has_signal)
        self.pause_btn.setText("⏸  Pause")
        self.stop_btn.setEnabled(False)


# ═══════════════════════════════════════════════════════════════════
# Tab 2: DECODER
# ═══════════════════════════════════════════════════════════════════
class DecodeTab(QWidget):
    """Decoding studio for WAV audio with clean parsing."""

    send_to_visualizer = pyqtSignal(str)

    # ── multimon-ng noise lines to strip ──────────────────────────────
    _MM_SKIP = (
        "multimon-ng", "(C) 1996", "(C) 2012", "Available demodulators",
        "Enabled demodulators", "read:", "signal caught", "sample rate",
    )
    # ── Dire Wolf noise lines to strip ────────────────────────────────
    _DW_SKIP = (
        "Dire Wolf", "Warning:", "Reading config", "Could not open",
        "Available audio", "Channel 0:", "Note: PTT", "When using VOX",
        "ensure that it", "between the time", "For example,", "SignaLink",
        "Using VOX", "half second", "When using APRS", "a quiet carrier",
        "after your packet", "if using VOX", "frustration and",
        "which you will probably", "Read the User Guide",
        "Ready to accept", "End of file", "deviceid_decode",
        "audio level =", "Audio input level", "Solution is to",
        "Setting audio input", "will provide good",
        "*=selected", "stdin", "Headset", "Microphone", "Speaker",
        "Headphone", "Use of", "Tell the sender",
        "https://", "Tx AX.25", "baud, AFSK",
    )

    def __init__(self):
        super().__init__()
        self._clean_html: str = ""
        self._worker: Optional[WorkerThread] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── File selection ─────────────────────────────────────────
        file_group = QGroupBox("INPUT AUDIO WAV FILE")
        fg = QHBoxLayout(file_group)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select a WAV file to decode…")
        self.file_path.setReadOnly(True)
        fg.addWidget(self.file_path)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)
        fg.addWidget(browse_btn)
        layout.addWidget(file_group)

        # ── Decoder settings ─────────────────────────────────────
        settings_group = QGroupBox("DECODER CONFIGURATION")
        sg = QFormLayout(settings_group)

        self.decoder_combo = QComboBox()
        self.decoder_combo.addItems(["POCSAG", "APRS"])
        self.decoder_combo.currentIndexChanged.connect(self._on_decoder_change)
        sg.addRow("Protocol:", self.decoder_combo)

        self.demod_combo = QComboBox()
        self.demod_combo.addItems([
            "Auto-Detect (All Baud Rates)", "POCSAG1200", "POCSAG512", "POCSAG2400"
        ])
        sg.addRow("Baud Rate / Demodulator:", self.demod_combo)
        layout.addWidget(settings_group)

        # ── Action buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.decode_btn = QPushButton("🔍  Decode Signal")
        self.decode_btn.setObjectName("primary")
        self.decode_btn.clicked.connect(self._decode)
        self.decode_btn.setEnabled(False)
        btn_row.addWidget(self.decode_btn)

        self.visualize_btn = QPushButton("📊  Visualize in Wave Generator")
        self.visualize_btn.setObjectName("secondary")
        self.visualize_btn.clicked.connect(self._open_in_generator)
        self.visualize_btn.setEnabled(False)
        btn_row.addWidget(self.visualize_btn)

        self.copy_btn = QPushButton("📋  Copy Text")
        self.copy_btn.setObjectName("secondary")
        self.copy_btn.clicked.connect(self._copy_output)
        self.copy_btn.setEnabled(False)
        btn_row.addWidget(self.copy_btn)

        self.clear_btn = QPushButton("🗑  Clear")
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Results output ────────────────────────────────────────
        results_group = QGroupBox("DECODED OUTPUT")
        rg = QVBoxLayout(results_group)
        self.results_output = QTextEdit()
        self.results_output.setReadOnly(True)
        self.results_output.setPlaceholderText("Select a WAV file and click Decode Signal to process…")
        self.results_output.setStyleSheet(
            f"font-family: 'Consolas', monospace; font-size: 12px; line-height: 1.5; "
            f"background-color: {C_BASE}; color: {C_TEXT_NORMAL}; border: 1px solid {C_BORDER};"
        )
        rg.addWidget(self.results_output)
        layout.addWidget(results_group, 1)

    # ── UI helpers ────────────────────────────────────────────────
    def _on_decoder_change(self, idx: int):
        self.demod_combo.clear()
        if idx == 0:
            self.demod_combo.addItems([
                "Auto-Detect (All Baud Rates)", "POCSAG1200", "POCSAG512", "POCSAG2400"
            ])
        else:
            self.demod_combo.addItems(["AX.25 / Bell 202 — 1200 Baud"])

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", str(_ROOT / "data" / "generated"),
            "WAV Audio (*.wav *.pcm *.raw);;All Files (*)"
        )
        if path:
            self.file_path.setText(path)
            self.decode_btn.setEnabled(True)
            self.visualize_btn.setEnabled(True)
            fname = Path(path).name.lower()
            if "aprs" in fname and self.decoder_combo.currentIndex() == 0:
                self.decoder_combo.setCurrentIndex(1)
            elif "pocsag" in fname and self.decoder_combo.currentIndex() == 1:
                self.decoder_combo.setCurrentIndex(0)

    def _clear(self):
        self._clean_html = ""
        self.copy_btn.setEnabled(False)
        self.results_output.clear()

    def _copy_output(self):
        text = self.results_output.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)

    def _open_in_generator(self):
        path = self.file_path.text()
        if path and Path(path).exists():
            self.send_to_visualizer.emit(path)

    # ── Decode dispatch ───────────────────────────────────────────
    def _decode(self):
        fpath = self.file_path.text()
        if not fpath or not Path(fpath).exists():
            self._show_error("No valid file selected.")
            return

        self.decode_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.results_output.setHtml(self._html_info(
            f"⚡ Decoding signal in <b>{Path(fpath).name}</b>…"
        ))

        decoder_idx = self.decoder_combo.currentIndex()
        demod = self.demod_combo.currentText()

        if decoder_idx == 0:
            self._worker = WorkerThread(self._run_multimon, fpath, demod)
        else:
            self._worker = WorkerThread(self._run_direwolf, fpath)

        self._worker.finished.connect(self._on_decode_done)
        self._worker.error.connect(self._on_decode_error)
        self._worker.start()

    # ── multimon execution & parsing ──────────────────────────────
    def _run_multimon(self, wav_path: str, demod_choice: str) -> dict:
        if not MULTIMON_BIN.exists():
            return {
                "error": "POCSAG decoder component is not available on this system.",
            }

        fname = Path(wav_path).name.lower()
        if "aprs" in fname:
            return {
                "mismatch": "aprs_in_pocsag",
                "file": Path(wav_path).name,
            }

        try:
            with wave.open(wav_path, "rb") as wf:
                pcm_raw = wf.readframes(wf.getnframes())
                sr = wf.getframerate()
        except Exception as e:
            return {
                "error": f"Unable to read WAV file: {e}",
            }

        demods = ["POCSAG512", "POCSAG1200", "POCSAG2400"]

        def run_mm_with_pcm(pcm_bytes: bytes) -> tuple[str, str]:
            cmd = [str(MULTIMON_BIN), "-t", "raw"]
            for d in demods:
                cmd.extend(["-a", d])
            cmd.append("-")
            res = subprocess.run(cmd, input=pcm_bytes, capture_output=True, timeout=30)
            stdo = res.stdout.decode("utf-8", errors="replace")
            stde = res.stderr.decode("utf-8", errors="replace")
            return stdo, stde

        stdout, stderr = run_mm_with_pcm(pcm_raw)
        raw_combined = stdout + "\n" + stderr
        parsed = self._parse_multimon(raw_combined, demod_choice)

        # Fallback for audio-tone 2-FSK signals (1400/2400 Hz):
        if not parsed.get("frames"):
            try:
                import numpy as np
                import scipy.signal as sps
                import struct

                samples = np.frombuffer(pcm_raw, dtype=np.int16).astype(np.float32)
                if len(samples) > 256:
                    analytic = sps.hilbert(samples)
                    inst_phase = np.unwrap(np.angle(analytic))
                    inst_freq = np.diff(inst_phase) / (2.0 * math.pi) * sr
                    baseband = inst_freq - 1900.0
                    disc = np.clip(baseband * (16000.0 / 500.0), -20000, 20000).astype(np.int16)
                    disc_pcm = struct.pack(f"<{len(disc)}h", *disc)
                    stdo2, stde2 = run_mm_with_pcm(disc_pcm)
                    raw_combined2 = stdo2 + "\n" + stde2
                    parsed2 = self._parse_multimon(raw_combined2, demod_choice)
                    if parsed2.get("frames"):
                        parsed = parsed2
            except Exception:
                pass

        return {
            "proto": "POCSAG",
            "demod": demod_choice,
            "decoded": parsed,
            "file": Path(wav_path).name,
        }

    def _parse_multimon(self, raw: str, demod_choice: str) -> dict:
        """Extract only clean decoded POCSAG frames, detecting address, function, and message."""
        frames = []
        wrong_baud_hits = {}

        pattern = re.compile(
            r'^(POCSAG\d+):\s+Address:\s+(\d+)\s+Function:\s+(\d+)(?:\s+(Alpha|Numeric):\s*(.*))?$',
            re.IGNORECASE
        )

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(skip) or skip in line for skip in self._MM_SKIP):
                continue

            m = pattern.match(line)
            if m:
                baud_tag = m.group(1).upper()
                address = m.group(2)
                function = m.group(3)
                mode = m.group(4) or "Tone"
                message = (m.group(5) or "").strip()

                if demod_choice.startswith("POCSAG") and baud_tag != demod_choice:
                    wrong_baud_hits[baud_tag] = wrong_baud_hits.get(baud_tag, 0) + 1

                frames.append({
                    "baud": baud_tag,
                    "address": address,
                    "function": function,
                    "mode": mode,
                    "message": message,
                })
            elif any(line.startswith(p) for p in ("POCSAG512:", "POCSAG1200:", "POCSAG2400:")):
                frames.append({
                    "baud": line.split(":")[0],
                    "address": "Unknown",
                    "function": "0",
                    "mode": "Raw",
                    "message": line.split(":", 1)[1].strip(),
                })

        return {"frames": frames, "wrong_baud": wrong_baud_hits}

    # ── APRS execution & parsing ──────────────────────────────────
    def _run_direwolf(self, wav_path: str) -> dict:
        if not DIREWOLF_BIN.exists():
            return {
                "error": "APRS decoder component is not available on this system.",
            }

        fname = Path(wav_path).name.lower()
        if "pocsag" in fname:
            return {
                "mismatch": "pocsag_in_aprs",
                "file": Path(wav_path).name,
            }

        tmp_conf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False, prefix="dw_decode_"
        )
        tmp_conf.write("ADEVICE stdin none\nCHANNEL 0\nMODEM 1200\n")
        tmp_conf.close()

        try:
            with wave.open(wav_path, "rb") as wf:
                raw_pcm = wf.readframes(wf.getnframes())

            cmd = [str(DIREWOLF_BIN), "-c", tmp_conf.name, "-r", "22050", "-"]
            result = subprocess.run(
                cmd, input=raw_pcm, capture_output=True, text=False, timeout=30
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            raw_combined = stdout + "\n" + stderr
            packets = self._parse_direwolf(raw_combined)
            return {
                "proto": "APRS",
                "decoded": packets,
                "file": Path(wav_path).name,
            }
        except Exception as e:
            return {
                "error": f"APRS decode failed: {e}",
            }
        finally:
            try:
                os.unlink(tmp_conf.name)
            except OSError:
                pass

    def _parse_direwolf(self, raw: str) -> list:
        """Extract decoded APRS packets from output, stripping all noise."""
        packets = []
        lines = raw.splitlines()
        i = 0
        frame_pattern = re.compile(r'^(?:\[[\d\.]+\]\s*)?([A-Za-z0-9\-]+)>([^:]+):(.*)$')

        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            if any(skip in line for skip in self._DW_SKIP):
                continue

            m = frame_pattern.match(line)
            if m:
                source = m.group(1)
                dest_path = m.group(2)
                payload = m.group(3)
                details = []
                coords = ""

                while i < len(lines):
                    nxt = lines[i].strip()
                    i += 1
                    if not nxt:
                        break
                    if any(skip in nxt for skip in self._DW_SKIP):
                        continue
                    if frame_pattern.match(nxt):
                        i -= 1
                        break
                    if ("N " in nxt or "S " in nxt) and ("W " in nxt or "E " in nxt) and "," in nxt:
                        coords = nxt
                    else:
                        details.append(nxt)

                dest_parts = dest_path.split(",")
                dest = dest_parts[0]
                path = ", ".join(dest_parts[1:]) if len(dest_parts) > 1 else "Direct"

                packets.append({
                    "source": source,
                    "destination": dest,
                    "path": path,
                    "payload": payload,
                    "coords": coords,
                    "details": details,
                })

        return packets

    # ── Result presentation ───────────────────────────────────────
    def _on_decode_done(self, result: dict):
        self.decode_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._clean_html = self._format_result(result)
        self.results_output.setHtml(self._clean_html)

    def _on_decode_error(self, err_msg: str):
        self.decode_btn.setEnabled(True)
        self._show_error(f"Decode execution failed:\n{err_msg}")

    def _format_result(self, r: dict) -> str:
        C = {
            "bg": "#0d0d0d", "card": "#141414", "border": "#242424",
            "title": "#e0e0e0", "text": "#c0c0c0", "muted": "#666666",
            "blue": "#64b5f6", "amber": "#ffb74d", "green": "#81c784",
            "red": "#ef9a9a", "lavender": "#ce93d8", "cyan": "#4dd0e1",
        }

        def card(colour: str, icon: str, title: str, body_html: str) -> str:
            return (
                f'<div style="background:{C["card"]};border:1px solid {colour};'
                f'border-left:5px solid {colour};border-radius:6px;'
                f'padding:12px 16px;margin:8px 0;line-height:1.6;">'
                f'<div style="color:{colour};font-weight:700;font-size:13px;margin:0 0 8px 0;">'
                f'{icon} {title}</div>'
                f'{body_html}</div>'
            )

        html = (
            f'<body style="background:{C["bg"]};color:{C["text"]};'
            f'font-family:Consolas,monospace;font-size:12px;padding:8px;">'
        )

        if "error" in r:
            html += card(
                C["red"], "✗", "Decoder Error",
                f'<pre style="color:{C["red"]};margin:0;white-space:pre-wrap;">{r["error"]}</pre>'
            )

        elif "mismatch" in r:
            mtype = r["mismatch"]
            fname = r.get("file", "")
            if mtype == "aprs_in_pocsag":
                html += card(
                    C["amber"], "⚠", "Protocol Mismatch — APRS Signal in POCSAG Decoder",
                    f'<p style="margin:0 0 6px 0;">Audio File: <b style="color:{C["blue"]}">{fname}</b></p>'
                    f'<p style="margin:0 0 8px 0;color:{C["text"]};">This audio file contains an '
                    f'<b>APRS / Bell 202 AFSK</b> packet transmission.</p>'
                    f'<p style="margin:0;color:{C["amber"]};font-weight:600;">'
                    f'👉 Switch the Protocol dropdown above to <b>APRS</b> and click Decode Signal.</p>'
                )
            elif mtype == "pocsag_in_aprs":
                html += card(
                    C["amber"], "⚠", "Protocol Mismatch — POCSAG Signal in APRS Decoder",
                    f'<p style="margin:0 0 6px 0;">Audio File: <b style="color:{C["blue"]}">{fname}</b></p>'
                    f'<p style="margin:0 0 8px 0;color:{C["text"]};">This audio file contains a '
                    f'<b>POCSAG 2-FSK</b> pager transmission.</p>'
                    f'<p style="margin:0;color:{C["amber"]};font-weight:600;">'
                    f'👉 Switch the Protocol dropdown above to <b>POCSAG</b> and click Decode Signal.</p>'
                )

        elif r.get("proto") == "POCSAG":
            fname = r.get("file", "")
            demod = r.get("demod", "")
            parsed = r.get("decoded", {})
            frames = parsed.get("frames", [])
            wrong = parsed.get("wrong_baud", {})

            html += (
                f'<div style="color:{C["muted"]};font-size:11px;margin:0 0 10px 0;'
                f'border-bottom:1px solid {C["border"]};padding-bottom:6px;">'
                f'File: <b style="color:{C["title"]}">{fname}</b> &nbsp;|&nbsp; '
                f'Demodulator: <span style="color:{C["lavender"]}">{demod}</span> &nbsp;|&nbsp; '
                f'Decoded Frames: <b style="color:{C["green"]}">{len(frames)}</b></div>'
            )

            if wrong and not any(f.get("baud") == demod for f in frames):
                best = max(wrong, key=wrong.get)
                html += card(
                    C["amber"], "⚠", f"Baud Rate Mismatch — Signal detected as {best}",
                    f'<p style="margin:0;">Signal was auto-decoded at <b style="color:{C["green"]}">{best}</b> '
                    f'(selected was {demod}).</p>'
                )

            if frames:
                for fr in frames:
                    baud = fr.get("baud", "POCSAG")
                    address = fr.get("address", "—")
                    function = fr.get("function", "—")
                    mode = fr.get("mode", "—")
                    msg = fr.get("message", "")

                    body = (
                        f'<div style="margin-bottom:8px;">'
                        f'<span style="background:#1e2638;color:{C["blue"]};padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">'
                        f'RIC: {address}</span> &nbsp;'
                        f'<span style="background:#261f38;color:{C["lavender"]};padding:2px 8px;border-radius:3px;font-size:11px;">'
                        f'Func: {function}</span> &nbsp;'
                        f'<span style="background:#1c2d20;color:{C["green"]};padding:2px 8px;border-radius:3px;font-size:11px;">'
                        f'Mode: {mode}</span> &nbsp;'
                        f'<span style="background:#222;color:{C["text"]};padding:2px 8px;border-radius:3px;font-size:11px;">'
                        f'{baud}</span>'
                        f'</div>'
                    )
                    if msg:
                        body += (
                            f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:4px;'
                            f'padding:8px 12px;font-size:13px;font-weight:600;color:#ffffff;white-space:pre-wrap;">'
                            f'{msg}</div>'
                        )
                    else:
                        body += f'<i style="color:{C["muted"]};">(Tone-only pager frame / no alpha payload)</i>'

                    html += card(C["green"], "✓", f"Decoded POCSAG Pager Frame", body)
            else:
                html += card(
                    C["muted"], "ℹ", "No POCSAG Frames Decoded",
                    f'<p style="margin:0 0 6px 0;">No decodable POCSAG frame found in this audio file.</p>'
                    f'<p style="margin:0;color:{C["muted"]};font-size:11px;">'
                    f'Tips: Verify audio volume, check if file is inverted, or ensure the signal is standard POCSAG.</p>'
                )

        elif r.get("proto") == "APRS":
            fname = r.get("file", "")
            packets = r.get("decoded", [])

            html += (
                f'<div style="color:{C["muted"]};font-size:11px;margin:0 0 10px 0;'
                f'border-bottom:1px solid {C["border"]};padding-bottom:6px;">'
                f'File: <b style="color:{C["title"]}">{fname}</b> &nbsp;|&nbsp; '
                f'Modem: <span style="color:{C["lavender"]}">Bell 202 AFSK 1200 Baud</span> &nbsp;|&nbsp; '
                f'Decoded Packets: <b style="color:{C["green"]}">{len(packets)}</b></div>'
            )

            if packets:
                for pkt in packets:
                    src = pkt.get("source", "—")
                    dest = pkt.get("destination", "—")
                    path = pkt.get("path", "—")
                    payload = pkt.get("payload", "")
                    coords = pkt.get("coords", "")
                    details = pkt.get("details", [])

                    body = (
                        f'<div style="margin-bottom:8px;">'
                        f'<span style="background:#1e2a3a;color:{C["cyan"]};padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">'
                        f'FROM: {src}</span> &nbsp;'
                        f'<span style="background:#261f38;color:{C["lavender"]};padding:2px 8px;border-radius:3px;font-size:11px;">'
                        f'TO: {dest}</span> &nbsp;'
                        f'<span style="background:#1f2820;color:{C["text"]};padding:2px 8px;border-radius:3px;font-size:11px;">'
                        f'VIA: {path}</span>'
                        f'</div>'
                    )

                    if coords:
                        body += (
                            f'<div style="margin-bottom:6px;color:{C["amber"]};font-weight:600;">'
                            f'📍 Position: <span style="color:#ffffff;">{coords}</span></div>'
                        )

                    if payload:
                        body += (
                            f'<div style="background:{C["bg"]};border:1px solid {C["border"]};border-radius:4px;'
                            f'padding:8px 12px;font-size:13px;font-weight:600;color:#ffffff;white-space:pre-wrap;margin-bottom:6px;">'
                            f'{payload}</div>'
                        )

                    if details:
                        details_str = "<br>".join(details)
                        body += (
                            f'<div style="color:{C["text"]};font-size:11px;padding:4px 2px;">'
                            f'{details_str}</div>'
                        )

                    html += card(C["green"], "✓", f"Decoded APRS / AX.25 Packet", body)
            else:
                html += card(
                    C["muted"], "ℹ", "No APRS Packets Decoded",
                    f'<p style="margin:0 0 6px 0;">No valid AX.25 / APRS packets found in this audio file.</p>'
                    f'<p style="margin:0;color:{C["muted"]};font-size:11px;">'
                    f'Possible causes: Audio level too high/low, not an APRS audio file, or corrupted transmission.</p>'
                )

        html += "</body>"
        return html

    def _show_error(self, msg: str):
        self.results_output.setHtml(
            f'<body style="font-family:Consolas,monospace;font-size:12px;'
            f'background:#0d0d0d;color:#ef9a9a;padding:10px;">'
            f'<b>✗ Error</b><br><pre>{msg}</pre></body>'
        )

    @staticmethod
    def _html_info(msg: str) -> str:
        return (
            f'<body style="font-family:Consolas,monospace;font-size:12px;'
            f'background:#0d0d0d;color:#888888;padding:10px;">{msg}</body>'
        )


# ═══════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════
class PagerMoleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PagerMole DCS — Digital Communications Suite")
        self.setMinimumSize(1100, 750)
        self.resize(1260, 840)

        # ── Central tab widget ────────────────────────────────────
        self.tabs = QTabWidget()
        self.wave_gen_tab = WaveGeneratorTab()
        self.decode_tab = DecodeTab()

        self.tabs.addTab(self.wave_gen_tab, "  ⚡ WAVE GENERATOR & VISUALIZER  ")
        self.tabs.addTab(self.decode_tab, "  🔍 DECODER  ")
        self.setCentralWidget(self.tabs)

        # Cross-tab connection: allow decoder tab to send file to visualizer
        self.decode_tab.send_to_visualizer.connect(self._on_visualize_request)

        # ── Status bar ────────────────────────────────────────────
        sb = self.statusBar()
        sb.showMessage("PagerMole DCS — Studio Ready")

    def closeEvent(self, event):
        """Safely shutdown active threads before closing."""
        for tab in (self.wave_gen_tab, self.decode_tab):
            if hasattr(tab, "_worker") and tab._worker and tab._worker.isRunning():
                tab._worker.terminate()
                tab._worker.wait(500)
        super().closeEvent(event)

    def _on_visualize_request(self, file_path: str):
        """Switch to wave generator tab and load the requested file."""
        p = Path(file_path)
        if p.exists():
            try:
                raw_bytes = p.read_bytes()
                self.tabs.setCurrentIndex(0)
                if "aprs" in p.name.lower():
                    self.wave_gen_tab.proto_combo.setCurrentIndex(1)
                elif "pocsag" in p.name.lower():
                    self.wave_gen_tab.proto_combo.setCurrentIndex(0)
                self.wave_gen_tab._process_active_wav(raw_bytes, p)
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to load file for visualization: {exc}", 5000)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PagerMole DCS")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(GLOBAL_STYLESHEET)
    window = PagerMoleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

