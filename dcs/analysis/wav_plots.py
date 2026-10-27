"""DSP analysis and Matplotlib plotting for POCSAG and APRS WAV signals.

Both protocols get three consistent views:
  1. Time Overlay  — full waveform + digital/frequency track
  2. PSD Spectrum  — Welch FFT with tone markers
  3. Spectrogram   — short-time FFT waterfall for the full packet
  4. All Views     — combined tri-panel overview

All graphs use real PCM samples from generated or loaded WAV files.
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from dcs.analysis.theme import PLOT_THEME
from dcs.generators.pocsag import POCSAG_MARK_HZ, POCSAG_SPACE_HZ

# ── Graph color palette ─────────────────────────────────────────────────
_C_AUDIO   = "#d4d4d4"   # Raw audio waveform (light grey)
_C_BITS    = "#ce93d8"   # Digital bit stream (lavender / purple)
_C_MARK    = "#64b5f6"   # Mark tone / bit-1 / 2400 Hz (sky blue)
_C_SPACE   = "#ffb74d"   # Space tone / bit-0 / 1200|1400 Hz (amber)
_C_FREQ    = "#80cbc4"   # Instantaneous frequency track (teal)
_C_THRESH  = "#81c784"   # Threshold / decision / slicer (green)
_C_WORD    = "#546e7a"   # Codeword boundary markers (slate)
_C_GRID    = "#1a1a1a"   # Grid lines
_C_CLOCK   = "#2e3a3a"   # Symbol boundary ticks
_C_TEXT    = "#b0b0b0"   # Axis labels
_C_TITLE   = "#e0e0e0"   # Plot titles
_CMAP_POCSAG = "plasma"  # FSK spectrogram colormap
_CMAP_APRS   = "magma"   # AFSK spectrogram colormap


# ═══════════════════════════════════════════════════════════════════
# WAV I/O helpers
# ═══════════════════════════════════════════════════════════════════

def read_wav(source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[int, np.ndarray]:
    """Read mono normalised float64 samples in [-1.0, 1.0]."""
    if isinstance(source, (str, Path)):
        wf = wave.open(str(source), "rb")
    elif isinstance(source, bytes):
        wf = wave.open(io.BytesIO(source), "rb")
    else:
        wf = wave.open(source, "rb")

    with wf:
        channels   = wf.getnchannels()
        width      = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw        = wf.readframes(wf.getnframes())

    if width == 1:
        s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        s = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        s = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width * 8} bits")

    if channels > 1:
        s = s.reshape(-1, channels).mean(axis=1)
    if s.size == 0:
        raise ValueError("WAV file contains no audio samples")
    return sample_rate, s


def active_region(samples: np.ndarray) -> Tuple[int, int]:
    """Trim leading/trailing silence."""
    peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    thresh = max(peak * 0.015, 1e-4)
    active = np.flatnonzero(np.abs(samples) >= thresh)
    if active.size == 0:
        return 0, len(samples)
    return max(0, int(active[0]) - 1), min(len(samples), int(active[-1]) + 2)


def _welch_psd(audio: np.ndarray, sr: int):
    """Compute normalised Welch PSD up to Nyquist."""
    nperseg = min(4096, len(audio))
    freqs, psd = signal.welch(audio, fs=sr, window="hann",
                              nperseg=nperseg, noverlap=nperseg // 2)
    db = 10.0 * np.log10(np.maximum(psd, 1e-15))
    db -= np.max(db)
    return freqs, db


def _spectrogram(audio: np.ndarray, sr: int):
    """Short-time FFT spectrogram, returned as (freqs, times, db_matrix)."""
    nperseg = min(512, len(audio))
    f, t, spec = signal.spectrogram(audio, fs=sr, window="hann", nperseg=nperseg,
                                    noverlap=max(0, nperseg - 64), mode="psd")
    db = 10.0 * np.log10(np.maximum(spec, 1e-15))
    db -= np.max(db)
    return f, t, db


def _inst_freq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Instantaneous frequency via Hilbert transform (Hz)."""
    analytic  = signal.hilbert(audio)
    phase     = np.unwrap(np.angle(analytic))
    freq_raw  = np.diff(phase, prepend=phase[0]) * sr / (2.0 * np.pi)
    w = min(len(freq_raw) // 2 * 2 - 1, 21)
    if w > 3 and len(freq_raw) > w:
        return signal.savgol_filter(freq_raw, w, 2)
    return freq_raw


# ═══════════════════════════════════════════════════════════════════
# POCSAG — Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_pocsag_metrics(samples: np.ndarray, sample_rate: int,
                           baud: int = 1200) -> Dict[str, Any]:
    """Compute all POCSAG DSP metrics from raw PCM samples."""
    start, end = active_region(samples)
    audio = samples[start:end]
    if len(audio) < 100:
        audio = samples

    spb = sample_rate / baud
    n_bits = max(1, int(round(len(audio) / spb)))

    # Bit recovery via IQ energy correlation
    bits: List[int] = []
    for i in range(n_bits):
        lo = int(round(i * spb))
        hi = min(len(audio), int(round((i + 1) * spb)))
        seg = audio[lo:hi]
        if len(seg) < 3:
            continue
        t = np.arange(len(seg), dtype=float) / sample_rate
        em = ((np.dot(seg, np.cos(2 * np.pi * POCSAG_MARK_HZ  * t)) ** 2) +
              (np.dot(seg, np.sin(2 * np.pi * POCSAG_MARK_HZ  * t)) ** 2))
        es = ((np.dot(seg, np.cos(2 * np.pi * POCSAG_SPACE_HZ * t)) ** 2) +
              (np.dot(seg, np.sin(2 * np.pi * POCSAG_SPACE_HZ * t)) ** 2))
        bits.append(1 if em > es else 0)
    bit_arr = np.asarray(bits, dtype=int)

    # PSD
    psd_freqs, psd_db = _welch_psd(audio, sample_rate)

    # Spectrogram
    sg_f, sg_t, sg_db = _spectrogram(audio, sample_rate)

    # Instantaneous frequency (for time overlay secondary axis)
    inst_f = _inst_freq(audio, sample_rate)

    return {
        "audio":        audio,
        "sample_rate":  sample_rate,
        "baud":         baud,
        "bits":         bit_arr,
        "spb":          spb,
        "psd_freqs":    psd_freqs,
        "psd_db":       psd_db,
        "sg_f":         sg_f,
        "sg_t":         sg_t,
        "sg_db":        sg_db,
        "inst_freq":    inst_f,
        "duration":     len(audio) / sample_rate,
        "peak":         float(np.max(np.abs(audio))),
        "rms":          float(np.sqrt(np.mean(audio ** 2))),
    }


# ═══════════════════════════════════════════════════════════════════
# POCSAG — Drawing
# ═══════════════════════════════════════════════════════════════════

def draw_pocsag_time(ax_bits: plt.Axes, ax_audio: plt.Axes,
                     m: Dict[str, Any], label: str = ""):
    """Full-duration POCSAG bit stream + FSK audio overlay."""
    bits  = m["bits"]
    audio = m["audio"]
    sr    = m["sample_rate"]
    baud  = m["baud"]
    spb   = m["spb"]

    n_bits   = len(bits)
    clk_ms   = np.arange(n_bits + 1) / baud * 1000.0
    t_ms     = np.arange(len(audio))  / sr   * 1000.0
    t_end    = t_ms[-1] if len(t_ms) > 0 else clk_ms[-1]

    # ── Bit stream ──────────────────────────────────────────────────
    ax_bits.clear()
    ax_bits.step(
        clk_ms, np.append(bits, bits[-1] if len(bits) else 0),
        where="post", color=_C_BITS, linewidth=1.6, label="Recovered Bits"
    )
    ax_bits.fill_between(clk_ms, np.append(bits, bits[-1] if len(bits) else 0),
                         step="post", color=_C_BITS, alpha=0.18)
    ax_bits.set_ylim(-0.35, 1.45)
    ax_bits.set_yticks([0, 1])
    ax_bits.set_yticklabels(["0  Space\n(1400 Hz)", "1  Mark\n(2400 Hz)"],
                            color=_C_TEXT, fontsize=8)
    ax_bits.set_ylabel("Digital Logic", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_bits.set_title(
        f"POCSAG: Full Bit Stream + FSK Audio — {baud} Baud  "
        f"({n_bits} bits · {t_end:.1f} ms)  "
        "[ Scroll to zoom · Drag to pan ]",
        fontsize=10, fontweight="bold", color=_C_TITLE, pad=8
    )
    ax_bits.set_xlim(0, t_end)
    ax_bits.grid(True, alpha=0.35, color=_C_GRID)

    # Codeword boundaries every 32 bits (POCSAG word = 32 bits)
    for b in clk_ms[::32]:
        ax_bits.axvline(b, color=_C_WORD, linestyle=":", linewidth=0.7, alpha=0.7)

    # ── Audio waveform ───────────────────────────────────────────────
    ax_audio.clear()
    ax_audio.plot(t_ms, audio, color=_C_AUDIO, linewidth=0.55,
                  label="FSK Audio (CPFSK)", rasterized=True)
    ax_audio.axhline(0, color=_C_GRID, linewidth=0.6)

    # Colour the mark vs space regions based on recovered bits
    for i, bit in enumerate(bits):
        lo = clk_ms[i]; hi = clk_ms[i + 1] if i + 1 < len(clk_ms) else t_end
        colour = _C_MARK if bit else _C_SPACE
        ax_audio.axvspan(lo, hi, alpha=0.07, color=colour, linewidth=0)

    ax_audio.set_xlabel("Time (ms)  [ Drag to pan · Scroll to zoom ]",
                        fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_audio.set_ylabel("Amplitude", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_audio.set_ylim(-1.18, 1.18)
    ax_audio.set_xlim(0, t_end)
    ax_audio.grid(True, alpha=0.35, color=_C_GRID)

    legend_patch = [
        matplotlib.patches.Patch(color=_C_MARK,  alpha=0.5, label=f"Mark  {POCSAG_MARK_HZ:.0f} Hz"),
        matplotlib.patches.Patch(color=_C_SPACE, alpha=0.5, label=f"Space {POCSAG_SPACE_HZ:.0f} Hz"),
    ]
    ax_audio.legend(handles=legend_patch, loc="upper right",
                    fontsize=8, framealpha=0.7)

    if label:
        ax_audio.text(0.995, 0.05, label, transform=ax_audio.transAxes,
                      ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_pocsag_psd(ax: plt.Axes, m: Dict[str, Any], label: str = ""):
    """POCSAG Welch Power Spectral Density."""
    freqs = m["psd_freqs"]
    db    = m["psd_db"]

    ax.clear()
    mask = freqs <= 5500
    ax.plot(freqs[mask], db[mask], color=_C_AUDIO, linewidth=1.4, label="Welch PSD")
    ax.fill_between(freqs[mask], db[mask], db[mask].min(), alpha=0.12, color=_C_AUDIO)

    ax.axvline(POCSAG_SPACE_HZ, color=_C_SPACE, linestyle="--", linewidth=1.6,
               label=f"Space  {POCSAG_SPACE_HZ:.0f} Hz  (bit 0)")
    ax.axvline(POCSAG_MARK_HZ,  color=_C_MARK,  linestyle="--", linewidth=1.6,
               label=f"Mark   {POCSAG_MARK_HZ:.0f} Hz  (bit 1)")

    # Shade the two tone regions
    ax.axvspan(POCSAG_SPACE_HZ - 200, POCSAG_SPACE_HZ + 200,
               alpha=0.12, color=_C_SPACE, linewidth=0)
    ax.axvspan(POCSAG_MARK_HZ  - 200, POCSAG_MARK_HZ  + 200,
               alpha=0.12, color=_C_MARK,  linewidth=0)

    ax.set_xlim(0, 5500)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (Hz)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_ylabel("Relative Power (dB)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_title("POCSAG: Power Spectral Density — Mark & Space Frequency Lobes",
                 fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)
    ax.grid(True, alpha=0.4, color=_C_GRID)
    ax.legend(loc="upper right", framealpha=0.75, fontsize=9)

    if label:
        ax.text(0.98, 0.05, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_pocsag_spectrogram(ax: plt.Axes, m: Dict[str, Any], label: str = "",
                             fig: Optional[plt.Figure] = None):
    """POCSAG Short-Time FFT Spectrogram — shows FSK frequency switching."""
    f   = m["sg_f"]
    t   = m["sg_t"]
    db  = m["sg_db"]

    ax.clear()
    mesh = ax.pcolormesh(t, f, db, shading="auto",
                         cmap=_CMAP_POCSAG, vmin=-55, vmax=0)

    ax.axhline(POCSAG_SPACE_HZ, color=_C_SPACE, linestyle="--", linewidth=1.4,
               label=f"Space {POCSAG_SPACE_HZ:.0f} Hz")
    ax.axhline(POCSAG_MARK_HZ,  color=_C_MARK,  linestyle="--", linewidth=1.4,
               label=f"Mark  {POCSAG_MARK_HZ:.0f} Hz")

    ax.set_ylim(500, 4000)
    ax.set_xlabel("Time (s)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_title("POCSAG: Spectrogram — FSK Frequency Switching Over Time",
                 fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)
    ax.legend(loc="upper right", framealpha=0.75, fontsize=9)

    if fig:
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("Power (dB)", color=_C_TEXT, fontsize=8)
        cb.ax.tick_params(colors=_C_TEXT, labelsize=7.5)

    if label:
        ax.text(0.98, 0.05, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_pocsag_all(fig: plt.Figure, m: Dict[str, Any], label: str = ""):
    """Tri-panel POCSAG overview: Time Overlay + PSD + Spectrogram."""
    fig.clear()
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0],
        hspace=0.40, wspace=0.30,
        left=0.09, right=0.96, top=0.94, bottom=0.09
    )

    # Top span: time overlay
    gs_top = gs[0, :].subgridspec(2, 1, height_ratios=[1.0, 2.0], hspace=0.10)
    ax_bits  = fig.add_subplot(gs_top[0])
    ax_audio = fig.add_subplot(gs_top[1], sharex=ax_bits)
    draw_pocsag_time(ax_bits, ax_audio, m, label)
    plt.setp(ax_bits.get_xticklabels(), visible=False)

    ax_psd = fig.add_subplot(gs[1, 0])
    draw_pocsag_psd(ax_psd, m, label)

    ax_sg = fig.add_subplot(gs[1, 1])
    draw_pocsag_spectrogram(ax_sg, m, label, fig=None)


# ═══════════════════════════════════════════════════════════════════
# APRS — Metrics
# ═══════════════════════════════════════════════════════════════════

_APRS_MARK_HZ  = 1200.0
_APRS_SPACE_HZ = 2200.0
_APRS_BAUD     = 1200

def compute_aprs_metrics(samples: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute all APRS Bell 202 DSP metrics from raw PCM samples."""
    start, end = active_region(samples)
    audio = samples[start:end]
    if len(audio) < 100:
        audio = samples

    n = len(audio)
    sr = sample_rate

    # PSD
    psd_freqs, psd_db = _welch_psd(audio, sr)

    # Spectrogram
    sg_f, sg_t, sg_db = _spectrogram(audio, sr)

    # Instantaneous frequency for time overlay
    inst_f = _inst_freq(audio, sr)
    # Clip to reasonable AFSK range for display
    inst_f_display = np.clip(inst_f, 400, 3200)

    # Symbol & frame boundaries for time overlay markers
    n_symbols      = max(1, int(round(n * _APRS_BAUD / sr)))
    sym_bounds_ms  = np.arange(n_symbols + 1) / _APRS_BAUD * 1000.0
    n_frames       = n_symbols // 8 + 2
    frame_bounds_ms = np.arange(n_frames + 1) * 8.0 / _APRS_BAUD * 1000.0

    # Zero-crossing intervals (kept for completeness / export)
    idx = np.flatnonzero(np.signbit(audio[:-1]) != np.signbit(audio[1:]))
    if len(idx) > 2:
        offs = -audio[idx] / (audio[idx + 1] - audio[idx])
        xings = idx + offs
        iv_us = np.diff(xings) / sr * 1_000_000.0
        valid_iv = iv_us[(iv_us >= 130.0) & (iv_us <= 600.0)]
    else:
        valid_iv = np.array([])

    return {
        "audio":           audio,
        "sample_rate":     sr,
        "baud":            _APRS_BAUD,
        "psd_freqs":       psd_freqs,
        "psd_db":          psd_db,
        "sg_f":            sg_f,
        "sg_t":            sg_t,
        "sg_db":           sg_db,
        "inst_freq":       inst_f_display,
        "sym_bounds_ms":   sym_bounds_ms,
        "frame_bounds_ms": frame_bounds_ms,
        "intervals_us":    valid_iv,
        "duration":        n / sr,
        "peak":            float(np.max(np.abs(audio))),
        "rms":             float(np.sqrt(np.mean(audio ** 2))),
    }


# ═══════════════════════════════════════════════════════════════════
# APRS — Drawing
# ═══════════════════════════════════════════════════════════════════

def draw_aprs_time(ax_audio: plt.Axes, ax_freq: plt.Axes,
                   m: Dict[str, Any], label: str = ""):
    """Full-duration APRS audio waveform + instantaneous frequency track."""
    audio        = m["audio"]
    sr           = m["sample_rate"]
    inst_f       = m["inst_freq"]
    frame_bounds = m["frame_bounds_ms"]
    sym_bounds   = m["sym_bounds_ms"]

    t_ms     = np.arange(len(audio)) / sr * 1000.0
    dur_ms   = t_ms[-1] if len(t_ms) > 0 else 0.0

    # ── Audio waveform ───────────────────────────────────────────────
    ax_audio.clear()
    ax_audio.plot(t_ms, audio, color=_C_AUDIO, linewidth=0.55,
                  label="AFSK Audio", rasterized=True)
    ax_audio.axhline(0, color=_C_GRID, linewidth=0.5)

    # Frame boundaries (every 8 bits = 1 HDLC byte)
    vis_fb = frame_bounds[frame_bounds <= dur_ms * 1.01]
    for fb in vis_fb:
        ax_audio.axvline(fb, color=_C_SPACE, linestyle="--",
                         linewidth=0.75, alpha=0.65)

    ax_audio.set_ylabel("Amplitude", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_audio.set_title(
        f"APRS: Full Audio Waveform + Instantaneous Frequency — "
        f"{dur_ms:.1f} ms · 1200 baud Bell 202  [ Drag to pan · Scroll to zoom ]",
        fontsize=10, fontweight="bold", color=_C_TITLE, pad=8
    )
    ax_audio.set_xlim(0, dur_ms)
    ax_audio.set_ylim(-1.18, 1.18)
    ax_audio.grid(True, alpha=0.30, color=_C_GRID)

    # ── Instantaneous frequency ──────────────────────────────────────
    ax_freq.clear()
    ax_freq.plot(t_ms, inst_f, color=_C_FREQ, linewidth=0.8,
                 label="Inst. Freq (Hz)", rasterized=True)

    ax_freq.axhline(_APRS_MARK_HZ,  color=_C_MARK,  linestyle="--",
                    linewidth=1.4, label=f"Mark  {_APRS_MARK_HZ:.0f} Hz")
    ax_freq.axhline(_APRS_SPACE_HZ, color=_C_SPACE, linestyle="--",
                    linewidth=1.4, label=f"Space {_APRS_SPACE_HZ:.0f} Hz")

    # Shade mark vs space regions
    thresh_hz = (_APRS_MARK_HZ + _APRS_SPACE_HZ) / 2.0
    freq_interp = np.interp(np.arange(len(t_ms)), np.arange(len(inst_f)), inst_f)
    ax_freq.fill_between(t_ms, freq_interp, thresh_hz,
                         where=(freq_interp < thresh_hz),
                         color=_C_MARK,  alpha=0.15, interpolate=True)
    ax_freq.fill_between(t_ms, freq_interp, thresh_hz,
                         where=(freq_interp >= thresh_hz),
                         color=_C_SPACE, alpha=0.15, interpolate=True)

    # Frame boundaries on freq panel
    for fb in vis_fb:
        ax_freq.axvline(fb, color=_C_SPACE, linestyle="--",
                        linewidth=0.75, alpha=0.55)

    ax_freq.set_xlabel("Time (ms)  [ Drag to pan · Scroll to zoom ]",
                       fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_freq.set_ylabel("Freq (Hz)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax_freq.set_xlim(0, dur_ms)
    ax_freq.set_ylim(500, 3000)
    ax_freq.set_yticks([_APRS_MARK_HZ, (_APRS_MARK_HZ + _APRS_SPACE_HZ) / 2, _APRS_SPACE_HZ])
    ax_freq.set_yticklabels(
        [f"{_APRS_MARK_HZ:.0f}", "1700", f"{_APRS_SPACE_HZ:.0f}"],
        color=_C_TEXT, fontsize=8
    )
    ax_freq.grid(True, alpha=0.30, color=_C_GRID)
    ax_freq.legend(loc="upper right", fontsize=8, framealpha=0.70)

    if label:
        ax_audio.text(0.995, 0.05, label, transform=ax_audio.transAxes,
                      ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_aprs_psd(ax: plt.Axes, m: Dict[str, Any], label: str = ""):
    """APRS Welch Power Spectral Density — 1200 / 2200 Hz lobes."""
    freqs = m["psd_freqs"]
    db    = m["psd_db"]

    ax.clear()
    mask = freqs <= 4000
    ax.plot(freqs[mask], db[mask], color=_C_AUDIO, linewidth=1.4, label="Welch PSD")
    ax.fill_between(freqs[mask], db[mask], db[mask].min(), alpha=0.12, color=_C_AUDIO)

    ax.axvline(_APRS_MARK_HZ,  color=_C_MARK,  linestyle="--", linewidth=1.6,
               label=f"Mark   {_APRS_MARK_HZ:.0f} Hz  (bit 1)")
    ax.axvline(_APRS_SPACE_HZ, color=_C_SPACE, linestyle="--", linewidth=1.6,
               label=f"Space  {_APRS_SPACE_HZ:.0f} Hz  (bit 0)")

    ax.axvspan(_APRS_MARK_HZ  - 200, _APRS_MARK_HZ  + 200, alpha=0.13, color=_C_MARK,  linewidth=0)
    ax.axvspan(_APRS_SPACE_HZ - 200, _APRS_SPACE_HZ + 200, alpha=0.13, color=_C_SPACE, linewidth=0)

    ax.set_xlim(0, 4000)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency (Hz)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_ylabel("Relative Power (dB)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_title("APRS: Power Spectral Density — Bell 202 AFSK Mark & Space Lobes",
                 fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)
    ax.grid(True, alpha=0.4, color=_C_GRID)
    ax.legend(loc="upper right", framealpha=0.75, fontsize=9)

    if label:
        ax.text(0.98, 0.05, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_aprs_spectrogram(ax: plt.Axes, m: Dict[str, Any], label: str = "",
                           fig: Optional[plt.Figure] = None):
    """APRS Spectrogram / Waterfall — full packet time-frequency view."""
    f  = m["sg_f"]
    t  = m["sg_t"]
    db = m["sg_db"]

    ax.clear()
    mesh = ax.pcolormesh(t, f, db, shading="auto",
                         cmap=_CMAP_APRS, vmin=-55, vmax=0)

    ax.axhline(_APRS_MARK_HZ,  color=_C_MARK,  linestyle="--", linewidth=1.4,
               label=f"Mark  {_APRS_MARK_HZ:.0f} Hz")
    ax.axhline(_APRS_SPACE_HZ, color=_C_SPACE, linestyle="--", linewidth=1.4,
               label=f"Space {_APRS_SPACE_HZ:.0f} Hz")

    ax.set_ylim(400, 3200)
    ax.set_xlabel("Time (s)  [ Drag to pan · Scroll to zoom ]",
                  fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold", color=_C_TEXT)
    ax.set_title("APRS: Spectrogram / Waterfall — Complete Packet Burst",
                 fontsize=11, fontweight="bold", color=_C_TITLE, pad=8)
    ax.legend(loc="upper right", framealpha=0.75, fontsize=9)

    if fig:
        cb = fig.colorbar(mesh, ax=ax, pad=0.02)
        cb.set_label("Power (dB)", color=_C_TEXT, fontsize=8)
        cb.ax.tick_params(colors=_C_TEXT, labelsize=7.5)

    if label:
        ax.text(0.98, 0.05, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_C_THRESH)


def draw_aprs_all(fig: plt.Figure, m: Dict[str, Any], label: str = ""):
    """Tri-panel APRS overview: Time Overlay + PSD + Spectrogram."""
    fig.clear()
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0],
        hspace=0.40, wspace=0.32,
        left=0.09, right=0.96, top=0.94, bottom=0.09
    )

    # Top span: time overlay (audio + freq track)
    gs_top = gs[0, :].subgridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.10)
    ax_audio = fig.add_subplot(gs_top[0])
    ax_freq  = fig.add_subplot(gs_top[1], sharex=ax_audio)
    draw_aprs_time(ax_audio, ax_freq, m, label)
    plt.setp(ax_audio.get_xticklabels(), visible=False)

    ax_psd = fig.add_subplot(gs[1, 0])
    draw_aprs_psd(ax_psd, m, label)

    ax_sg = fig.add_subplot(gs[1, 1])
    draw_aprs_spectrogram(ax_sg, m, label, fig=None)


# ═══════════════════════════════════════════════════════════════════
# Backward-compatible image export helpers
# ═══════════════════════════════════════════════════════════════════

def _fig_to_png_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pocsag_wav_plots(path_or_bytes: Union[str, Path, bytes],
                     baud: int = 1200) -> Dict[str, str]:
    sr, samples = read_wav(path_or_bytes)
    lbl = Path(path_or_bytes).name if isinstance(path_or_bytes, (str, Path)) else "Signal"
    m = compute_pocsag_metrics(samples, sr, baud)
    with plt.rc_context(PLOT_THEME):
        fig1, (ab, aa) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2]})
        draw_pocsag_time(ab, aa, m, lbl); fig1.tight_layout()
        p1 = _fig_to_png_uri(fig1)
        fig2, ax2 = plt.subplots(figsize=(9, 5)); draw_pocsag_psd(ax2, m, lbl)
        fig2.tight_layout(); p2 = _fig_to_png_uri(fig2)
        fig3, ax3 = plt.subplots(figsize=(9, 5)); draw_pocsag_spectrogram(ax3, m, lbl)
        fig3.tight_layout(); p3 = _fig_to_png_uri(fig3)
    return {"time_overlay": p1, "psd": p2, "spectrogram": p3}


def aprs_wav_plots(path_or_bytes: Union[str, Path, bytes]) -> Dict[str, str]:
    sr, samples = read_wav(path_or_bytes)
    lbl = Path(path_or_bytes).name if isinstance(path_or_bytes, (str, Path)) else "Signal"
    m = compute_aprs_metrics(samples, sr)
    with plt.rc_context(PLOT_THEME):
        fig1, (aa, af) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.4]})
        draw_aprs_time(aa, af, m, lbl); fig1.tight_layout(); p1 = _fig_to_png_uri(fig1)
        fig2, ax2 = plt.subplots(figsize=(9, 5)); draw_aprs_psd(ax2, m, lbl)
        fig2.tight_layout(); p2 = _fig_to_png_uri(fig2)
        fig3, ax3 = plt.subplots(figsize=(9, 5)); draw_aprs_spectrogram(ax3, m, lbl)
        fig3.tight_layout(); p3 = _fig_to_png_uri(fig3)
    return {"time_overlay": p1, "psd": p2, "spectrogram": p3}
