"""APRS / AX.25 AFSK1200 Bell 202 packet signal generator and audio synthesizer."""

import io
import math
import struct
import wave
from pathlib import Path
from typing import List, Optional, Union


def crc16_ccitt(data: bytes) -> int:
    """Compute standard 16-bit CRC-CCITT (polynomial 0x8408 reversed)."""
    crc = 0xFFFF
    for byte in data:
        for bit in range(8):
            b = (byte >> bit) & 1
            low = crc & 1
            crc >>= 1
            if low ^ b:
                crc ^= 0x8408
    return (~crc) & 0xFFFF


def encode_callsign(call: str, ssid: int = 0, is_last: bool = False) -> bytes:
    """Encode an amateur radio callsign and SSID into standard AX.25 7-byte address format."""
    call = call.upper().strip().ljust(6)[:6]
    encoded = bytearray([ord(c) << 1 for c in call])
    ssid_byte = 0x60 | ((ssid & 0x0F) << 1)
    if is_last:
        ssid_byte |= 0x01
    encoded.append(ssid_byte)
    return bytes(encoded)


def format_aprs_position(lat: float, lon: float, symbol_table: str = "/", symbol_code: str = ">", comment: str = "") -> str:
    """Format decimal latitude and longitude into standard APRS position string format."""
    # Latitude format: DDMM.MMN or DDMM.MMS
    lat_dir = "N" if lat >= 0 else "S"
    lat_abs = abs(lat)
    lat_deg = int(lat_abs)
    lat_min = (lat_abs - lat_deg) * 60.0
    lat_str = f"{lat_deg:02d}{lat_min:05.2f}{lat_dir}"

    # Longitude format: DDDMM.MME or DDDMM.MMW
    lon_dir = "E" if lon >= 0 else "W"
    lon_abs = abs(lon)
    lon_deg = int(lon_abs)
    lon_min = (lon_abs - lon_deg) * 60.0
    lon_str = f"{lon_deg:03d}{lon_min:05.2f}{lon_dir}"

    sym_tbl = symbol_table[0] if symbol_table else "/"
    sym_cd = symbol_code[0] if symbol_code else ">"

    body = f"!{lat_str}{sym_tbl}{lon_str}{sym_cd}"
    if comment:
        body += f"{comment}"
    return body


def build_ax25_frame(
    source: str,
    destination: str = "APRS",
    path: Optional[List[str]] = None,
    info: str = "",
) -> bytes:
    """Build an AX.25 UI-frame (Unnumbered Information) with CRC."""
    path_list = path or ["WIDE1-1", "WIDE2-1"]

    src_parts = source.split("-")
    src_call = src_parts[0]
    src_ssid = int(src_parts[1]) if len(src_parts) > 1 else 0

    dst_parts = destination.split("-")
    dst_call = dst_parts[0]
    dst_ssid = int(dst_parts[1]) if len(dst_parts) > 1 else 0

    has_path = len(path_list) > 0
    header = bytearray()
    header.extend(encode_callsign(dst_call, dst_ssid, is_last=False))
    header.extend(encode_callsign(src_call, src_ssid, is_last=(not has_path)))

    for i, p in enumerate(path_list):
        p_parts = p.split("-")
        p_call = p_parts[0]
        p_ssid = int(p_parts[1]) if len(p_parts) > 1 else 0
        is_last = (i == len(path_list) - 1)
        header.extend(encode_callsign(p_call, p_ssid, is_last=is_last))

    header.append(0x03)  # Control: UI Frame (0x03)
    header.append(0xF0)  # PID: No layer 3 (0xF0)

    payload = bytes(header) + info.encode("latin1", errors="replace")
    fcs = crc16_ccitt(payload)
    fcs_bytes = struct.pack("<H", fcs)
    return payload + fcs_bytes


def frame_to_bits(frame: bytes, preamble_flags: int = 35, postamble_flags: int = 15) -> List[int]:
    """Convert AX.25 frame to bitstream with bit-stuffing and 0x7E sync flags."""
    FLAG = 0x7E
    bits: List[int] = []

    def send_byte(val: int) -> None:
        for i in range(8):
            bits.append((val >> i) & 1)

    for _ in range(preamble_flags):
        send_byte(FLAG)

    ones_count = 0
    for byte in frame:
        for i in range(8):
            bit = (byte >> i) & 1
            bits.append(bit)
            if bit == 1:
                ones_count += 1
                if ones_count == 5:
                    bits.append(0)  # bit-stuffing
                    ones_count = 0
            else:
                ones_count = 0

    for _ in range(postamble_flags):
        send_byte(FLAG)

    return bits


def nrzi_encode(bits: List[int]) -> List[int]:
    """NRZI (Non-Return-to-Zero Inverted) encoding: 0 inverts state, 1 stays constant."""
    current_state = 1
    nrzi: List[int] = []
    for bit in bits:
        if bit == 0:
            current_state = 1 - current_state
        nrzi.append(current_state)
    return nrzi


def afsk_modulate(
    nrzi_bits: List[int],
    sample_rate: int = 22050,
    baud: int = 1200,
    amplitude: float = 24000.0,
) -> bytes:
    """Modulate NRZI bits into Bell 202 continuous-phase AFSK audio (1200 Hz Mark, 2200 Hz Space)."""
    samples: List[int] = []
    MARK_FREQ = 1200.0   # State 1
    SPACE_FREQ = 2200.0  # State 0
    TWO_PI = 2.0 * math.pi
    phase = 0.0

    pos = 0.0
    for state in nrzi_bits:
        freq = MARK_FREQ if state == 1 else SPACE_FREQ
        phase_inc = TWO_PI * freq / sample_rate
        pos += sample_rate / baud
        count = int(pos)
        pos -= count
        for _ in range(count):
            val = int(amplitude * math.sin(phase))
            samples.append(val)
            phase += phase_inc
            if phase >= TWO_PI:
                phase -= TWO_PI

    # 150ms quiet padding
    silence = [0] * int(sample_rate * 0.15)
    all_samples = silence + samples + silence
    return struct.pack(f"<{len(all_samples)}h", *all_samples)


def generate_aprs_wav(
    source: str = "N0CALL-1",
    destination: str = "APRS",
    path: Optional[List[str]] = None,
    info: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    comment: str = "PagerMole Signal Test",
    symbol_table: str = "/",
    symbol_code: str = ">",
    sample_rate: int = 22050,
    baud: int = 1200,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """Generate an APRS Bell 202 AFSK audio WAV file.

    Args:
        source: Callsign of transmitting station (e.g. N0CALL-1).
        destination: Destination callsign (default "APRS").
        path: Digipeater path list (e.g. ["WIDE1-1", "WIDE2-1"]).
        info: Custom info text or message. If lat/lon are provided, this will be formatted as position.
        latitude: Optional decimal latitude (e.g. 42.16667).
        longitude: Optional decimal longitude (e.g. -71.0).
        comment: Optional station comment.
        symbol_table: APRS primary or alternate table char (default "/").
        symbol_code: APRS symbol table icon char (default ">" for car).
        sample_rate: Audio sample rate (default 22050 Hz).
        baud: Baud rate (default 1200 baud).
        output_path: Optional file path to write the WAV file to.

    Returns:
        Raw WAV bytes.
    """
    if latitude is not None and longitude is not None:
        aprs_payload = format_aprs_position(latitude, longitude, symbol_table, symbol_code, comment)
    elif info:
        aprs_payload = info
    else:
        aprs_payload = f">000000z/PagerMole Tactical Node {source}"

    frame = build_ax25_frame(source, destination, path, aprs_payload)
    bits = frame_to_bits(frame)
    nrzi = nrzi_encode(bits)
    raw_pcm = afsk_modulate(nrzi, sample_rate, baud)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)

    wav_bytes = wav_buf.getvalue()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(wav_bytes)

    return wav_bytes
