"""POCSAG pager signal generator (512, 1200, 2400 baud FSK audio/WAV)."""

import io
import math
import struct
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Union

POCSAG_POLY = 0x769
POCSAG_SYNC = 0x7CD215D8
POCSAG_IDLE = 0x7A89C197
POCSAG_PREAMBLE_BITS = 576
# Audio tones used for the sound-card representation of POCSAG.  These are
# intentionally distinct from the RF deviation specified by POCSAG; a WAV
# file carries the baseband 2-FSK signal, not an RF carrier.
POCSAG_MARK_HZ = 2400.0   # logic 1
POCSAG_SPACE_HZ = 1400.0  # logic 0

# Precompute 10-bit parity lookup table for BCH(31,21)
_pocsag_parity_tbl: List[int] = []
for databit in range(21):
    shreg = 1 << (databit + 10)
    for i in range(20, -1, -1):
        if shreg & (1 << (i + 10)):
            shreg ^= (POCSAG_POLY << i)
    _pocsag_parity_tbl.append(shreg & 0x3FF)


def bch_encode_pocsag(data21: int) -> int:
    """Encode 21 bits into standard 32-bit POCSAG codeword with BCH parity and even parity."""
    d = data21 & 0x1FFFFF
    parity = 0
    tmp = d
    bit = 0
    while tmp:
        if tmp & 1:
            parity ^= _pocsag_parity_tbl[bit]
        tmp >>= 1
        bit += 1
    cw = (d << 11) | (parity << 1)
    # Bit 0 is even parity across all 32 bits
    if bin(cw).count("1") % 2 != 0:
        cw |= 1
    return cw


def build_address_codeword(address: int, function: int = 3) -> int:
    """Build 32-bit address codeword."""
    data = ((address >> 3) << 2) | (function & 3)
    return bch_encode_pocsag(data)


def _rev7(b: int) -> int:
    """Bit-reverse a 7-bit character value."""
    return (
        ((b << 6) & 64)
        | ((b >> 6) & 1)
        | ((b << 4) & 32)
        | ((b >> 4) & 2)
        | ((b << 2) & 16)
        | ((b >> 2) & 4)
        | (b & 8)
    )


def encode_alphanumeric_codewords(msg: str) -> List[int]:
    """Encode an alphanumeric message into a list of 32-bit POCSAG message codewords."""
    buffer = bytearray(512)
    for i, ch in enumerate(msg[:512 * 8 // 7]):
        c = _rev7(ord(ch) & 0x7F)
        start_bit = i * 7
        for b in range(7):
            bit_pos = start_bit + b
            byte_idx = bit_pos // 8
            bit_in_byte = 7 - (bit_pos % 8)
            if c & (0x40 >> b):
                buffer[byte_idx] |= 1 << bit_in_byte

    total_bits = len(msg) * 7
    total_nibbles = (total_bits + 3) // 4
    nibble_idx = 0
    codewords: List[int] = []

    while nibble_idx < total_nibbles:
        data20 = 0
        for _ in range(5):
            if nibble_idx < total_nibbles:
                byte_pos = nibble_idx // 2
                if nibble_idx % 2 == 0:
                    nibble = (buffer[byte_pos] >> 4) & 0xF
                else:
                    nibble = buffer[byte_pos] & 0xF
                nibble_idx += 1
            else:
                nibble = 0
            data20 = (data20 << 4) | nibble
        data = (1 << 20) | (data20 & 0xFFFFF)
        codewords.append(bch_encode_pocsag(data))

    return codewords


def encode_numeric_codewords(digits: str) -> List[int]:
    """Encode numeric digits into 32-bit POCSAG message codewords."""
    bcd_map = {
        "0": 0, "1": 8, "2": 4, "3": 12, "4": 2, "5": 10, "6": 6, "7": 14,
        "8": 1, "9": 9, "U": 13, "u": 13, " ": 3, "-": 11, ".": 5, "[": 15, "]": 7
    }
    codewords: List[int] = []
    i = 0
    while i < len(digits):
        data20 = 0
        for _ in range(5):
            if i < len(digits):
                val = bcd_map.get(digits[i], 3)
                i += 1
            else:
                val = 3  # space padding
            data20 = (data20 << 4) | (val & 0xF)
        data = (1 << 20) | (data20 & 0xFFFFF)
        codewords.append(bch_encode_pocsag(data))
    return codewords


def generate_pocsag_bitstream(
    address: int,
    message: str,
    function: int = 3,
    is_numeric: bool = False,
) -> List[int]:
    """Generate the full raw bitstream (preamble + sync + batches + idle)."""
    addr_cw = build_address_codeword(address, function)
    if is_numeric:
        msg_cws = encode_numeric_codewords(message)
    else:
        msg_cws = encode_alphanumeric_codewords(message)

    frame_pos = address & 7
    slots_needed = 1 + len(msg_cws) + 1
    slots_in_first = 16 - (frame_pos * 2)
    if slots_needed <= slots_in_first:
        batch_count = 1
    else:
        batch_count = 1 + ((slots_needed - slots_in_first + 15) // 16)

    bits: List[int] = []
    # 1. Preamble (576 alternating 1/0 bits)
    for i in range(POCSAG_PREAMBLE_BITS):
        bits.append(1 if (i % 2 == 0) else 0)

    # 2. Batches
    msg_idx = 0
    addr_sent = False
    for _ in range(batch_count):
        # Sync word (32 bits, MSB first)
        for b in range(31, -1, -1):
            bits.append((POCSAG_SYNC >> b) & 1)

        # 16 codewords per batch (8 frames of 2 codewords)
        for slot in range(16):
            if not addr_sent and slot == frame_pos * 2:
                cw = addr_cw
                addr_sent = True
            elif addr_sent and msg_idx < len(msg_cws):
                cw = msg_cws[msg_idx]
                msg_idx += 1
            else:
                cw = POCSAG_IDLE

            for b in range(31, -1, -1):
                bits.append((cw >> b) & 1)

    return bits


def generate_pocsag_wav(
    address: int,
    message: str,
    function: int = 3,
    baud: int = 1200,
    sample_rate: int = 22050,
    is_numeric: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """Generate a standard 16-bit mono PCM WAV audio file of the POCSAG signal.

    Args:
        address: Receiver Identity Code (RIC / pager address).
        message: Alphanumeric or numeric message text.
        function: Function bit 0-3 (default 3 for alpha).
        baud: Baud rate (512, 1200, 2400).
        sample_rate: Audio sampling frequency (default 22050 Hz).
        is_numeric: True if message is numeric BCD.
        output_path: Optional file path to write the WAV file to.

    Returns:
        Raw WAV bytes.
    """
    bits = generate_pocsag_bitstream(address, message, function, is_numeric)
    samples_per_bit = sample_rate / baud

    # Continuous-phase audio 2-FSK.  Keeping the phase accumulator across
    # symbol changes is essential: it produces a real FSK WAV rather than a
    # rectangular discriminator placeholder and avoids phase discontinuities.
    audio_samples: List[int] = []
    # Add a short 100ms silence before transmission
    silence_count = int(sample_rate * 0.1)
    audio_samples.extend([0] * silence_count)

    phase = 0.0
    symbol_position = 0.0
    for bit in bits:
        frequency = POCSAG_MARK_HZ if bit else POCSAG_SPACE_HZ
        phase_increment = 2.0 * math.pi * frequency / sample_rate
        symbol_position += samples_per_bit
        sample_count = int(symbol_position)
        symbol_position -= sample_count
        for _ in range(sample_count):
            audio_samples.append(int(20000 * math.sin(phase)))
            phase = (phase + phase_increment) % (2.0 * math.pi)

    # 100ms trailing silence
    audio_samples.extend([0] * silence_count)

    raw_pcm = struct.pack(f"<{len(audio_samples)}h", *audio_samples)

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
