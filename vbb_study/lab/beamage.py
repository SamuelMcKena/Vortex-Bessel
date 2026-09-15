"""Strict adapter for the supplied Beamage-4M v1 BMG layout.

Grounded in 52 supplied files (2048x2048, 481-byte prefix, little-endian
signed int32 pixels). Other layouts fail explicitly; no end-of-file guessing.
The exported pixels can be negative (camera background correction). BMG is a
quantitative camera export, not necessarily untouched ADC counts.
"""
from pathlib import Path
import struct
import numpy as np


def read_bmg(path):
    raw = Path(path).read_bytes()
    if len(raw) < 481:
        raise ValueError("BMG is shorter than its supported header.")
    version, bits, width, height = struct.unpack_from('<4I', raw)
    if (version, bits, width, height) != (1, 12, 2048, 2048):
        raise ValueError(f"Unsupported BMG header {(version, bits, width, height)}; use a verified numeric export.")
    if len(raw) != 481 + width*height*4:
        raise ValueError("BMG length does not match the validated 481-byte header and int32 image layout.")
    image = np.frombuffer(raw, dtype='<i4', offset=481).reshape(height, width).copy()
    return image, {"version": version, "adc_bits": bits, "shape_yx": [height, width],
                   "header_bytes": 481, "pixel_dtype": "<i4", "nominal_adc_full_scale": 4095,
                   "may_contain_camera_background_correction": True}
