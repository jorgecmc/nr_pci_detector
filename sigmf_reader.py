"""
sigmf_reader.py
Memory-efficient SigMF metadata parser and IQ sample loader.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


SUPPORTED_DATATYPES = {
    "ci16_le": (np.dtype("<i2"), 2, 32768.0),
    "ci16_be": (np.dtype(">i2"), 2, 32768.0),
    "cf32_le": (np.dtype("<f4"), 2, 1.0),
    "ci8": (np.dtype("i1"), 2, 128.0),
}


@dataclass
class SigMFMetadata:
    """Structured container for validated SigMF metadata."""
    data_path: Path
    meta_path: Path
    datatype: str
    sample_rate: float
    center_frequency: float
    datetime_str: str
    author: str
    description: str
    recorder: str
    version: str
    total_samples: int
    duration_sec: float
    bytes_per_sample: int


def resolve_sigmf_paths(input_path: str) -> Tuple[Path, Path]:
    """
    Resolve both .sigmf-data and .sigmf-meta paths from a base path or either file.
    """
    p = Path(input_path)
    if p.suffix == ".sigmf-data":
        data_path = p
        meta_path = p.with_suffix(".sigmf-meta")
    elif p.suffix == ".sigmf-meta":
        meta_path = p
        data_path = p.with_suffix(".sigmf-data")
    else:
        data_path = Path(str(p) + ".sigmf-data")
        meta_path = Path(str(p) + ".sigmf-meta")

    if not meta_path.is_file():
        raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")

    return data_path, meta_path


def read_sigmf_metadata(input_path: str) -> SigMFMetadata:
    """
    Parse and validate SigMF metadata and verify binary file size alignment.
    """
    data_path, meta_path = resolve_sigmf_paths(input_path)

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in SigMF metadata file {meta_path}: {exc}") from exc

    global_info = meta.get("global", {})
    captures = meta.get("captures", [])

    if not captures or not isinstance(captures, list):
        raise ValueError(f"SigMF metadata {meta_path} is missing a valid 'captures' list.")

    datatype = global_info.get("core:datatype", "")
    if datatype not in SUPPORTED_DATATYPES:
        raise ValueError(
            f"Unsupported SigMF datatype '{datatype}'. "
            f"Supported types: {list(SUPPORTED_DATATYPES.keys())}"
        )

    sample_rate = global_info.get("core:sample_rate")
    if sample_rate is None or float(sample_rate) <= 0.0:
        raise ValueError(f"Invalid or missing 'core:sample_rate': {sample_rate}")
    sample_rate = float(sample_rate)

    first_capture = captures[0]
    center_freq = first_capture.get("core:frequency")
    if center_freq is None or float(center_freq) <= 0.0:
        raise ValueError(f"Invalid or missing 'core:frequency' in capture 0: {center_freq}")
    center_freq = float(center_freq)

    datetime_str = str(first_capture.get("core:datetime", "Unknown"))
    author = str(global_info.get("core:author", ""))
    description = str(global_info.get("core:description", ""))
    recorder = str(global_info.get("core:recorder", "Unknown"))
    version = str(global_info.get("core:version", "Unknown"))

    comp_dtype, components, _ = SUPPORTED_DATATYPES[datatype]
    bytes_per_sample = comp_dtype.itemsize * components

    file_size = os.path.getsize(data_path)
    if file_size == 0:
        raise ValueError(f"SigMF data file is empty (0 bytes): {data_path}")

    total_samples = file_size // bytes_per_sample
    if total_samples == 0:
        raise ValueError(
            f"SigMF data file ({file_size} bytes) is smaller than a single complex sample "
            f"({bytes_per_sample} bytes)."
        )

    duration_sec = total_samples / sample_rate

    return SigMFMetadata(
        data_path=data_path,
        meta_path=meta_path,
        datatype=datatype,
        sample_rate=sample_rate,
        center_frequency=center_freq,
        datetime_str=datetime_str,
        author=author,
        description=description,
        recorder=recorder,
        version=version,
        total_samples=total_samples,
        duration_sec=duration_sec,
        bytes_per_sample=bytes_per_sample,
    )


def load_iq_samples(
    metadata: SigMFMetadata,
    start_sample: int = 0,
    num_samples: Optional[int] = None,
    duration_sec: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load complex64 IQ samples from a SigMF data file using memory-mapped I/O.

    Parameters
    ----------
    metadata : SigMFMetadata
        Validated metadata returned by read_sigmf_metadata().
    start_sample : int
        Starting complex sample index.
    num_samples : Optional[int]
        Number of complex samples to read. Takes precedence over duration_sec if both given.
    duration_sec : Optional[float]
        Duration in seconds to read if num_samples is not specified.
    normalize : bool
        If True, scales integer formats (e.g. ci16_le) to [-1.0, 1.0) in float32.

    Returns
    -------
    np.ndarray
        1D array of np.complex64 IQ samples.
    """
    if start_sample < 0 or start_sample >= metadata.total_samples:
        raise ValueError(
            f"start_sample ({start_sample}) out of range [0, {metadata.total_samples})."
        )

    available = metadata.total_samples - start_sample
    if num_samples is None:
        if duration_sec is not None:
            if duration_sec <= 0:
                raise ValueError(f"duration_sec must be positive, got {duration_sec}")
            num_samples = min(int(round(duration_sec * metadata.sample_rate)), available)
        else:
            num_samples = available
    else:
        num_samples = min(int(num_samples), available)

    if num_samples <= 0:
        raise ValueError("Requested number of samples to load is 0.")

    comp_dtype, components, scale_factor = SUPPORTED_DATATYPES[metadata.datatype]
    byte_offset = start_sample * metadata.bytes_per_sample

    # Memory-map only the requested slice of interleaved I/Q components
    mm = np.memmap(
        metadata.data_path,
        dtype=comp_dtype,
        mode="r",
        offset=byte_offset,
        shape=(num_samples * components,),
    )

    # Allocate single complex64 output buffer to avoid multiple intermediate float arrays
    iq = np.empty(num_samples, dtype=np.complex64)
    iq.real = mm[0::2]
    iq.imag = mm[1::2]

    if normalize and scale_factor != 1.0:
        iq /= np.float32(scale_factor)

    return iq