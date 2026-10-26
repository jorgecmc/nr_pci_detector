"""
tests/test_pci_detector.py
Unit and synthetic-signal integration tests for the 5G NR PCI Detector.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from sigmf_reader import read_sigmf_metadata, load_iq_samples
from nr_pss import (
    generate_pss_sequence,
    get_ssb_ofdm_params,
    ddc_and_resample,
    detect_pss_at_frequency,
)
from nr_sss import (
    generate_sss_sequence,
    compute_pci,
    detect_sss_for_pss_candidate,
)
from cell_info import group_ssb_detections


def synthesize_nr_wideband_recording(
    fs_raw: float,
    capture_center_hz: float,
    ssb_center_hz: float,
    scs_hz: float,
    duration_sec: float,
    cells: list,
    snr_db: float = 12.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a realistic synthetic 5G NR wideband IQ recording containing one or more
    SS/PBCH blocks modulated onto a high-sample-rate carrier (e.g., 61.44 MS/s) with
    configurable PCI, timing offset, frequency offset, amplitude, and AWGN.

    Each item in `cells` is a dict:
        {"pci": int, "timing_sec": float, "cfo_hz": float, "amplitude": float}
    """
    rng = np.random.default_rng(seed)
    total_samples = int(round(duration_sec * fs_raw))

    # Determine OFDM FFT size at raw sample rate (e.g. 61.44 MS/s / 30 kHz = 2048)
    n_fft_raw = int(round(fs_raw / scs_hz))
    n_cp_raw = (18 * n_fft_raw) // 256
    sym_len_raw = n_fft_raw + n_cp_raw

    # Complex AWGN background noise
    noise_std = 10.0 ** (-snr_db / 20.0)
    rx = (
        rng.normal(0, noise_std / np.sqrt(2), total_samples)
        + 1j * rng.normal(0, noise_std / np.sqrt(2), total_samples)
    ).astype(np.complex64)

    ssb_offset_hz = ssb_center_hz - capture_center_hz

    for cell in cells:
        pci = cell["pci"]
        n_id_1 = pci // 3
        n_id_2 = pci % 3
        timing_sec = cell["timing_sec"]
        cfo_hz = cell.get("cfo_hz", 0.0)
        amp = cell.get("amplitude", 1.0)

        d_pss = generate_pss_sequence(n_id_2)
        d_sss = generate_sss_sequence(n_id_1, n_id_2)

        # Build 4-symbol SS/PBCH block at baseband: l=0 (PSS), l=1 (PBCH), l=2 (SSS), l=3 (PBCH)
        ssb_symbols = []
        center_bin = n_fft_raw // 2
        for sym_idx in range(4):
            grid_shifted = np.zeros(n_fft_raw, dtype=np.complex128)
            if sym_idx == 0:
                grid_shifted[center_bin - 64 : center_bin + 63] = d_pss
            elif sym_idx == 2:
                grid_shifted[center_bin - 64 : center_bin + 63] = d_sss
            else:
                # Random QPSK payload on all 240 PBCH subcarriers
                qpsk = (
                    rng.choice([-1.0, 1.0], 240) + 1j * rng.choice([-1.0, 1.0], 240)
                ) / np.sqrt(2.0)
                grid_shifted[center_bin - 120 : center_bin + 120] = qpsk

            td_useful = np.fft.ifft(np.fft.ifftshift(grid_shifted)) * np.sqrt(n_fft_raw)
            td_sym = np.concatenate([td_useful[-n_cp_raw:], td_useful])
            ssb_symbols.append(td_sym)

        ssb_waveform = np.concatenate(ssb_symbols)
        start_idx = int(round(timing_sec * fs_raw))
        if start_idx + len(ssb_waveform) <= total_samples:
            t_vec = (np.arange(len(ssb_waveform), dtype=np.float64) + start_idx) / fs_raw
            total_freq_shift = ssb_offset_hz + cfo_hz
            upconverted = amp * ssb_waveform * np.exp(2j * np.pi * total_freq_shift * t_vec)
            rx[start_idx : start_idx + len(ssb_waveform)] += upconverted.astype(np.complex64)

    return rx


def test_sigmf_metadata_and_ci16_le_conversion(tmp_path: Path):
    """Test 1 & 2: SigMF metadata parsing and ci16_le interleaved IQ conversion."""
    meta_file = tmp_path / "test_rec.sigmf-meta"
    data_file = tmp_path / "test_rec.sigmf-data"

    meta_dict = {
        "annotations": [],
        "captures": [
            {
                "core:datetime": "2026-10-26T06:53:04.156Z",
                "core:frequency": 3463459998.0,
                "core:sample_start": 0,
            }
        ],
        "global": {
            "core:author": "Unit Test",
            "core:datatype": "ci16_le",
            "core:description": "Test vector",
            "core:recorder": "Maia SDR v0.5.2",
            "core:sample_rate": 61439999.0,
            "core:version": "1.0.0",
        },
    }
    meta_file.write_text(json.dumps(meta_dict), encoding="utf-8")

    # Write 4 interleaved ci16_le samples: (I0, Q0, I1, Q1, ...)
    raw_i16 = np.array([16384, -16384, 32767, -32768, 0, 8192, -4096, 4096], dtype="<i2")
    raw_i16.tofile(data_file)

    meta = read_sigmf_metadata(str(data_file))
    assert meta.total_samples == 4
    assert meta.datatype == "ci16_le"
    assert np.isclose(meta.center_frequency, 3463459998.0)
    assert np.isclose(meta.sample_rate, 61439999.0)

    iq_norm = load_iq_samples(meta, normalize=True)
    assert len(iq_norm) == 4
    assert np.isclose(iq_norm[0].real, 0.5, atol=1e-4)
    assert np.isclose(iq_norm[0].imag, -0.5, atol=1e-4)
    assert np.isclose(iq_norm[1].imag, -1.0, atol=1e-4)


def test_pss_sequence_generation():
    """Test 3: Verify 3GPP TS 38.211 PSS sequence properties for N_ID_2 in {0, 1, 2}."""
    seqs = [generate_pss_sequence(nid2) for nid2 in (0, 1, 2)]
    for s in seqs:
        assert len(s) == 127
        assert set(np.unique(s)) == {-1.0, 1.0}
        # m-sequence balance property: 64 ones (-1.0) and 63 zeros (+1.0) -> sum is -1.0
        assert np.isclose(np.sum(s), -1.0)

    # Verify cyclic shift relationship: d_PSS^(1)(n) == d_PSS^(0)((n + 43) mod 127)
    n = np.arange(127)
    assert np.allclose(seqs[1], seqs[0][(n + 43) % 127])
    assert np.allclose(seqs[2], seqs[0][(n + 86) % 127])


def test_sss_sequence_generation_and_pci():
    """Test 4 & 5: Verify 3GPP TS 38.211 SSS sequences and PCI calculation."""
    # Check verification targets from prompt
    assert compute_pci(331, 2) == 995
    assert compute_pci(256, 0) == 768
    assert compute_pci(0, 0) == 0
    assert compute_pci(335, 2) == 1007

    s_995 = generate_sss_sequence(331, 2)
    s_768 = generate_sss_sequence(256, 0)
    assert len(s_995) == 127
    assert set(np.unique(s_995)) == {-1.0, 1.0}
    # Distinct SSS sequences must have low cross-correlation
    xcorr = np.abs(np.dot(s_995, s_768)) / 127.0
    assert xcorr < 0.25


def test_synthetic_dual_pci_detection():
    """
    Test 6 & 7: End-to-end synthetic test at 61.44 MS/s with SSB +220.002 kHz above
    capture center containing BOTH verification candidates:
      - PCI 995 (N_ID_1=331, N_ID_2=2) at t = 2.0 ms and t = 22.0 ms with +1.8 kHz CFO
      - PCI 768 (N_ID_1=256, N_ID_2=0) at t = 6.5 ms with -1.2 kHz CFO (weaker secondary cell)
    """
    fs_raw = 61.44e6
    capture_center_hz = 3463.459998e6
    ssb_center_hz = 3463.680e6
    scs_hz = 30e3

    cells = [
        {"pci": 995, "timing_sec": 0.0020, "cfo_hz": 1800.0, "amplitude": 1.0},
        {"pci": 995, "timing_sec": 0.0220, "cfo_hz": 1800.0, "amplitude": 1.0},
        {"pci": 768, "timing_sec": 0.0065, "cfo_hz": -1200.0, "amplitude": 0.65},
    ]

    iq = synthesize_nr_wideband_recording(
        fs_raw=fs_raw,
        capture_center_hz=capture_center_hz,
        ssb_center_hz=ssb_center_hz,
        scs_hz=scs_hz,
        duration_sec=0.030,
        cells=cells,
        snr_db=10.0,
    )

    proc_fs, _, _, _ = get_ssb_ofdm_params(scs_hz=scs_hz)
    rx_proc = ddc_and_resample(
        iq=iq,
        fs_in=fs_raw,
        freq_offset_hz=ssb_center_hz - capture_center_hz,
        fs_out=proc_fs,
    )

    pss_cands, _ = detect_pss_at_frequency(
        rx_proc=rx_proc,
        raw_fs=fs_raw,
        proc_fs=proc_fs,
        ssb_center_freq_hz=ssb_center_hz,
        scs_hz=scs_hz,
        cfo_search_range_hz=15e3,
    )

    sss_results = []
    for cand in pss_cands:
        res = detect_sss_for_pss_candidate(rx_proc, cand)
        if res is not None:
            sss_results.append(res)

    grouped = group_ssb_detections(sss_results)
    detected_pcis = {g.pci: g for g in grouped}

    assert 995 in detected_pcis, f"Expected PCI 995 in {detected_pcis.keys()}"
    assert 768 in detected_pcis, f"Expected PCI 768 in {detected_pcis.keys()}"

    cell_995 = detected_pcis[995]
    assert cell_995.n_id_1 == 331
    assert cell_995.n_id_2 == 2
    assert len(cell_995.occasions) == 2
    assert np.isclose(cell_995.estimated_periodicity_ms, 20.0, atol=0.1)
    assert np.isclose(cell_995.mean_cfo_hz, 1800.0, atol=600.0)

    cell_768 = detected_pcis[768]
    assert cell_768.n_id_1 == 256
    assert cell_768.n_id_2 == 0