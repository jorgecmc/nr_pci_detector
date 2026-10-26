"""
nr_sss.py
3GPP TS 38.211 NR Secondary Synchronization Signal (SSS) sequence generation,
OFDM symbol extraction, channel-compensated correlation, and PCI calculation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from nr_pss import PSSCandidate, generate_pss_sequence


# Module-level cache for the (3, 336, 127) SSS table
_SSS_TABLE_CACHE: Optional[np.ndarray] = None


@dataclass
class SSSDetectionResult:
    """Result of evaluating all 336 N_ID_1 hypotheses for a PSS candidate."""
    pss_candidate: PSSCandidate
    n_id_2: int
    n_id_1: int
    pci: int
    sss_metric: float                  # Normalized correlation [0, 1] of winning N_ID_1
    runner_up_n_id_1: int              # Second-strongest N_ID_1 candidate
    runner_up_pci: int                 # PCI corresponding to runner-up N_ID_1
    runner_up_metric: float            # Normalized correlation [0, 1] of runner-up
    peak_to_runner_up_ratio_db: float  # Separation in dB between #1 and #2
    z_score: float                     # Standard deviations above the 335 non-winning hypotheses
    all_scores: np.ndarray             # Shape (336,) array of normalized correlation scores
    sss_symbol_sample_offset: int      # Sample offset from PSS start to SSS start at proc_fs
    combined_confidence: str           # Qualitative confidence ("High", "Medium", "Low")


def compute_pci(n_id_1: int, n_id_2: int) -> int:
    """
    Calculate Physical Cell Identity per 3GPP TS 38.211 Section 7.4.2.1:
        N_ID^cell = 3 * N_ID_1 + N_ID_2
    """
    if not (0 <= int(n_id_1) <= 335):
        raise ValueError(f"N_ID_1 must be in 0..335, got {n_id_1}")
    if not (0 <= int(n_id_2) <= 2):
        raise ValueError(f"N_ID_2 must be in 0..2, got {n_id_2}")
    pci = 3 * int(n_id_1) + int(n_id_2)
    assert 0 <= pci <= 1007
    return pci


def generate_sss_base_m_sequences() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the base length-127 m-sequences x0 and x1 per 3GPP TS 38.211 Section 7.4.2.3.1:
        x0(i+7) = (x0(i+4) + x0(i)) mod 2
        x1(i+7) = (x1(i+1) + x1(i)) mod 2
    Initial condition for both:
        [x(6) x(5) x(4) x(3) x(2) x(1) x(0)] = [0 0 0 0 0 0 1]
    """
    x0 = np.zeros(127, dtype=np.int8)
    x1 = np.zeros(127, dtype=np.int8)
    x0[0] = 1
    x1[0] = 1

    for i in range(120):
        x0[i + 7] = (x0[i + 4] + x0[i]) & 1
        x1[i + 7] = (x1[i + 1] + x1[i]) & 1

    return x0, x1


def generate_sss_sequence(n_id_1: int, n_id_2: int) -> np.ndarray:
    """
    Generate the 127-element NR SSS sequence d_SSS(n) per 3GPP TS 38.211 Section 7.4.2.3.1.
    """
    if not (0 <= n_id_1 <= 335):
        raise ValueError(f"N_ID_1 must be in [0, 335], got {n_id_1}")
    if n_id_2 not in (0, 1, 2):
        raise ValueError(f"N_ID_2 must be in {{0, 1, 2}}, got {n_id_2}")

    x0, x1 = generate_sss_base_m_sequences()
    s0 = 1.0 - 2.0 * x0.astype(np.float64)
    s1 = 1.0 - 2.0 * x1.astype(np.float64)

    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112

    n = np.arange(127)
    d_sss = s0[(n + m0) % 127] * s1[(n + m1) % 127]
    return d_sss


def get_sss_lookup_table() -> np.ndarray:
    """
    Build or return cached (3, 336, 127) table of all valid 3GPP NR SSS sequences.
    Each sequence has length 127 and norm sqrt(127).
    """
    global _SSS_TABLE_CACHE
    if _SSS_TABLE_CACHE is not None:
        return _SSS_TABLE_CACHE

    x0, x1 = generate_sss_base_m_sequences()
    s0 = 1.0 - 2.0 * x0.astype(np.float64)
    s1 = 1.0 - 2.0 * x1.astype(np.float64)

    table = np.empty((3, 336, 127), dtype=np.float64)
    n = np.arange(127)

    for nid2 in range(3):
        for nid1 in range(336):
            m0 = 15 * (nid1 // 112) + 5 * nid2
            m1 = nid1 % 112
            table[nid2, nid1, :] = s0[(n + m0) % 127] * s1[(n + m1) % 127]

    _SSS_TABLE_CACHE = table
    return _SSS_TABLE_CACHE


def extract_active_127_re(
    rx_cfo_corrected: np.ndarray,
    sym_start_idx: int,
    n_fft: int,
    n_cp: int,
) -> np.ndarray:
    """
    Remove the cyclic prefix, perform an N_FFT-point FFT, and extract the 127 subcarriers
    corresponding to k = 56..182 of the 240-subcarrier SSB block (centered at DC).
    """
    useful_start = sym_start_idx + n_cp
    useful_td = rx_cfo_corrected[useful_start : useful_start + n_fft]
    fd_unshifted = np.fft.fft(useful_td, n=n_fft)
    fd_shifted = np.fft.fftshift(fd_unshifted)

    center_bin = n_fft // 2
    return fd_shifted[center_bin - 64 : center_bin + 63]


def detect_sss_for_pss_candidate(
    rx_proc: np.ndarray,
    pss_cand: PSSCandidate,
) -> Optional[SSSDetectionResult]:
    """
    Given a detected PSS candidate, extract the associated SSS symbol (2 OFDM symbols
    after PSS), apply CFO compensation and PSS-aided channel phase equalization, and
    correlate against all 336 valid N_ID_1 sequences.
    """
    n_fft = pss_cand.n_fft
    n_cp = pss_cand.n_cp
    sym_len = n_fft + n_cp
    pss_idx = pss_cand.pss_sample_idx_proc
    nid2 = pss_cand.n_id_2

    # Nominal distance from PSS start (l=0) to SSS start (l=2) is 2 * (N_FFT + N_CP) = 548 samples
    nominal_sss_offset = 2 * sym_len
    # Search a small window around nominal_sss_offset to handle extended-CP edge cases or sample drift
    search_offsets = range(nominal_sss_offset - 4, nominal_sss_offset + 9)

    max_needed = pss_idx + nominal_sss_offset + 8 + sym_len
    if pss_idx < 0 or max_needed > len(rx_proc):
        return None

    # Extract a local segment covering PSS (l=0), PBCH (l=1), and SSS (l=2) and compensate CFO
    seg_len = nominal_sss_offset + 8 + sym_len
    local_rx = rx_proc[pss_idx : pss_idx + seg_len].astype(np.complex128)
    t_local = np.arange(seg_len, dtype=np.float64) / pss_cand.proc_sample_rate
    local_rx_corr = local_rx * np.exp(-2j * np.pi * pss_cand.total_cfo_hz * t_local)

    # Extract received PSS frequency-domain subcarriers to estimate the per-subcarrier channel H[k]
    y_pss = extract_active_127_re(local_rx_corr, 0, n_fft, n_cp)
    d_pss = generate_pss_sequence(nid2)
    # Least-squares channel estimate on the 127 active subcarriers: H_est[k] = Y_PSS[k] * d_PSS[k]
    h_est = y_pss * d_pss

    # Smooth channel estimate slightly with a 3-tap moving average across adjacent subcarriers
    # to suppress AWGN while preserving multipath phase ramp
    kernel = np.array([0.25, 0.5, 0.25])
    h_smooth = np.convolve(h_est, kernel, mode="same")
    h_smooth[0] = 0.75 * h_est[0] + 0.25 * h_est[1]
    h_smooth[-1] = 0.75 * h_est[-1] + 0.25 * h_est[-2]
    h_phase = np.exp(-1j * np.angle(h_smooth))

    sss_table = get_sss_lookup_table()[nid2]  # shape (336, 127)
    seq_norm = np.sqrt(127.0)

    best_overall_peak = -1.0
    best_scores = np.zeros(336, dtype=np.float64)
    best_offset = nominal_sss_offset

    for sss_offset in search_offsets:
        y_sss = extract_active_127_re(local_rx_corr, sss_offset, n_fft, n_cp)
        y_energy = float(np.sum(np.abs(y_sss) ** 2))
        if y_energy < 1e-15:
            continue

        # 1. Coherent PSS-channel-equalized correlation: real part of inner product after phase removal
        y_eq = y_sss * h_phase
        y_eq_norm = np.sqrt(float(np.sum(np.abs(y_eq) ** 2))) + 1e-12
        # Since d_SSS is real BPSK, after channel phase equalization the signal energy lies on the real axis
        corr_coherent = (np.abs(np.real(sss_table @ y_eq)) / (seq_norm * y_eq_norm)) ** 2

        # 2. Direct magnitude correlation (in case PSS and SSS experienced phase jump)
        corr_direct = (np.abs(sss_table @ y_sss) / (seq_norm * np.sqrt(y_energy))) ** 2

        scores = np.maximum(corr_coherent, corr_direct)
        peak_score = float(np.max(scores))
        if peak_score > best_overall_peak:
            best_overall_peak = peak_score
            best_scores = scores
            best_offset = sss_offset

    sorted_indices = np.argsort(best_scores)[::-1]
    win_nid1 = int(sorted_indices[0])
    run_nid1 = int(sorted_indices[1])

    win_metric = float(best_scores[win_nid1])
    run_metric = float(best_scores[run_nid1])

    win_pci = compute_pci(win_nid1, nid2)
    run_pci = compute_pci(run_nid1, nid2)

    ratio_db = 10.0 * np.log10(max(win_metric, 1e-12) / max(run_metric, 1e-12))

    # Compute z-score relative to the 335 non-winning hypotheses
    other_scores = best_scores[sorted_indices[1:]]
    mean_other = float(np.mean(other_scores))
    std_other = float(np.std(other_scores)) + 1e-12
    z_score = (win_metric - mean_other) / std_other

    if win_metric >= 0.25 and ratio_db >= 3.0 and pss_cand.peak_metric >= 0.12:
        confidence = "High (Strong PSS/SSS peak separation)"
    elif win_metric >= 0.12 and ratio_db >= 1.8 and z_score >= 4.5:
        confidence = "Medium (Clear SSS peak above background floor)"
    else:
        confidence = "Low (Marginal SNR / requires verification)"

    return SSSDetectionResult(
        pss_candidate=pss_cand,
        n_id_2=nid2,
        n_id_1=win_nid1,
        pci=win_pci,
        sss_metric=win_metric,
        runner_up_n_id_1=run_nid1,
        runner_up_pci=run_pci,
        runner_up_metric=run_metric,
        peak_to_runner_up_ratio_db=ratio_db,
        z_score=z_score,
        all_scores=best_scores.astype(np.float32),
        sss_symbol_sample_offset=best_offset,
        combined_confidence=confidence,
    )