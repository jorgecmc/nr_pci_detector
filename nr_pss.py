"""
nr_pss.py
3GPP TS 38.211 NR Primary Synchronization Signal (PSS) generation,
digital down-conversion, anti-aliased resampling, and FFT-based detection.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class PSSCandidate:
    """Represents a detected PSS synchronization candidate."""
    n_id_2: int
    pss_sample_idx_proc: int       # Sample index at processing sample rate (start of CP)
    pss_sample_idx_raw: int        # Equivalent sample index in original recording
    pss_timing_sec: float          # Time offset from start of analyzed buffer (seconds)
    coarse_cfo_hz: float           # Coarse CFO hypothesis from PSS grid search
    fine_cfo_hz: float             # Fine CFO from cyclic-prefix / sub-grid refinement
    total_cfo_hz: float            # Total estimated frequency offset relative to nominal SSB freq
    peak_metric: float             # Normalized energy correlation score [0, 1]
    papr_db: float                 # Peak-to-average correlation ratio in dB
    ssb_center_freq_hz: float      # Nominal SSB center frequency searched
    proc_sample_rate: float        # Processing sample rate used for OFDM demodulation
    n_fft: int                     # FFT size at processing sample rate
    n_cp: int                      # Normal CP length at processing sample rate


def generate_pss_sequence(n_id_2: int) -> np.ndarray:
    """
    Generate the 127-element NR PSS sequence d_PSS(n) per 3GPP TS 38.211 Section 7.4.2.2.

    Parameters
    ----------
    n_id_2 : int
        PSS identity in {0, 1, 2}.

    Returns
    -------
    np.ndarray
        1D array of length 127 containing values in {-1.0, +1.0} (float64).
    """
    if n_id_2 not in (0, 1, 2):
        raise ValueError(f"N_ID_2 must be 0, 1, or 2; got {n_id_2}")

    x = np.zeros(127, dtype=np.int8)
    # [x(6) x(5) x(4) x(3) x(2) x(1) x(0)] = [1 1 1 0 1 1 0]
    x[:7] = [0, 1, 1, 0, 1, 1, 1]

    for i in range(120):
        x[i + 7] = (x[i + 4] + x[i]) & 1

    n = np.arange(127)
    m = (n + 43 * n_id_2) % 127
    d_pss = 1.0 - 2.0 * x[m].astype(np.float64)
    return d_pss


def get_ssb_ofdm_params(scs_hz: float = 30e3, n_fft: int = 256) -> Tuple[float, int, int, int]:
    """
    Return canonical OFDM parameters for SSB processing.
    For 240 subcarriers (20 RBs), N_FFT = 256 provides full coverage with guard bands.
    - At SCS = 30 kHz, proc_fs = 256 * 30 kHz = 7.68 MS/s, N_CP = 18, N_CP_0 = 22.
    - At SCS = 15 kHz, proc_fs = 256 * 15 kHz = 3.84 MS/s, N_CP = 18, N_CP_0 = 20.
    """
    if abs(scs_hz - 30e3) < 1.0:
        proc_fs = float(n_fft) * 30e3
        n_cp = (18 * n_fft) // 256
        n_cp0 = (22 * n_fft) // 256
    elif abs(scs_hz - 15e3) < 1.0:
        proc_fs = float(n_fft) * 15e3
        n_cp = (18 * n_fft) // 256
        n_cp0 = (20 * n_fft) // 256
    else:
        raise ValueError(f"Unsupported SSB subcarrier spacing: {scs_hz} Hz (expected 15e3 or 30e3)")

    return proc_fs, n_fft, n_cp, n_cp0


def synthesize_pss_time_domain(
    n_id_2: int,
    n_fft: int = 256,
    n_cp: int = 18,
    include_cp: bool = True,
) -> np.ndarray:
    """
    Map the 127 PSS subcarriers to an OFDM symbol centered at DC and compute the
    unit-energy time-domain reference waveform.

    Per 3GPP TS 38.211 Table 7.4.3.1-1:
    - SSB occupies k = 0..239, with k = 120 as the SSB center frequency (DC at baseband).
    - PSS occupies k = 56..182 -> relative to SSB center k=120, subcarrier offsets are -64..+62.
    """
    d_pss = generate_pss_sequence(n_id_2)
    grid_shifted = np.zeros(n_fft, dtype=np.complex128)

    center_bin = n_fft // 2
    # Relative subcarrier indices -64 .. +62 inclusive (length 127)
    grid_shifted[center_bin - 64 : center_bin + 63] = d_pss

    grid_unshifted = np.fft.ifftshift(grid_shifted)
    td_useful = np.fft.ifft(grid_unshifted)

    if include_cp and n_cp > 0:
        td_sym = np.concatenate([td_useful[-n_cp:], td_useful])
    else:
        td_sym = td_useful

    # Normalize to unit energy so inner product directly yields energy correlation
    energy = np.sqrt(np.sum(np.abs(td_sym) ** 2))
    return (td_sym / energy).astype(np.complex64)


def ddc_and_resample(
    iq: np.ndarray,
    fs_in: float,
    freq_offset_hz: float,
    fs_out: float,
) -> np.ndarray:
    """
    Shift the target SSB center frequency (`freq_offset_hz` relative to capture center)
    to baseband (0 Hz) and apply anti-aliased rational polyphase resampling to `fs_out`.
    """
    n_samples = len(iq)

    # 1. Digital Down-Conversion (DDC) if SSB is not centered at capture DC
    if abs(freq_offset_hz) > 0.01:
        # Compute phase in float64 to avoid precision loss over tens of millions of samples
        t = np.arange(n_samples, dtype=np.float64) / fs_in
        nco = np.exp(-2j * np.pi * freq_offset_hz * t).astype(np.complex64)
        shifted = iq * nco
    else:
        shifted = iq

    # 2. Check if sample rates match within 1 Hz (e.g., 61.439999 MS/s vs 61.44 MS/s is integer 8:1)
    ratio = fs_in / fs_out
    nearest_int = int(round(ratio))
    if abs(ratio - nearest_int) / ratio < 1e-5:
        if nearest_int == 1:
            return shifted.copy()
        # Exact integer decimation with built-in Kaiser anti-aliasing FIR filter
        resampled = signal.resample_poly(shifted, up=1, down=nearest_int)
        return resampled.astype(np.complex64)

    # General rational approximation for arbitrary capture sample rates
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    resampled = signal.resample_poly(shifted, up=frac.numerator, down=frac.denominator)
    return resampled.astype(np.complex64)


def compute_normalized_pss_correlation(
    rx_proc: np.ndarray,
    ref_td: np.ndarray,
) -> np.ndarray:
    """
    Compute the normalized energy correlation metric M(n) in [0, 1] using FFT convolution:
        M(n) = | sum_{k=0}^{L-1} rx[n+k] * conj(ref[k]) |^2 / sum_{k=0}^{L-1} |rx[n+k]|^2
    where ref has unit energy (sum |ref[k]|^2 = 1).
    """
    L = len(ref_td)
    N = len(rx_proc)
    if N < L:
        raise ValueError(f"Signal length ({N}) is shorter than PSS symbol length ({L}).")

    # Cross-correlation via FFT (valid mode gives N - L + 1 lags)
    xcorr = signal.fftconvolve(rx_proc, ref_td[::-1].conj(), mode="valid")
    num = np.abs(xcorr) ** 2

    # Sliding window signal energy using cumulative sum in float64 for numerical stability
    pow_inst = np.abs(rx_proc).astype(np.float64) ** 2
    csum = np.empty(N + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(pow_inst, out=csum[1:])
    window_energy = csum[L:] - csum[:-L]

    # Guard against zero-energy segments
    eps = 1e-12
    metric = num / np.maximum(window_energy, eps)
    return metric.astype(np.float32)


def estimate_fine_cfo_cp(
    rx_proc: np.ndarray,
    pss_start_idx: int,
    n_fft: int,
    n_cp: int,
    proc_fs: float,
) -> float:
    """
    Estimate fractional cyclic-prefix phase rotation between the CP and the end of the
    useful OFDM symbol:
        df_fine = -angle( sum_{k=0}^{N_cp-1} rx[n+k] * conj(rx[n+k+N_fft]) ) / (2*pi * N_fft / fs)
    Skip the first 2 samples of CP to avoid ISI from the preceding symbol.
    """
    guard = min(2, n_cp // 4)
    if pss_start_idx < 0 or pss_start_idx + n_cp + n_fft > len(rx_proc):
        return 0.0

    cp_seg = rx_proc[pss_start_idx + guard : pss_start_idx + n_cp]
    tail_seg = rx_proc[pss_start_idx + n_fft + guard : pss_start_idx + n_fft + n_cp]
    prod = np.vdot(cp_seg, tail_seg)  # sum(conj(cp_seg) * tail_seg)
    if np.abs(prod) < 1e-12:
        return 0.0

    subcarrier_spacing = proc_fs / n_fft
    fine_cfo_hz = float(np.angle(prod) / (2.0 * np.pi) * subcarrier_spacing)
    return fine_cfo_hz


def detect_pss_at_frequency(
    rx_proc: np.ndarray,
    raw_fs: float,
    proc_fs: float,
    ssb_center_freq_hz: float,
    scs_hz: float = 30e3,
    cfo_search_range_hz: float = 45e3,
    cfo_step_hz: float = 7.5e3,
    threshold_metric: float = 0.06,
    max_candidates_per_nid2: int = 12,
) -> Tuple[List[PSSCandidate], Dict[int, np.ndarray]]:
    """
    Detect PSS candidates across all N_ID_2 in {0, 1, 2} and across a coarse CFO grid.

    Returns
    -------
    candidates : List[PSSCandidate]
        Sorted list of detected PSS synchronization candidates (strongest first).
    corr_traces : Dict[int, np.ndarray]
        Best correlation trace vs sample index for each N_ID_2 in {0, 1, 2} for plotting.
    """
    _, n_fft, n_cp, _ = get_ssb_ofdm_params(scs_hz=scs_hz)
    sym_len = n_fft + n_cp
    N = len(rx_proc)

    if cfo_search_range_hz <= 0:
        cfo_grid = np.array([0.0])
    else:
        cfo_grid = np.arange(-cfo_search_range_hz, cfo_search_range_hz + 0.1, cfo_step_hz)

    t_proc = np.arange(N, dtype=np.float64) / proc_fs
    refs = {
        nid2: synthesize_pss_time_domain(nid2, n_fft=n_fft, n_cp=n_cp, include_cp=True)
        for nid2 in (0, 1, 2)
    }

    corr_traces: Dict[int, np.ndarray] = {}
    all_candidates: List[PSSCandidate] = []

    # Minimum distance between distinct PSS peaks: 1 OFDM symbol (274 samples at 7.68 MS/s)
    min_peak_dist = sym_len

    for nid2 in (0, 1, 2):
        ref_td = refs[nid2]
        best_metric_trace = np.zeros(N - sym_len + 1, dtype=np.float32)
        best_cfo_idx_trace = np.zeros(N - sym_len + 1, dtype=np.int16)

        for c_idx, cfo_hz in enumerate(cfo_grid):
            if abs(cfo_hz) > 0.1:
                # Apply trial CFO compensation to the short reference waveform (much faster than rotating rx_proc)
                t_ref = np.arange(sym_len, dtype=np.float64) / proc_fs
                ref_rot = (ref_td * np.exp(2j * np.pi * cfo_hz * t_ref)).astype(np.complex64)
            else:
                ref_rot = ref_td

            metric = compute_normalized_pss_correlation(rx_proc, ref_rot)
            improved = metric > best_metric_trace
            best_metric_trace[improved] = metric[improved]
            best_cfo_idx_trace[improved] = c_idx

        corr_traces[nid2] = best_metric_trace

        mean_floor = float(np.median(best_metric_trace)) + 1e-12
        # Adaptive threshold: at least 6x median noise floor or user threshold
        eff_thresh = max(threshold_metric, min(0.15, 6.0 * mean_floor))

        peaks, props = signal.find_peaks(
            best_metric_trace,
            height=eff_thresh,
            distance=min_peak_dist,
        )

        if len(peaks) == 0:
            # Always keep the top global peak for diagnostic inspection even if weak
            top_idx = int(np.argmax(best_metric_trace))
            peaks = np.array([top_idx])

        # Sort peaks by correlation height descending
        peak_heights = best_metric_trace[peaks]
        order = np.argsort(peak_heights)[::-1][:max_candidates_per_nid2]

        for p_idx in peaks[order]:
            p_idx = int(p_idx)
            peak_val = float(best_metric_trace[p_idx])
            coarse_cfo = float(cfo_grid[best_cfo_idx_trace[p_idx]])

            # Refine CFO around coarse_cfo using fine grid (+/- cfo_step_hz)
            fine_offsets = np.linspace(-cfo_step_hz * 0.75, cfo_step_hz * 0.75, 13)
            seg = rx_proc[p_idx : p_idx + sym_len]
            seg_energy = float(np.sum(np.abs(seg) ** 2)) + 1e-12
            t_ref = np.arange(sym_len, dtype=np.float64) / proc_fs

            best_sub_cfo = coarse_cfo
            best_sub_val = peak_val
            for df in fine_offsets:
                trial_cfo = coarse_cfo + df
                ref_rot = (ref_td * np.exp(2j * np.pi * trial_cfo * t_ref)).astype(np.complex64)
                val = float(np.abs(np.vdot(ref_rot, seg)) ** 2 / seg_energy)
                if val > best_sub_val:
                    best_sub_val = val
                    best_sub_cfo = trial_cfo

            # Combine with cyclic prefix correlation phase check
            seg_cfo_corr = seg * np.exp(-2j * np.pi * best_sub_cfo * t_ref)
            cp_residual = estimate_fine_cfo_cp(seg_cfo_corr, 0, n_fft, n_cp, proc_fs)
            total_cfo = best_sub_cfo + 0.5 * cp_residual

            papr_db = 10.0 * np.log10(max(best_sub_val, 1e-12) / mean_floor)
            timing_sec = p_idx / proc_fs
            raw_sample_idx = int(round(timing_sec * raw_fs))

            all_candidates.append(
                PSSCandidate(
                    n_id_2=nid2,
                    pss_sample_idx_proc=p_idx,
                    pss_sample_idx_raw=raw_sample_idx,
                    pss_timing_sec=timing_sec,
                    coarse_cfo_hz=coarse_cfo,
                    fine_cfo_hz=total_cfo - coarse_cfo,
                    total_cfo_hz=total_cfo,
                    peak_metric=best_sub_val,
                    papr_db=papr_db,
                    ssb_center_freq_hz=ssb_center_freq_hz,
                    proc_sample_rate=proc_fs,
                    n_fft=n_fft,
                    n_cp=n_cp,
                )
            )

    all_candidates.sort(key=lambda c: c.peak_metric, reverse=True)
    return all_candidates, corr_traces

def get_gscn_frequencies_in_span(center_freq_hz: float, span_hz: float) -> np.ndarray:
    """Calculate all the ligal 3GPP TS 38.104 GSCN frequencies in the acquired band."""
    f_min = center_freq_hz - span_hz / 2.0
    f_max = center_freq_hz + span_hz / 2.0
    gscn_freqs = []

    if f_min < 3000e6:
        # TS 38.104: F_SSB = N * 1200 kHz + M * 50 kHz, M in {1, 3, 5}
        n_min = int(np.floor((f_min - 250e3) / 1200e3))
        n_max = int(np.ceil((f_max - 50e3) / 1200e3))
        for n in range(n_min, n_max + 1):
            for m in (1, 3, 5):
                f = n * 1200e3 + m * 50e3
                if f_min <= f <= f_max:
                    gscn_freqs.append(f)

    if f_max >= 3000e6:
        # TS 38.104: F_SSB = 3000 MHz + N * 1.44 MHz
        n_min = int(np.floor((max(f_min, 3000e6) - 3000e6) / 1440e3))
        n_max = int(np.ceil((f_max - 3000e6) / 1440e3))
        for n in range(n_min, n_max + 1):
            f = 3000e6 + n * 1440e3
            if f_min <= f <= f_max:
                gscn_freqs.append(f)

    return np.array(sorted(list(set(gscn_freqs))))

def blind_search_ssb_frequencies(
    iq: np.ndarray,
    capture_center_freq_hz: float,
    raw_fs: float,
    scs_hz: float = 30e3,
    freq_step_hz: float = 360e3,
    search_fraction_bw: float = 0.80,
    max_search_samples: Optional[int] = None,
) -> List[Tuple[float, float, int]]:
    """
    Scan the captured bandwidth for candidate SSB center frequencies (--search-ssb mode).
    Uses a fast block-FFT frequency-domain shift and PSS correlation over a short slice
    (e.g., 25 ms, enough for at least one 20 ms SSB burst period) to rank candidate
    SSB frequencies before running full-precision DDC + SSS detection.

    Returns
    -------
    List of (candidate_ssb_freq_hz, best_pss_metric, best_nid2) sorted by metric descending.
    """
    proc_fs, n_fft, n_cp, _ = get_ssb_ofdm_params(scs_hz=scs_hz)
    # 25 ms is guaranteed to contain at least one SSB burst (default periodicity = 20 ms)
    slice_len = min(len(iq), int(0.025 * raw_fs))
    if max_search_samples is not None:
        slice_len = min(slice_len, max_search_samples)
    iq_slice = iq[:slice_len]

    usable_bw = raw_fs * search_fraction_bw
    ssb_bw = 240.0 * scs_hz  # 7.2 MHz for 30 kHz SCS
    max_offset = max(0.0, (usable_bw - ssb_bw) / 2.0)

    #offsets = np.arange(-max_offset, max_offset + 1.0, freq_step_hz)
    gscn_candidates = get_gscn_frequencies_in_span(capture_center_freq_hz, usable_bw - ssb_bw)
    if len(gscn_candidates) > 0:
        offsets = gscn_candidates - capture_center_freq_hz
    else:
        offsets = np.arange(-max_offset, max_offset + 1.0, freq_step_hz)

    refs = {
        nid2: synthesize_pss_time_domain(nid2, n_fft=n_fft, n_cp=n_cp, include_cp=True)
        for nid2 in (0, 1, 2)
    }

    scored_freqs: List[Tuple[float, float, int]] = []
    for offset_hz in offsets:
        rx_proc = ddc_and_resample(iq_slice, raw_fs, float(offset_hz), proc_fs)
        best_val = 0.0
        best_nid2 = 0
        for nid2, ref_td in refs.items():
            m = compute_normalized_pss_correlation(rx_proc, ref_td)
            val = float(np.max(m))
            if val > best_val:
                best_val = val
                best_nid2 = nid2
        scored_freqs.append((capture_center_freq_hz + float(offset_hz), best_val, best_nid2))

    scored_freqs.sort(key=lambda item: item[1], reverse=True)
    return scored_freqs
