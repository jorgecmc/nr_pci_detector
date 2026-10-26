"""
visualization.py
Matplotlib diagnostic plots for 5G NR SigMF recordings and PSS/SSS synchronization.
Implements Figures 1 through 5 specified in Section 4.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from sigmf_reader import SigMFMetadata
from nr_pss import PSSCandidate
from nr_sss import SSSDetectionResult
from cell_info import GroupedCellCandidate


def plot_iq_time_domain(
    iq: np.ndarray,
    metadata: SigMFMetadata,
    display_duration_ms: float = 2.0,
    max_plot_points: int = 15000,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 1: I/Q Time-Domain Waveforms (I, Q, and Magnitude vs Time in ms).
    Decimates or slices purely for visual rendering without altering the detector buffer.
    """
    num_disp = min(len(iq), int(round((display_duration_ms * 1e-3) * metadata.sample_rate)))
    if num_disp <= 0:
        num_disp = min(len(iq), max_plot_points)

    step = max(1, num_disp // max_plot_points)
    iq_vis = iq[:num_disp:step]
    t_ms = (np.arange(0, num_disp, step, dtype=np.float64) / metadata.sample_rate) * 1e3
    mag = np.abs(iq_vis)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    fig.suptitle(
        f"Figure 1: 5G NR I/Q Time-Domain Waveforms\n"
        f"Capture Center: {metadata.center_frequency / 1e6:.6f} MHz | "
        f"Sample Rate: {metadata.sample_rate / 1e6:.6f} MS/s | "
        f"Total Capture Duration: {metadata.duration_sec * 1e3:.2f} ms "
        f"(Showing {t_ms[-1]:.2f} ms, step={step})",
        fontsize=11,
        fontweight="bold",
    )

    axes[0].plot(t_ms, iq_vis.real, color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("I (Normalized)")
    axes[0].set_title("In-Phase (I) Component vs. Time", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, iq_vis.imag, color="#ff7f0e", linewidth=0.7)
    axes[1].set_ylabel("Q (Normalized)")
    axes[1].set_title("Quadrature (Q) Component vs. Time", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_ms, mag, color="#2ca02c", linewidth=0.7)
    axes[2].set_ylabel("|IQ| (Normalized)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("Complex Envelope Magnitude |I + jQ| vs. Time", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_iq_constellation(
    iq: np.ndarray,
    metadata: SigMFMetadata,
    num_samples: int = 10000,
    normalize_rms: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 2: Raw Wideband IQ Constellation Diagram.
    Clearly explains why a raw time-domain OFDM waveform appears Gaussian rather than discrete QAM points.
    """
    n_plot = min(len(iq), num_samples)
    # Stride evenly across the buffer for a representative sample
    stride = max(1, len(iq) // n_plot)
    iq_sub = iq[::stride][:n_plot].copy()

    unit_label = "Normalized Full-Scale Units"
    if normalize_rms:
        rms = float(np.sqrt(np.mean(np.abs(iq_sub) ** 2)))
        if rms > 1e-12:
            iq_sub /= rms
            unit_label = "RMS-Normalized Units"

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(iq_sub.real, iq_sub.imag, s=2.5, alpha=0.35, color="#1f77b4", edgecolors="none")

    max_lim = float(np.max(np.abs( np.concatenate([iq_sub.real, iq_sub.imag]) ))) * 1.1
    max_lim = max(max_lim, 1e-3)
    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel(f"In-Phase (I) [{unit_label}]")
    ax.set_ylabel(f"Quadrature (Q) [{unit_label}]")
    ax.set_title(
        f"Figure 2: Raw Wideband Time-Domain IQ Constellation ({n_plot:,} samples)\n"
        "Note: A raw wideband OFDM signal is the IFFT sum of hundreds of independent subcarriers;\n"
        "by the Central Limit Theorem it exhibits a circularly symmetric complex Gaussian distribution\n"
        "until timing synchronization, FFT demodulation, and channel equalization are performed.",
        fontsize=9.5,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_psd(
    iq: np.ndarray,
    metadata: SigMFMetadata,
    ssb_frequencies_hz: List[float],
    ssb_is_detected: bool = True,
    nfft: int = 4096,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 3: Relative Power Spectral Density (PSD) via Welch's Method.
    Marks the capture center frequency, captured bandwidth, and configured/detected SSB band.
    """
    max_psd_samples = min(len(iq), int(0.1 * metadata.sample_rate))
    freqs_baseband, psd_linear = signal.welch(
        iq[:max_psd_samples],
        fs=metadata.sample_rate,
        window="hann",
        nperseg=nfft,
        noverlap=nfft // 2,
        return_onesided=False,
        scaling="density",
    )

    freqs_shifted = np.fft.fftshift(freqs_baseband) + metadata.center_frequency
    psd_shifted = np.fft.fftshift(psd_linear)
    psd_db_rel = 10.0 * np.log10(np.maximum(psd_shifted, 1e-20))

    freqs_mhz = freqs_shifted / 1e6
    cf_mhz = metadata.center_frequency / 1e6
    f_min_mhz = (metadata.center_frequency - metadata.sample_rate / 2.0) / 1e6
    f_max_mhz = (metadata.center_frequency + metadata.sample_rate / 2.0) / 1e6

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(freqs_mhz, psd_db_rel, color="#004c6d", linewidth=0.9, label="Welch Relative PSD")

    ax.axvline(
        cf_mhz,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"Capture Center ({cf_mhz:.6f} MHz)",
    )

    ssb_label_prefix = "Detected SSB" if ssb_is_detected else "Configured SSB"
    for idx, ssb_f in enumerate(ssb_frequencies_hz):
        ssb_mhz = ssb_f / 1e6
        # 20 RBs * 12 subcarriers * 30 kHz = 7.2 MHz bandwidth
        ssb_bw_mhz = 7.2
        ax.axvline(
            ssb_mhz,
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.5,
            label=f"{ssb_label_prefix} #{idx+1} ({ssb_mhz:.6f} MHz)",
        )
        ax.axvspan(
            ssb_mhz - ssb_bw_mhz / 2.0,
            ssb_mhz + ssb_bw_mhz / 2.0,
            color="#2ca02c",
            alpha=0.15,
            label="SSB 20-RB Bandwidth (7.2 MHz)" if idx == 0 else None,
        )

    ax.set_xlim(f_min_mhz, f_max_mhz)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Uncalibrated Relative PSD (dBFS/Hz)")
    ax.set_title(
        f"Figure 3: Welch Power Spectral Density (NFFT={nfft}, Fs={metadata.sample_rate / 1e6:.4f} MS/s)\n"
        f"Captured Span: [{f_min_mhz:.3f} MHz to {f_max_mhz:.3f} MHz] — Uncalibrated Relative Power (dBFS/Hz)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pss_results(
    corr_traces: Dict[int, np.ndarray],
    proc_fs: float,
    candidates: List[PSSCandidate],
    max_plot_points: int = 25000,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 4: PSS Detection Results (Normalized Correlation vs Time for N_ID_2 in {0, 1, 2}).
    Uses max-pooling envelope decimation when plotting long recordings so sharp correlation peaks are never lost.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for nid2 in (0, 1, 2):
        ax = axes[nid2]
        trace = corr_traces.get(nid2, np.zeros(100, dtype=np.float32))
        N = len(trace)
        block = max(1, N // max_plot_points)
        trim_len = (N // block) * block
        if block > 1 and trim_len > 0:
            trace_dec = np.max(trace[:trim_len].reshape(-1, block), axis=1)
            t_ms = (np.arange(0, trim_len, block, dtype=np.float64) / proc_fs) * 1e3
        else:
            trace_dec = trace
            t_ms = (np.arange(N, dtype=np.float64) / proc_fs) * 1e3

        ax.plot(t_ms, trace_dec, color=colors[nid2], linewidth=0.8, label=f"PSS Correlation (N_ID_2 = {nid2})")

        nid2_cands = [c for c in candidates if c.n_id_2 == nid2][:6]
        for rank, c in enumerate(nid2_cands):
            c_ms = c.pss_timing_sec * 1e3
            ax.plot(c_ms, c.peak_metric, "ro", markersize=5)
            if rank < 3:
                ax.annotate(
                    f"t={c_ms:.2f}ms\nM={c.peak_metric:.3f} ({c.papr_db:.1f}dB)",
                    xy=(c_ms, c.peak_metric),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6),
                )

        ax.set_ylabel("Norm. Corr [0-1]")
        ax.set_title(f"PSS Candidate Identity N_ID_2 = {nid2}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)

    axes[2].set_xlabel("Time (ms)")
    fig.suptitle(
        "Figure 4: 5G NR Primary Synchronization Signal (PSS) Correlation vs. Time",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_sss_results(
    grouped_cells: List[GroupedCellCandidate],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Figure 5: SSS Detection Results across all 336 N_ID_1 hypotheses for each detected cell candidate.
    Displays strongest candidate, runner-up candidate, selected N_ID_1, and corresponding PCI.
    """
    num_cells = max(1, min(len(grouped_cells), 4))
    fig, axes = plt.subplots(num_cells, 1, figsize=(11, 4.2 * num_cells), squeeze=False)

    if not grouped_cells:
        ax = axes[0, 0]
        ax.text(0.5, 0.5, "No valid SSS candidates detected", ha="center", va="center")
        return fig

    nid1_axis = np.arange(336)

    for idx in range(num_cells):
        ax = axes[idx, 0]
        cell = grouped_cells[idx]
        det: SSSDetectionResult = cell.best_occasion
        scores = det.all_scores

        ax.vlines(nid1_axis, 0, scores, color="#5c768d", alpha=0.6, linewidth=1.0)
        ax.plot(nid1_axis, scores, ".", color="#1f77b4", markersize=3.5)

        # Highlight Strongest Candidate
        ax.plot(
            det.n_id_1,
            det.sss_metric,
            "ro",
            markersize=8,
            label=(
                f"Winner: N_ID_1={det.n_id_1} -> PCI={det.pci} "
                f"(Metric={det.sss_metric:.4f}, Z={det.z_score:.1f})"
            ),
        )
        ax.annotate(
            f"PCI {det.pci}\n(N_ID_1={det.n_id_1}, N_ID_2={det.n_id_2})\nScore={det.sss_metric:.3f}",
            xy=(det.n_id_1, det.sss_metric),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", alpha=0.85),
        )

        # Highlight Runner-Up Candidate
        ax.plot(
            det.runner_up_n_id_1,
            det.runner_up_metric,
            "s",
            color="#ff7f0e",
            markersize=6.5,
            label=(
                f"Runner-up: N_ID_1={det.runner_up_n_id_1} -> PCI={det.runner_up_pci} "
                f"(Metric={det.runner_up_metric:.4f}, Δ={det.peak_to_runner_up_ratio_db:+.2f} dB)"
            ),
        )

        ax.set_xlim(-5, 341)
        ax.set_ylim(0, max(float(det.sss_metric) * 1.22, 0.1))
        ax.set_xlabel("SSS Identity Candidate N_ID_1 (0 .. 335)")
        ax.set_ylabel("Norm. SSS Correlation")
        ax.set_title(
            f"Cell Candidate #{idx+1}: PCI = {cell.pci} (3*{det.n_id_1} + {det.n_id_2}) | "
            f"SSB Freq = {cell.ssb_center_freq_hz / 1e6:.6f} MHz | "
            f"t = {det.pss_candidate.pss_timing_sec * 1e3:.3f} ms | "
            f"Occasions = {len(cell.occasions)}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Figure 5: 5G NR Secondary Synchronization Signal (SSS) Correlation Across All 336 N_ID_1 Candidates",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig