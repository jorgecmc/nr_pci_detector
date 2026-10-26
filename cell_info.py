"""
cell_info.py
Multi-occasion SSB grouping, periodicity estimation, and structured cell reporting.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from sigmf_reader import SigMFMetadata
from nr_sss import SSSDetectionResult


@dataclass
class GroupedCellCandidate:
    """Aggregated information for a unique (PCI, SSB Center Frequency) cell detection."""
    pci: int
    n_id_1: int
    n_id_2: int
    ssb_center_freq_hz: float
    scs_hz: float
    best_occasion: SSSDetectionResult
    occasions: List[SSSDetectionResult] = field(default_factory=list)
    mean_cfo_hz: float = 0.0
    estimated_periodicity_ms: Optional[float] = None


def infer_nr_band(freq_hz: float) -> str:
    """
    Infer common FR1 5G NR operating band from frequency (3GPP TS 38.104 Table 5.2-1).
    """
    freq_mhz = freq_hz / 1e6
    if 3300.0 <= freq_mhz <= 3800.0:
        return "n78 (3300 - 3800 MHz TDD)"
    if 3700.0 < freq_mhz <= 4200.0:
        return "n77 (3300 - 4200 MHz TDD)"
    if 4400.0 <= freq_mhz <= 5000.0:
        return "n79 (4400 - 5000 MHz TDD)"
    if 2496.0 <= freq_mhz <= 2690.0:
        return "n41 (2496 - 2690 MHz TDD)"
    if 617.0 <= freq_mhz <= 652.0:
        return "n71 (600 MHz FDD)"
    return "Unknown / Custom Band"


def group_ssb_detections(
    detections: List[SSSDetectionResult],
    min_sss_metric: float = 0.09,
    min_z_score: float = 3.8,
    freq_tolerance_hz: float = 90e3,
) -> List[GroupedCellCandidate]:
    """
    Filter spurious correlation peaks and group repeated SSB occasions from the same cell
    by (PCI, SSB Center Frequency). Retains weaker secondary cells (e.g., neighbor cells
    on the same SSB frequency) as separate GroupedCellCandidates.
    """
    valid = [
        d for d in detections
        if d.sss_metric >= min_sss_metric and d.z_score >= min_z_score
    ]
    # If nothing passed the strict filter, keep the top candidate for diagnostic reporting
    if not valid and detections:
        valid = [max(detections, key=lambda d: d.sss_metric * d.pss_candidate.peak_metric)]

    groups: List[GroupedCellCandidate] = []

    for det in valid:
        matched_group: Optional[GroupedCellCandidate] = None
        for grp in groups:
            if (
                grp.pci == det.pci
                and abs(grp.ssb_center_freq_hz - det.pss_candidate.ssb_center_freq_hz) <= freq_tolerance_hz
            ):
                matched_group = grp
                break

        if matched_group is None:
            new_grp = GroupedCellCandidate(
                pci=det.pci,
                n_id_1=det.n_id_1,
                n_id_2=det.n_id_2,
                ssb_center_freq_hz=det.pss_candidate.ssb_center_freq_hz,
                scs_hz=det.pss_candidate.proc_sample_rate / det.pss_candidate.n_fft,
                best_occasion=det,
                occasions=[det],
            )
            groups.append(new_grp)
        else:
            # Avoid duplicate triggers on the exact same OFDM symbol
            is_duplicate_time = any(
                abs(det.pss_candidate.pss_timing_sec - existing.pss_candidate.pss_timing_sec) < 15e-6
                for existing in matched_group.occasions
            )
            if not is_duplicate_time:
                matched_group.occasions.append(det)
                if (
                    det.sss_metric * det.pss_candidate.peak_metric
                    > matched_group.best_occasion.sss_metric * matched_group.best_occasion.pss_candidate.peak_metric
                ):
                    matched_group.best_occasion = det

    # Sort occasions chronologically within each group and compute periodicity & mean CFO
    for grp in groups:
        grp.occasions.sort(key=lambda d: d.pss_candidate.pss_timing_sec)
        weights = np.array([d.sss_metric for d in grp.occasions], dtype=np.float64)
        cfos = np.array([d.pss_candidate.total_cfo_hz for d in grp.occasions], dtype=np.float64)
        grp.mean_cfo_hz = float(np.average(cfos, weights=weights))

        if len(grp.occasions) >= 2:
            times_ms = np.array([d.pss_candidate.pss_timing_sec * 1e3 for d in grp.occasions])
            diffs_ms = np.diff(times_ms)
            # Filter inter-burst intervals (> 3.5 ms) vs intra-burst beam sweeps (< 2.5 ms)
            burst_diffs = diffs_ms[diffs_ms >= 3.5]
            if len(burst_diffs) > 0:
                grp.estimated_periodicity_ms = float(np.median(burst_diffs))

    # Sort groups by combined PSS*SSS score descending
    groups.sort(
        key=lambda g: g.best_occasion.sss_metric * g.best_occasion.pss_candidate.peak_metric,
        reverse=True,
    )
    return groups


def format_cell_report(
    metadata: SigMFMetadata,
    analyzed_samples: int,
    processing_time_sec: float,
    grouped_cells: List[GroupedCellCandidate],
    all_occasions: List[SSSDetectionResult],
    ssb_mode_desc: str,
    verification_targets: tuple[int, ...] = (995, 768),
) -> str:
    """
    Generate the structured 5G NR Cell Detection report required by Section 9 & Section 11.
    """
    lines: List[str] = []
    analyzed_dur = analyzed_samples / metadata.sample_rate

    lines.append("========== 5G NR CELL DETECTION ==========")
    lines.append("")
    lines.append("Capture Information")
    lines.append("-------------------")
    lines.append(f"Capture Center Frequency : {metadata.center_frequency / 1e6:.6f} MHz")
    lines.append(f"Sample Rate              : {metadata.sample_rate / 1e6:.6f} MS/s")
    lines.append(
        f"Recording Duration       : {metadata.duration_sec:.4f} s total "
        f"({analyzed_dur:.4f} s / {analyzed_samples:,} samples analyzed)"
    )
    lines.append(f"IQ Format                : {metadata.datatype}")
    lines.append(f"Capture Timestamp        : {metadata.datetime_str}")
    lines.append(f"Operating Mode           : {ssb_mode_desc}")
    lines.append(f"Processing Time          : {processing_time_sec:.3f} s")
    lines.append("")

    if not grouped_cells:
        lines.append("Detected NR Signal")
        lines.append("------------------")
        lines.append("No valid 5G NR SSB/PCI candidates exceeded detection thresholds.")
        lines.append("==========================================")
        return "\n".join(lines)

    for idx, cell in enumerate(grouped_cells, start=1):
        best = cell.best_occasion
        pss = best.pss_candidate
        actual_ssb_freq_hz = cell.ssb_center_freq_hz + cell.mean_cfo_hz
        offset_from_capture_khz = (actual_ssb_freq_hz - metadata.center_frequency) / 1e3

        lines.append(f"Detected NR Signal (Cell Candidate #{idx} of {len(grouped_cells)})")
        lines.append("--------------------------------------------------")
        lines.append(f"NR Band                  : {infer_nr_band(cell.ssb_center_freq_hz)}")
        lines.append(
            f"SSB Center Frequency     : {cell.ssb_center_freq_hz / 1e6:.6f} MHz (nominal) | "
            f"{actual_ssb_freq_hz / 1e6:.6f} MHz (estimated w/ CFO, {offset_from_capture_khz:+.3f} kHz vs capture center)"
        )
        lines.append(f"NR RF Channel Center     : Unknown (requires SIB1 offsetToPointA / k_SSB)")
        lines.append(f"SSB Subcarrier Spacing   : {cell.scs_hz / 1e3:.1f} kHz")
        lines.append(f"PSS Identity (N_ID_2)    : {cell.n_id_2}")
        lines.append(f"SSS Identity (N_ID_1)    : {cell.n_id_1} (Runner-up: {best.runner_up_n_id_1}, PCI {best.runner_up_pci})")
        lines.append(f"Physical Cell ID (PCI)   : {cell.pci}")
        lines.append(
            f"PSS Timing (Best Peak)   : {pss.pss_timing_sec * 1e3:.4f} ms "
            f"(Raw sample #{pss.pss_sample_idx_raw:,})"
        )
        lines.append(
            f"SSS Timing (Best Peak)   : {(pss.pss_timing_sec + best.sss_symbol_sample_offset / pss.proc_sample_rate) * 1e3:.4f} ms "
            f"(+{best.sss_symbol_sample_offset} proc samples after PSS)"
        )
        lines.append(f"Frequency Offset         : {cell.mean_cfo_hz:+.2f} Hz (relative to {cell.ssb_center_freq_hz / 1e6:.6f} MHz)")
        lines.append(f"PSS Correlation Metric   : {pss.peak_metric:.4f} (PAPR: {pss.papr_db:.2f} dB)")
        lines.append(
            f"SSS Correlation Metric   : {best.sss_metric:.4f} "
            f"(Runner-up: {best.runner_up_metric:.4f}, Separation: +{best.peak_to_runner_up_ratio_db:.2f} dB, Z-score: {best.z_score:.1f})"
        )
        lines.append(f"Detection Confidence     : {best.combined_confidence}")
        lines.append(f"Observed SSB Occasions   : {len(cell.occasions)}")
        if cell.estimated_periodicity_ms is not None:
            lines.append(f"Est. SSB Periodicity     : ~{cell.estimated_periodicity_ms:.2f} ms")
        else:
            lines.append("Est. SSB Periodicity     : Unknown (single burst or insufficient duration)")
        lines.append("")
        lines.append("Additional Information")
        lines.append("----------------------")
        lines.append("SSB Index                : Unknown unless successfully decoded (requires PBCH DMRS)")
        lines.append("System Frame Number      : Unknown unless successfully decoded (requires PBCH MIB)")
        lines.append("MIB                      : Not decoded")
        lines.append("SIB1                     : Not decoded")
        lines.append("NR Cell Identity (NCI)   : Unknown")
        lines.append("PLMN / Operator          : Unknown")
        lines.append("Tracking Area Code       : Unknown")
        lines.append("")

    # Individual synchronization occasions summary table
    lines.append("Individual Synchronization Occasions")
    lines.append("------------------------------------")
    lines.append(
        f"{'#':>3} | {'Time (ms)':>10} | {'PCI':>4} | {'N_ID_1':>6} | {'N_ID_2':>6} | "
        f"{'PSS Corr':>8} | {'SSS Corr':>8} | {'Sep (dB)':>8} | {'CFO (Hz)':>9}"
    )
    lines.append("-" * 86)
    for i, occ in enumerate(all_occasions[:25], start=1):
        p = occ.pss_candidate
        lines.append(
            f"{i:>3} | {p.pss_timing_sec * 1e3:>10.4f} | {occ.pci:>4} | {occ.n_id_1:>6} | "
            f"{occ.n_id_2:>6} | {p.peak_metric:>8.4f} | {occ.sss_metric:>8.4f} | "
            f"{occ.peak_to_runner_up_ratio_db:>+8.2f} | {p.total_cfo_hz:>+9.1f}"
        )
    if len(all_occasions) > 25:
        lines.append(f"... ({len(all_occasions) - 25} additional occasions omitted)")
    lines.append("")

    # Verification target comparison
    detected_pcis = {c.pci for c in grouped_cells}
    matched_targets = [t for t in verification_targets if t in detected_pcis]
    lines.append("Verification Target Check")
    lines.append("-------------------------")
    lines.append(f"Target PCIs Checked      : {list(verification_targets)}")
    lines.append(f"Detected Grouped PCIs    : {sorted(list(detected_pcis))}")
    if matched_targets:
        lines.append(f"Target Match Status      : MATCHED {matched_targets}")
    else:
        lines.append(
            "Target Match Status      : No match with target PCIs in this segment. "
            "Verify SSB frequency, recording time window, or SINR."
        )
    lines.append("==========================================")

    return "\n".join(lines)
