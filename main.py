#!/usr/bin/env python3
"""
main.py
Command-line entry point for the 5G NR Physical Cell Identity (PCI) Detector
using 3GPP TS 38.211 PSS and SSS synchronization.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

from sigmf_reader import read_sigmf_metadata, load_iq_samples
from nr_pss import (
    get_ssb_ofdm_params,
    ddc_and_resample,
    detect_pss_at_frequency,
    blind_search_ssb_frequencies,
    PSSCandidate,
)
from nr_sss import detect_sss_for_pss_candidate, SSSDetectionResult
from cell_info import group_ssb_detections, format_cell_report
from visualization import (
    plot_iq_time_domain,
    plot_iq_constellation,
    plot_psd,
    plot_pss_results,
    plot_sss_results,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5G NR Physical Cell Identity (PCI) Detector using 3GPP PSS and SSS."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to recording.sigmf-data or recording.sigmf-meta",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ssb-frequency",
        type=float,
        default=None,
        help="Known nominal SSB center frequency in Hz (e.g., 3463.680e6)",
    )
    mode_group.add_argument(
        "--search-ssb",
        action="store_true",
        help="Blindly search the captured bandwidth for candidate SSB center frequencies",
    )
    parser.add_argument(
        "--ssb-scs",
        type=float,
        default=30e3,
        help="SSB subcarrier spacing in Hz (15e3 or 30e3, default: 30e3)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.2,
        help="Duration of recording to analyze in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time offset into recording in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--cfo-range",
        type=float,
        default=45e3,
        help="Coarse frequency offset search range +/- Hz around SSB frequency (default: 45e3)",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate all 5 diagnostic figures (IQ, Constellation, PSD, PSS, SSS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_plots",
        help="Directory to save generated PNG plots (default: output_plots)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots to --output-dir without opening an interactive GUI window",
    )
    return parser.parse_args(argv)


def run_detector(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()

    # 1. Read and validate SigMF metadata
    try:
        metadata = read_sigmf_metadata(args.input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load SigMF metadata: {exc}", file=sys.stderr)
        return 1

    start_sample = int(round(args.start_time * metadata.sample_rate))
    try:
        iq = load_iq_samples(
            metadata,
            start_sample=start_sample,
            duration_sec=args.duration,
            normalize=True,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to read IQ samples: {exc}", file=sys.stderr)
        return 1

    proc_fs, _, _, _ = get_ssb_ofdm_params(scs_hz=args.ssb_scs)

    # 2. Determine SSB frequencies to inspect
    candidate_ssb_freqs: List[float] = []
    if args.search_ssb:
        print(
            f"[INFO] Running Blind SSB Search across capture bandwidth "
            f"({metadata.sample_rate / 1e6:.2f} MS/s centered at {metadata.center_frequency / 1e6:.6f} MHz)..."
        )
        scored = blind_search_ssb_frequencies(
            iq=iq,
            capture_center_freq_hz=metadata.center_frequency,
            raw_fs=metadata.sample_rate,
            scs_hz=args.ssb_scs,
        )
        # Inspect top 3 candidate SSB frequencies from the coarse scan
        candidate_ssb_freqs = [item[0] for item in scored[:3]]
        ssb_mode_desc = (
            f"Blind SSB Search Mode (--search-ssb) -> Top candidate: "
            f"{candidate_ssb_freqs[0] / 1e6:.6f} MHz"
        )
    else:
        if args.ssb_frequency is not None:
            ssb_freq = float(args.ssb_frequency)
            ssb_mode_desc = f"Known SSB Frequency Mode ({ssb_freq / 1e6:.6f} MHz)"
        else:
            # Default to initial test case offset (+220.002 kHz above 3463.459998 MHz = 3463.680 MHz)
            # if capture matches 3463.46 MHz, otherwise capture center
            if abs(metadata.center_frequency - 3463.459998e6) < 1e6:
                ssb_freq = 3463.680e6
                ssb_mode_desc = f"Default Known SSB Frequency ({ssb_freq / 1e6:.6f} MHz)"
            else:
                ssb_freq = metadata.center_frequency
                ssb_mode_desc = f"Capture Center Default ({ssb_freq / 1e6:.6f} MHz)"

        # Verify that the requested SSB frequency falls within the captured Nyquist bandwidth
        half_bw = metadata.sample_rate / 2.0
        if abs(ssb_freq - metadata.center_frequency) > half_bw:
            print(
                f"[ERROR] Specified SSB frequency ({ssb_freq / 1e6:.6f} MHz) lies outside "
                f"the captured frequency range "
                f"[{(metadata.center_frequency - half_bw) / 1e6:.3f}, "
                f"{(metadata.center_frequency + half_bw) / 1e6:.3f}] MHz.",
                file=sys.stderr,
            )
            return 1
        candidate_ssb_freqs = [ssb_freq]

    # 3. Perform DDC, anti-aliased resampling, PSS detection, and SSS evaluation
    all_pss_candidates: List[PSSCandidate] = []
    all_sss_detections: List[SSSDetectionResult] = []
    primary_corr_traces = {}

    for idx, ssb_freq in enumerate(candidate_ssb_freqs):
        freq_offset_hz = ssb_freq - metadata.center_frequency
        rx_proc = ddc_and_resample(
            iq=iq,
            fs_in=metadata.sample_rate,
            freq_offset_hz=freq_offset_hz,
            fs_out=proc_fs,
        )

        # In blind search mode, expand CFO range to cover the raster step size
        cfo_range = max(args.cfo_range, 180e3) if args.search_ssb else args.cfo_range

        pss_cands, corr_traces = detect_pss_at_frequency(
            rx_proc=rx_proc,
            raw_fs=metadata.sample_rate,
            proc_fs=proc_fs,
            ssb_center_freq_hz=ssb_freq,
            scs_hz=args.ssb_scs,
            cfo_search_range_hz=cfo_range,
        )
        if idx == 0:
            primary_corr_traces = corr_traces

        all_pss_candidates.extend(pss_cands)

        for p_cand in pss_cands:
            sss_res = detect_sss_for_pss_candidate(rx_proc, p_cand)
            if sss_res is not None:
                all_sss_detections.append(sss_res)

    # 4. Group detections into distinct cells and format report
    grouped_cells = group_ssb_detections(all_sss_detections)
    elapsed_sec = time.perf_counter() - t0

    report_str = format_cell_report(
        metadata=metadata,
        analyzed_samples=len(iq),
        processing_time_sec=elapsed_sec,
        grouped_cells=grouped_cells,
        all_occasions=all_sss_detections,
        ssb_mode_desc=ssb_mode_desc,
    )
    print(report_str)

    # 5. Visualization (Figures 1 to 5)
    if args.plot_all:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        detected_freqs = (
            [c.ssb_center_freq_hz + c.mean_cfo_hz for c in grouped_cells]
            if grouped_cells
            else candidate_ssb_freqs
        )

        plot_iq_time_domain(
            iq, metadata, save_path=out_dir / "fig1_iq_time_domain.png"
        )
        plot_iq_constellation(
            iq, metadata, save_path=out_dir / "fig2_iq_constellation.png"
        )
        plot_psd(
            iq,
            metadata,
            ssb_frequencies_hz=detected_freqs,
            ssb_is_detected=bool(grouped_cells),
            save_path=out_dir / "fig3_psd.png",
        )
        plot_pss_results(
            primary_corr_traces,
            proc_fs=proc_fs,
            candidates=all_pss_candidates,
            save_path=out_dir / "fig4_pss_detection.png",
        )
        plot_sss_results(
            grouped_cells, save_path=out_dir / "fig5_sss_detection.png"
        )
        print(f"\n[INFO] Saved Figures 1-5 to directory: {out_dir.resolve()}")

        if not args.no_show:
            plt.show()
        else:
            plt.close("all")

    return 0


if __name__ == "__main__":
    sys.exit(run_detector(parse_args()))