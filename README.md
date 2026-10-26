# 5G NR Physical Cell Identity (PCI) Detector (3GPP TS 38.211)

A modular Python signal-processing suite that reads 5G NR wideband IQ recordings in SigMF format, performs digital down-conversion (DDC) and anti-aliased rational resampling, detects Primary Synchronization Signals (PSS) and Secondary Synchronization Signals (SSS), computes Physical Cell Identities (PCI), groups multi-occasion bursts, and generates diagnostic visualizations.

## 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Running the Unit & Synthetic Verification Suite

Always run the unit and synthetic-signal verification tests before analyzing field recordings:

```bash
pytest -v
```

## 3. Usage Examples

**Mode 1: Known SSB Frequency Mode (Recommended)**
Search near the expected SSB center frequency (`3463.680 MHz`, which is `+220.002 kHz` above the `3463.459998 MHz` capture center) and generate all five diagnostic figures:

```bash
python3 main.py recording.sigmf-data \
    --ssb-frequency 3463.680e6 \
    --ssb-scs 30e3 \
    --duration 0.2 \
    --plot-all
```

**Mode 2: Blind SSB Search Mode**
Scan the captured bandwidth for candidate SSB synchronization rasters without prior knowledge of the SSB frequency:

```bash
python3 main.py recording.sigmf-data \
    --search-ssb \
    --duration 1.0 \
    --plot-all
```

## 4. Troubleshooting & Causes of Disagreement on Real Recordings

If the detector does not reproduce previously observed PCIs (such as **PCI 995** or **PCI 768**) on a real recording, investigate the following physical-layer causes in order:

*   **1. Incorrect SSB Frequency:** PSS correlation trace stays near the noise floor (<= 0.04) for all N_ID_2. Verify the GSCN raster frequency or run `--search-ssb`.
*   **2. Large Carrier Frequency Offset (CFO):** PSS correlation attenuates sharply when |CFO| > Delta f / 2 (+/- 15 kHz) due to inter-carrier interference across the OFDM symbol duration. Increase `--cfo-range` (e.g., `--cfo-range 90e3`).
*   **3. Timing Drift / Symbol Offset:** PSS is detected strongly, but all 336 SSS correlation scores remain flat (<= 0.05). Check if the recording contains dropped DMA buffers between l=0 (PSS) and l=2 (SSS) or an incorrect sample rate in `.sigmf-meta`.
*   **4. Multipath Phase Rotation Across Subcarriers:** Direct SSS inner product fails even with high PSS SNR because delay spread rotates the phase between subcarrier k=56 and k=182. Handled automatically in `nr_sss.py` via PSS-aided least-squares per-subcarrier channel phase equalization (`h_est = y_pss * d_pss`).
*   **5. Insufficient Recording Duration:** Zero detections on short slices (< 20 ms). Standard standalone 5G NR cells transmit SSB bursts once every 20 ms (up to 160 ms in some configurations). Use `--duration 0.1` or `0.2`.
*   **6. Incorrect IQ Endianness / Scaling:** Spectrum is flat white noise or exhibits a mirror image around DC. Verify `core:datatype` is `ci16_le` and that I and Q were not swapped during capture.

## 5. Scope & Limitations

*   **Synchronization vs. Broadcast Decoding:** Detecting PSS and SSS uniquely identifies N_ID_2 in {0,1,2}, N_ID_1 in {0..335}, and PCI = 3*N_ID_1 + N_ID_2. It **does not** decode the Physical Broadcast Channel (PBCH), Master Information Block (MIB), System Information Block 1 (SIB1), SSB Beam Index, System Frame Number (SFN), PLMN ID, Tracking Area Code (TAC), or 36-bit NR Cell Identity (NCI).
*   **Extensibility:** To add PBCH/MIB decoding in Stage 8, pass the CFO-corrected 4-symbol SSB complex grid from `nr_sss.py` into a PBCH DMRS blind detector (8 SSB-index hypotheses) followed by QPSK demapping, descrambling, and 3GPP Polar decoding (TS 38.212).