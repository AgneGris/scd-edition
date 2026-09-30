# Changelog

All notable changes to SCD Edition are documented here.

## Unreleased

- Add an editable automatic preview for splitting merged motor units into
  high-amplitude A and low-amplitude B children, with persistent lineage and
  split-aware filter recalculation through sequential peel-off replay.
- Add reproducibility audit reports alongside new and edited decompositions.
- Add rotating application logs and copyable runtime diagnostics for support.
- Validate decomposition schemas and warn before loading pickle files.
- Preserve an explicitly selected CPU or CUDA backend during development checks.
- Replay SCD's whitening faithfully during filter recalculation: apply
  autocorrelation whitening when the decomposition used it, centre the EMG
  before the saved whitening matrix, and fit any missing whitening on the
  plateau with the recorded method.
- Draw combo-box arrows as chevrons instead of small squares, show metric names
  such as "MUAP stability" in full, and show the ampersand in "Review & Apply".
- Refresh the README demo and screenshots from a realistic synthetic
  contraction rendered with native glyphs, and let `scripts/capture_demo.py`
  also export square and portrait MP4 videos.

## 0.1.0 — 2026-09-19

First public release.

- Configure and decompose surface and intramuscular EMG recordings.
- Load OT Bioelettronica, Intan, MATLAB, HDF5, NumPy and CSV data.
- Inspect unfamiliar MATLAB/HDF5 layouts in the GUI and reuse the selected
  array path and orientation through saved configurations.
- Convert proprietary recordings to a documented portable SCD HDF5 format.
- Review, edit, quality-check and save motor-unit spike trains.
- Visualise rasters, discharge rates, cumulative spike trains and quality metrics.
- Open raw Swarm-Contrastive Decomposition output for downstream editing.
- Include a configured example recording through `scd-edition --example`.
- Distribute a self-contained Windows application with tagged releases.
