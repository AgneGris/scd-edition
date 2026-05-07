"""
batch_decompose.py
──────────────────
Headless (no GUI) batch decomposition using the same SCD pipeline as the app.

Usage
─────
    python scripts/batch_decompose.py \\
        --config  path/to/session.yaml \\
        --layout  path/to/data_layout.yaml \\
        --files   data/*.otb+ \\
        --output  results/ \\
        [--bad-channels  Grid_1:63  Grid_2:63] \\
        [--params  sil_threshold=0.85  iterations=200]

    # Concatenate two runs and decompose as one signal:
    python scripts/batch_decompose.py \\
        --channel-config channel_config.json \\
        --files subject_run_01.h5 subject_run_02.h5 \\
        --concat \\
        --output results/

Arguments
─────────
--config      Session YAML (defines ports, electrodes, channel assignments).
--layout      Data-layout YAML (defines how to read the EMG files).
--files       One or more EMG file paths (globs are expanded by the shell or
              you can pass a directory).
--output      Output directory for .pkl files (default: same dir as input).
--bad-channels  Per-grid bad channel indices in the form  GridName:idx[,idx,...]
              e.g.  --bad-channels Grid_1:63  Grid_2:63
              Indices are 0-based relative to that grid's channel list.
              Each specified channel is replaced with baseline noise before
              decomposition (same as the GUI's channel rejection).
--params      Override decomposition parameters, e.g.
              --params sil_threshold=0.90 iterations=300 peel_off=True
--concat      Concatenate all --files along the time axis and decompose them
              as a single signal instead of decomposing each file separately.
              When --rejections-file is provided, per-file time masks are
              automatically offset to the correct position in the concatenated
              signal; channel masks are OR-combined across files.
--concat-stem Custom output filename stem when --concat is used.
              Default: auto-generated from the input file names.

All decomposition defaults match the GUI's automatic-mode defaults.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_bad_channels(specs: List[str], grid_configs: dict) -> List[np.ndarray]:
    """Convert  ["Grid_1:63", "Grid_2:62,63"]  to per-grid rejection masks."""
    bad: Dict[str, List[int]] = {}
    for spec in (specs or []):
        name, _, idx_str = spec.partition(":")
        name = name.strip()
        indices = [int(i.strip()) for i in idx_str.split(",") if i.strip()]
        bad.setdefault(name, []).extend(indices)

    masks = []
    for port_name, config in grid_configs.items():
        n = len(config["channels"])
        mask = np.zeros(n, dtype=int)
        for idx in bad.get(port_name, []):
            if 0 <= idx < n:
                mask[idx] = 1
            else:
                print(f"  Warning: bad-channel index {idx} out of range for "
                      f"{port_name} ({n} channels) — ignored.")
        masks.append(mask)
    return masks


def _load_per_file_rejections(
    json_path: Path,
    file_path: Path,
    grid_configs: dict,
) -> tuple:
    """
    Load per-file rejection masks and time masks from a rejections JSON produced
    by batch_channel_check.py.  Falls back to all-zeros / empty if not found.

    Supports both the old format (grid value = list of ints) and the new format
    (grid value = {"channels": [...], "time_masks": [[start_s, end_s], ...]}).

    Returns
    -------
    channel_masks : List[np.ndarray]   one binary mask per grid
    time_masks    : List[List]         per grid: list of [start_s, end_s] pairs
    """
    with open(json_path) as f:
        data = json.load(f)

    fname  = file_path.name
    entry  = data.get(fname, {})
    masks: List[np.ndarray] = []
    all_time_masks: List[List] = []

    for port_name, config in grid_configs.items():
        n     = len(config["channels"])
        saved = entry.get(port_name)
        if saved is None:
            mask            = np.zeros(n, dtype=int)
            grid_time_masks = []
        elif isinstance(saved, list):
            # Old format: plain channel-mask array
            mask = np.array(saved, dtype=int)
            if len(mask) != n:
                print(f"  Warning: rejection mask length mismatch for "
                      f"{port_name} in {fname} — resetting.")
                mask = np.zeros(n, dtype=int)
            grid_time_masks = []
        else:
            # New format: {"channels": [...], "time_masks": [...]}
            ch   = saved.get("channels", [0] * n)
            mask = np.array(ch, dtype=int)
            if len(mask) != n:
                print(f"  Warning: rejection mask length mismatch for "
                      f"{port_name} in {fname} — resetting.")
                mask = np.zeros(n, dtype=int)
            grid_time_masks = saved.get("time_masks", [])

        masks.append(mask)
        all_time_masks.append(grid_time_masks)

    n_rej   = sum(int(m.sum()) for m in masks)
    n_tmask = sum(len(tm) for tm in all_time_masks)
    if n_rej:
        print(f"  Loaded rejections for {fname}: {n_rej} channel(s) rejected")
    if n_tmask:
        print(f"  Loaded time masks for {fname}: {n_tmask} region(s)")
    return masks, all_time_masks


def _setup_from_channel_config(json_path: Path, param_overrides: dict) -> tuple:
    """
    Parse the app's native channel_config.json into (grid_configs, layout, sampling_rate).
    This is an alternative to providing a session YAML + layout YAML.
    """
    with open(json_path) as f:
        data = json.load(f)

    sampling_rate = data.get("sampling_rate", 10240)

    grid_configs: Dict[str, dict] = {}
    for g in data.get("grids", []):
        channels   = list(range(g["start_chan"], g["end_chan"]))
        n          = len(channels)
        is_surface = g.get("type", "").lower() == "surface"
        ext_factor = int(np.ceil(1000 / n))
        defaults = {
            "extension_factor": ext_factor,
            "lowpass_hz":       500  if is_surface else 4400,
            "highpass_hz":      10,
            "notch_filter":     "None",
            "notch_harmonics":  False,
            "sil_threshold":    0.85,
            "iterations":       200,
            "clamp":            True,
            "fitness":          "CoV",
            "peel_off":         True,
            "peel_off_repeats": True,
            "swarm":            True,
            "muap_window_ms":   20,
        }
        defaults.update(param_overrides)
        grid_configs[g["name"]] = {
            "params":          defaults,
            "channels":        channels,
            "num_channels":    n,
            "electrode_type":  g.get("config", ""),
            "electrode_class": "surface_grid" if is_surface else "intramuscular",
        }

    from scd_app.io.data_loader import load_layout
    loader_map = {
        ".otb+": "loader_otb+.yaml",
        ".h5":   "loader_h5.yaml",
        ".mat":  "loader_mat.yaml",
    }
    loader_key  = data.get("loader", ".otb+")
    loader_file = loader_map.get(loader_key, "loader_otb+.yaml")
    _here       = Path(__file__).resolve().parent.parent
    loader_path = _here / "src" / "scd_app" / "resources" / "loaders_configs" / loader_file
    if not loader_path.exists():
        import importlib.resources as pkg_res
        loader_path = Path(str(
            pkg_res.files("scd_app") / "resources" / "loaders_configs" / loader_file))
    layout = load_layout(loader_path)

    return grid_configs, layout, sampling_rate


def _parse_param_overrides(pairs: List[str]) -> dict:
    """Convert ["sil_threshold=0.9", "peel_off=True"] to a dict."""
    overrides = {}
    for pair in (pairs or []):
        k, _, v = pair.partition("=")
        k, v = k.strip(), v.strip()
        # Coerce type
        if v.lower() == "true":
            overrides[k] = True
        elif v.lower() == "false":
            overrides[k] = False
        else:
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v
    return overrides


def _build_grid_configs(session_config, param_overrides: dict) -> dict:
    """Build grid_configs dict from a SessionConfig (same logic as the GUI)."""
    grid_configs: Dict[str, dict] = {}
    for port in session_config.ports:
        if not port.enabled:
            continue
        n_channels = len(port.electrode.channels)
        extension_factor = int(np.ceil(1000 / n_channels))
        is_surface = port.electrode.type in ("surface", "surface_grid")
        defaults = {
            "extension_factor": extension_factor,
            "lowpass_hz":       500  if is_surface else 4400,
            "highpass_hz":      10,
            "notch_filter":     "None",
            "notch_harmonics":  False,
            # Global decomp defaults (same as GUI)
            "sil_threshold":    0.85,
            "iterations":       200,
            "clamp":            True,
            "fitness":          "CoV",
            "peel_off":         True,
            "peel_off_repeats": True,
            "swarm":            True,
            "fixed_exponent":   2,
        }
        defaults.update(param_overrides)
        grid_configs[port.name] = {
            "params":         defaults,
            "channels":       port.electrode.channels,
            "num_channels":   n_channels,
            "electrode_type": port.electrode.name,
            "electrode_class": port.electrode.type,
        }
    return grid_configs


# ── core decomposition (no Qt dependency) ─────────────────────────────────────

def decompose_files(
    file_paths: List[Path],
    layout: dict,
    grid_configs: dict,
    bad_channel_masks: List[np.ndarray],
    sampling_rate: int,
    output_dir: Path,
    plateau_s: Optional[tuple] = None,        # (start_s, end_s) in seconds, or None = full file
    time_masks_per_grid: Optional[List] = None,  # per-grid list of [start_s, end_s] pairs
    verbose: bool = True,
):
    """
    Run SCD decomposition on every file in file_paths and save a .pkl per file.

    This is the Qt-free equivalent of DecompositionWorker.run() + _save_results().
    The source_callback is omitted (saves ~15 % wall time per MU on large sessions).
    """
    from scd_app.core.decomp_worker import DecompositionWorker

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_idx, file_path in enumerate(file_paths):
        t0 = time.perf_counter()
        print(f"\n[{file_idx + 1}/{len(file_paths)}] {file_path.name}")

        # ── load EMG ──────────────────────────────────────────────────────────
        from scd_app.io.data_loader import load_field
        try:
            emg = load_field(file_path, layout, "emg")   # (samples, channels)
        except Exception as exc:
            print(f"  ERROR loading file: {exc} — skipping.")
            continue

        if plateau_s is not None:
            start_smp = int(round(plateau_s[0] * sampling_rate))
            end_smp   = int(round(plateau_s[1] * sampling_rate))
            start_smp = max(0, min(start_smp, emg.shape[0]))
            end_smp   = max(start_smp + 1, min(end_smp, emg.shape[0]))
            plateau_coords = np.array([start_smp, end_smp])
            print(f"  Plateau: {plateau_s[0]}s – {plateau_s[1]}s  "
                  f"(samples {start_smp}–{end_smp})")
        else:
            plateau_coords = np.array([0, emg.shape[0]])   # full file

        # ── build a worker just to reuse its helpers ───────────────────────
        # (no QThread.start() is called — we call run() directly)
        stem = file_path.stem
        save_path = output_dir / f"{stem}_decomp_output.pkl"

        worker = _HeadlessWorker(
            emg_data            = emg,
            grid_configs        = grid_configs,
            rejected_channels   = bad_channel_masks,
            plateau_coords      = plateau_coords,
            sampling_rate       = sampling_rate,
            save_path           = save_path,
            time_masks_per_grid = time_masks_per_grid or [],
        )
        worker.run()

        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s → {save_path}")


def decompose_concatenated(
    file_paths: List[Path],
    layout: dict,
    grid_configs: dict,
    bad_channel_masks_per_file: List[List[np.ndarray]],
    sampling_rate: int,
    output_dir: Path,
    plateau_s: Optional[tuple] = None,
    time_masks_per_grid_per_file: Optional[List[List[List]]] = None,
    output_stem: Optional[str] = None,
):
    """
    Load all files, concatenate along the time axis, then decompose as one signal.

    bad_channel_masks_per_file : list (one entry per file) of per-grid masks.
        Channel masks are OR-combined across files so any channel rejected in
        any file is rejected for the whole concatenated signal.
    time_masks_per_grid_per_file : list (one entry per file) of per-grid
        time-mask lists.  Each mask's timestamps are automatically offset by
        the cumulative file length so they remain valid in the concatenated signal.
    output_stem : override the auto-generated output filename stem.
    """
    from scd_app.io.data_loader import load_field

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConcatenating {len(file_paths)} file(s):")
    segments = []
    sample_offsets = [0]          # start sample index of each file in the concat
    for fp in file_paths:
        print(f"  Loading {fp.name} …")
        t_load = time.perf_counter()
        try:
            emg = load_field(fp, layout, "emg")
        except Exception as exc:
            print(f"  ERROR loading {fp.name}: {exc} — aborting concat.")
            return
        t_load = time.perf_counter() - t_load
        dur_s = emg.shape[0] / sampling_rate
        print(f"    shape    : {tuple(emg.shape)}  ({emg.shape[1]} channels, {dur_s:.2f} s)")
        print(f"    dtype    : {emg.dtype}")
        print(f"    loaded in: {t_load:.1f}s")
        segments.append(emg)
        sample_offsets.append(sample_offsets[-1] + emg.shape[0])

    # load_field returns torch tensors; use torch.cat to preserve type
    if torch.is_tensor(segments[0]):
        emg_concat = torch.cat(segments, dim=0)
    else:
        emg_concat = np.concatenate(segments, axis=0)
    total_s = emg_concat.shape[0] / sampling_rate
    print(f"  Concatenated shape: {emg_concat.shape}  ({total_s:.2f} s)")

    # ── combine channel masks (OR across files) ────────────────────────────────
    n_grids = len(grid_configs)
    combined_masks: List[np.ndarray] = []
    for grid_idx in range(n_grids):
        masks_for_grid = [
            per_file[grid_idx]
            for per_file in bad_channel_masks_per_file
            if grid_idx < len(per_file)
        ]
        if masks_for_grid:
            combined = np.zeros_like(masks_for_grid[0])
            for m in masks_for_grid:
                combined = np.logical_or(combined, m).astype(int)
        else:
            n = list(grid_configs.values())[grid_idx]["num_channels"]
            combined = np.zeros(n, dtype=int)
        combined_masks.append(combined)

    # ── offset time masks ──────────────────────────────────────────────────────
    merged_time_masks: List[List] = [[] for _ in range(n_grids)]
    if time_masks_per_grid_per_file:
        for file_i, file_masks in enumerate(time_masks_per_grid_per_file):
            offset_s = sample_offsets[file_i] / sampling_rate
            for grid_i, grid_masks in enumerate(file_masks):
                for (t0, t1) in grid_masks:
                    merged_time_masks[grid_i].append(
                        [t0 + offset_s, t1 + offset_s])

    # ── output path ───────────────────────────────────────────────────────────
    if output_stem is None:
        stems = [fp.stem for fp in file_paths]
        # Find common prefix, then append only the unique suffixes of each stem
        # e.g. ["base_run-01", "base_run-02"] → "base_run-01_run-02_concat"
        from os.path import commonprefix
        prefix = commonprefix(stems).rstrip("_")
        suffixes = [s[len(prefix):].lstrip("_") for s in stems]
        suffixes = [s for s in suffixes if s]  # drop empty
        if suffixes:
            output_stem = prefix + "_" + "_".join(suffixes) + "_concat"
        else:
            output_stem = prefix + "_concat"
    save_path = output_dir / f"{output_stem}_decomp_output.pkl"

    # ── plateau ───────────────────────────────────────────────────────────────
    if plateau_s is not None:
        start_smp = int(round(plateau_s[0] * sampling_rate))
        end_smp   = int(round(plateau_s[1] * sampling_rate))
        start_smp = max(0, min(start_smp, emg_concat.shape[0]))
        end_smp   = max(start_smp + 1, min(end_smp, emg_concat.shape[0]))
        plateau_coords = np.array([start_smp, end_smp])
        print(f"  Plateau : {plateau_s[0]}s – {plateau_s[1]}s  "
              f"(samples {start_smp}–{end_smp})")
    else:
        plateau_coords = np.array([0, emg_concat.shape[0]])

    t0 = time.perf_counter()
    worker = _HeadlessWorker(
        emg_data            = emg_concat,
        grid_configs        = grid_configs,
        rejected_channels   = combined_masks,
        plateau_coords      = plateau_coords,
        sampling_rate       = sampling_rate,
        save_path           = save_path,
        time_masks_per_grid = merged_time_masks,
    )
    worker.run()
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s → {save_path}")


class _HeadlessWorker:
    """
    Thin wrapper around DecompositionWorker's logic with Qt signals replaced by
    plain print() calls.  We subclass nothing — we just copy the interface that
    DecompositionWorker.run() expects from `self`.
    """

    def __init__(self, emg_data, grid_configs, rejected_channels,
                 plateau_coords, sampling_rate, save_path,
                 time_masks_per_grid=None):
        self.emg_data            = emg_data
        self.grid_configs        = grid_configs
        self.rejected_channels   = rejected_channels
        self.plateau_coords      = plateau_coords
        self.sampling_rate       = sampling_rate
        self.save_path           = save_path
        self.time_masks_per_grid = time_masks_per_grid or []
        self.aux_configs         = []
        self.emg_file_path       = None
        self._is_running         = True

    # Mimic Qt signals as no-ops so we can share the run() implementation
    class _Signal:
        def emit(self, *args): pass

    progress             = _Signal()
    electrode_completed  = _Signal()
    finished             = _Signal()
    error                = _Signal()
    source_found         = _Signal()

    def run(self):
        """Identical to DecompositionWorker.run() but prints progress directly."""
        from scd_app.core.decomp_worker import DecompositionWorker

        # Bind all helpers that call self.* internally
        self._parse_notch          = DecompositionWorker._parse_notch.__get__(self)
        self._create_notch_params  = DecompositionWorker._create_notch_params.__get__(self)
        _create_scd_config         = DecompositionWorker._create_scd_config.__get__(self)
        _save_results_fn           = DecompositionWorker._save_results.__get__(self)

        try:
            results = {
                "pulse_trains":         [],
                "discharge_times":      [],
                "mu_filters":           [],
                "ports":                [],
                "w_mat":                [],
                "peel_off_sequence":    [],
                "preprocessing_config": [],
            }
            total_mus = 0

            for grid_idx, (port_name, config) in enumerate(self.grid_configs.items()):
                if not self._is_running:
                    break

                print(f"  Processing {port_name} "
                      f"({grid_idx + 1}/{len(self.grid_configs)})...")

                channels = config["channels"]
                n_total  = self.emg_data.shape[1]
                bad_ch_idx = [c for c in channels if c >= n_total]
                if bad_ch_idx:
                    raise IndexError(
                        f"{port_name}: channel indices {bad_ch_idx} out of range "
                        f"for EMG with {n_total} channels."
                    )

                grid_data = self.emg_data[:, channels]

                # Bad channel replacement
                rejected     = self.rejected_channels[grid_idx]
                if len(rejected) != len(channels):
                    print(f"  Warning: rejection mask length mismatch for "
                          f"{port_name} — resetting.")
                    rejected = np.zeros(len(channels), dtype=int)

                good_channels = np.where(rejected == 0)[0]
                noise_std = (
                    grid_data[:, good_channels].std().item()
                    if len(good_channels) > 0 else 1e-6
                )

                bad_channels = np.where(rejected == 1)[0]
                if len(bad_channels) > 0:
                    print(f"    Masked channels : {list(bad_channels)}")
                    gen = torch.Generator()
                    gen.manual_seed(42)
                    noise = torch.randn(
                        grid_data.shape[0], len(bad_channels), generator=gen
                    ) * noise_std
                    grid_data[:, bad_channels] = noise
                else:
                    print(f"    Masked channels : none")

                # Apply time masks: replace all channels in masked segments with noise
                grid_time_masks = (
                    self.time_masks_per_grid[grid_idx]
                    if grid_idx < len(self.time_masks_per_grid) else []
                )
                for (t_start_s, t_end_s) in grid_time_masks:
                    t_start = int(round(t_start_s * self.sampling_rate))
                    t_end   = int(round(t_end_s   * self.sampling_rate))
                    t_start = max(0, min(t_start, grid_data.shape[0]))
                    t_end   = max(t_start + 1, min(t_end, grid_data.shape[0]))
                    n_samp  = t_end - t_start
                    gen_t   = torch.Generator()
                    gen_t.manual_seed(43 + t_start)
                    noise_t = torch.randn(
                        n_samp, grid_data.shape[1], generator=gen_t
                    ) * noise_std
                    grid_data[t_start:t_end, :] = noise_t
                    print(f"    Time masked : {t_start_s:.3f}s – {t_end_s:.3f}s  "
                          f"({n_samp} samples)")

                start_sample = int(self.plateau_coords[0])
                end_sample   = int(self.plateau_coords[1])
                grid_data    = grid_data[start_sample:end_sample, :]

                scd_config   = _create_scd_config(config["params"])

                # Run SCD — no source_callback so no real-time plot updates
                grid_data_dev = grid_data.to(
                    device=scd_config.device, dtype=torch.float32)
                from scd.models.scd import SwarmContrastiveDecomposition
                model = SwarmContrastiveDecomposition()
                timestamps, dictionary = model.run(
                    grid_data_dev, scd_config, source_callback=None)

                if dictionary and "filters" in dictionary:
                    cpu_timestamps = [
                        t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
                        for t in timestamps
                    ] if isinstance(timestamps, list) else timestamps

                    results["pulse_trains"].append(dictionary["source"])
                    results["discharge_times"].append(cpu_timestamps)
                    results["mu_filters"].append(dictionary["filters"])
                    results["ports"].append(port_name)
                    results["w_mat"].append(dictionary.get("w_mat"))
                    results["peel_off_sequence"].append(
                        dictionary.get("peel_off_sequence", []))

                    prep_cfg = dict(dictionary.get("preprocessing_config", {}))
                    prep_cfg.setdefault("square_sources_spike_det",
                                        bool(scd_config.square_sources_spike_det))
                    results["preprocessing_config"].append(prep_cfg)

                    n_mus = len(timestamps) if isinstance(timestamps, list) else 1
                    total_mus += n_mus
                    print(f"    {port_name}: {n_mus} MU(s) found")
                else:
                    results["pulse_trains"].append(np.array([]))
                    results["discharge_times"].append([])
                    results["mu_filters"].append(np.array([]))
                    results["ports"].append(port_name)
                    results["w_mat"].append(None)
                    results["peel_off_sequence"].append([])
                    results["preprocessing_config"].append({})
                    print(f"    {port_name}: 0 MUs found")

            print(f"  Saving → {self.save_path}  (total {total_mus} MU(s))")
            _save_results_fn(results)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ERROR during decomposition: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Headless batch SCD decomposition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Option A: app's native channel_config.json (simplest)
    ap.add_argument("--channel-config", default=None, metavar="JSON",
                    help="Path to the app's channel_config.json. "
                         "When provided, --config and --layout are not needed.")
    # Option B: session YAML + layout YAML
    ap.add_argument("--config",  default=None,  help="Session YAML path.")
    ap.add_argument("--layout",  default=None,  help="Data-layout YAML path.")
    ap.add_argument("--files",   required=True,  nargs="+",
                    help="EMG file path(s), directory, or glob pattern. "
                         "If a directory is given, all files matching --ext are processed.")
    ap.add_argument("--output",  default=None,
                    help="Output directory (default: same directory as each input file).")
    ap.add_argument("--bad-channels", nargs="*", default=[],
                    metavar="GRID:IDX",
                    help="Channels to reject, e.g.  Grid_1:63  Grid_2:62,63  "
                         "(applied to every file; ignored if --rejections-file is given).")
    ap.add_argument("--rejections-file", default=None, metavar="JSON",
                    help="Per-file rejection JSON produced by batch_channel_check.py. "
                         "Takes precedence over --bad-channels.")
    ap.add_argument("--plateau", nargs=2, type=float, metavar=("START_S", "END_S"),
                    default=None,
                    help="Time window in seconds, e.g.  --plateau 10.0 40.0  "
                         "Default: full file.")
    ap.add_argument("--ext",     default=None,
                    help="File extension to search when --files is a directory "
                         "(e.g. .h5, .mat, .otb+). Default: auto-detect from layout format.")
    ap.add_argument("--params",  nargs="*", default=[],
                    metavar="KEY=VALUE",
                    help="Override decomp params, e.g.  sil_threshold=0.90")
    ap.add_argument("--concat", action="store_true", default=False,
                    help="Concatenate all --files along the time axis and "
                         "decompose as a single signal instead of processing "
                         "each file separately.")
    ap.add_argument("--concat-stem", default=None, metavar="STEM",
                    help="Output filename stem when --concat is used "
                         "(default: auto-generated from input file names).")
    args = ap.parse_args()

    # ── load config ───────────────────────────────────────────────────────────
    param_overrides = _parse_param_overrides(args.params)

    if args.channel_config:
        grid_configs, layout, fs = _setup_from_channel_config(
            Path(args.channel_config), param_overrides)
        print(f"Channel config : {args.channel_config}")
        print(f"Fs             : {fs} Hz")
        print(f"Grids          : {list(grid_configs.keys())}")
    elif args.config and args.layout:
        from scd_app.core.config import ConfigManager
        from scd_app.io.data_loader import load_layout
        mgr    = ConfigManager()
        config = mgr.load_session(Path(args.config))
        fs     = config.sampling_frequency
        layout = load_layout(Path(args.layout))
        print(f"Session   : {config.name}  |  Fs: {fs} Hz")
        grid_configs = _build_grid_configs(config, param_overrides)
    else:
        ap.error("Provide either --channel-config OR both --config and --layout.")

    bad_masks = _parse_bad_channels(args.bad_channels, grid_configs)

    if param_overrides:
        print(f"Param overrides: {param_overrides}")
    if any(m.any() for m in bad_masks):
        for (pname, _), mask in zip(grid_configs.items(), bad_masks):
            if mask.any():
                print(f"Bad channels  : {pname} → {list(np.where(mask)[0])}")

    # ── file list ─────────────────────────────────────────────────────────────
    # Auto-detect extension from layout format if not specified
    _fmt_ext = {
        "h5": ".h5", "mat": ".mat", "npy": ".npy", "otb": ".otb+",
    }
    default_ext = _fmt_ext.get(layout.get("format", ""), ".h5")
    search_ext  = args.ext if args.ext else default_ext

    file_paths = []
    for pat in args.files:
        p = Path(pat)
        if p.is_dir():
            found = sorted(p.glob(f"*{search_ext}"))
            if not found:
                print(f"  Warning: no *{search_ext} files found in {p}")
            file_paths.extend(found)
        elif "*" in pat or "?" in pat:
            file_paths.extend(sorted(Path(".").glob(pat)))
        else:
            file_paths.append(p)
    file_paths = sorted(set(file_paths))

    if not file_paths:
        print("No files matched — exiting.")
        sys.exit(1)
    print(f"\nFiles to process ({len(file_paths)}):")
    for p in file_paths:
        print(f"  {p}")

    # ── output dir ────────────────────────────────────────────────────────────
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = None   # resolved per-file below

    # ── run ───────────────────────────────────────────────────────────────────
    plateau_s = tuple(args.plateau) if args.plateau else None
    if plateau_s:
        print(f"Plateau   : {plateau_s[0]}s – {plateau_s[1]}s")

    rejections_path = Path(args.rejections_file) if args.rejections_file else None
    if rejections_path:
        print(f"Rejections: loaded per-file from {rejections_path}")

    t_total = time.perf_counter()

    if args.concat:
        # ── concatenate all files and decompose as one ─────────────────────
        masks_per_file: List[List[np.ndarray]] = []
        time_masks_per_file: List[List[List]] = []
        for file_path in file_paths:
            if rejections_path:
                m, tm = _load_per_file_rejections(
                    rejections_path, file_path, grid_configs)
            else:
                m  = bad_masks
                tm = [[] for _ in grid_configs]
            masks_per_file.append(m)
            time_masks_per_file.append(tm)

        out = output_dir if output_dir else file_paths[0].parent
        decompose_concatenated(
            file_paths                  = file_paths,
            layout                      = layout,
            grid_configs                = grid_configs,
            bad_channel_masks_per_file  = masks_per_file,
            sampling_rate               = fs,
            output_dir                  = out,
            plateau_s                   = plateau_s,
            time_masks_per_grid_per_file= time_masks_per_file,
            output_stem                 = args.concat_stem,
        )
    else:
        # ── decompose each file separately (original behaviour) ────────────
        for file_path in file_paths:
            if rejections_path:
                masks, file_time_masks = _load_per_file_rejections(
                    rejections_path, file_path, grid_configs)
            else:
                masks           = bad_masks
                file_time_masks = [[] for _ in grid_configs]
            out = output_dir if output_dir else file_path.parent
            decompose_files(
                file_paths          = [file_path],
                layout              = layout,
                grid_configs        = grid_configs,
                bad_channel_masks   = masks,
                sampling_rate       = fs,
                plateau_s           = plateau_s,
                time_masks_per_grid = file_time_masks,
                output_dir          = out,
            )

    elapsed = time.perf_counter() - t_total
    print(f"\nAll done in {elapsed:.1f}s.")


if __name__ == "__main__":
    # Make sure src/ is on the path when run directly
    _src = Path(__file__).resolve().parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    main()
