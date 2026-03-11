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

All decomposition defaults match the GUI's automatic-mode defaults.
"""

import argparse
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
    plateau_s: Optional[tuple] = None,   # (start_s, end_s) in seconds, or None = full file
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
            emg_data         = emg,
            grid_configs     = grid_configs,
            rejected_channels= bad_channel_masks,
            plateau_coords   = plateau_coords,
            sampling_rate    = sampling_rate,
            save_path        = save_path,
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
                 plateau_coords, sampling_rate, save_path):
        self.emg_data          = emg_data
        self.grid_configs      = grid_configs
        self.rejected_channels = rejected_channels
        self.plateau_coords    = plateau_coords
        self.sampling_rate     = sampling_rate
        self.save_path         = save_path
        self._is_running       = True

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

                bad_channels = np.where(rejected == 1)[0]
                if len(bad_channels) > 0:
                    good_channels = np.where(rejected == 0)[0]
                    noise_std = (
                        grid_data[:, good_channels].std().item()
                        if len(good_channels) > 0 else 1e-6
                    )
                    gen = torch.Generator()
                    gen.manual_seed(42)
                    noise = torch.randn(
                        grid_data.shape[0], len(bad_channels), generator=gen
                    ) * noise_std
                    grid_data[:, bad_channels] = noise
                    print(f"    Replaced {len(bad_channels)} bad channel(s) "
                          f"with noise (std={noise_std:.4f})")

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
    ap.add_argument("--config",  required=True,  help="Session YAML path.")
    ap.add_argument("--layout",  required=True,  help="Data-layout YAML path.")
    ap.add_argument("--files",   required=True,  nargs="+",
                    help="EMG file path(s), directory, or glob pattern. "
                         "If a directory is given, all files matching --ext are processed.")
    ap.add_argument("--output",  default=None,
                    help="Output directory (default: same directory as each input file).")
    ap.add_argument("--bad-channels", nargs="*", default=[],
                    metavar="GRID:IDX",
                    help="Channels to reject, e.g.  Grid_1:63  Grid_2:62,63")
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
    args = ap.parse_args()

    # ── load session config ───────────────────────────────────────────────────
    from scd_app.core.config import ConfigManager
    mgr    = ConfigManager()
    config = mgr.load_session(Path(args.config))
    fs     = config.sampling_frequency
    print(f"Session   : {config.name}")
    print(f"Fs        : {fs} Hz")
    print(f"Ports     : {[p.name for p in config.ports if p.enabled]}")

    # ── layout ────────────────────────────────────────────────────────────────
    from scd_app.io.data_loader import load_layout
    layout = load_layout(Path(args.layout))
    print(f"Layout    : {layout['name']}")

    # ── build configs ─────────────────────────────────────────────────────────
    param_overrides = _parse_param_overrides(args.params)
    grid_configs    = _build_grid_configs(config, param_overrides)
    bad_masks       = _parse_bad_channels(args.bad_channels, grid_configs)

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
        "mat_dynamic": ".mat",
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

    t_total = time.perf_counter()
    for file_idx, file_path in enumerate(file_paths):
        out = output_dir if output_dir else file_path.parent
        decompose_files(
            file_paths        = [file_path],
            layout            = layout,
            grid_configs      = grid_configs,
            bad_channel_masks = bad_masks,
            sampling_rate     = fs,
            plateau_s         = plateau_s,
            output_dir        = out,
        )
    elapsed = time.perf_counter() - t_total
    print(f"\nAll done in {elapsed:.1f}s.")


if __name__ == "__main__":
    # Make sure src/ is on the path when run directly
    _src = Path(__file__).resolve().parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    main()
