"""
add_force_to_decomp.py
──────────────────────
Backfill force / aux-channel data into decomposition .pkl files that were
saved before force saving was implemented.

Single-file mode
────────────────
    python scripts/add_force_to_decomp.py \\
        --decomp     path/to/decomp_output.pkl \\
        --data       path/to/recording.otb+ \\
        --config     path/to/channel_config.json \\
        [--layout    path/to/data_layout.yaml] \\
        [--output    path/to/output.pkl]

    For concat decompositions pass all source files in order:
        --data  run-01.otb+  run-02.otb+  run-03.otb+

Batch mode  (processes a whole directory of pkl files)
──────────────────────────────────────────────────────
    python scripts/add_force_to_decomp.py \\
        --decomp-dir  path/to/decomp_dir/ \\
        --data-dir    path/to/raw_data_dir/ \\
        --config      path/to/channel_config.json \\
        --output-dir  path/to/output_dir/ \\
        [--layout     path/to/data_layout.yaml]

In batch mode the script derives raw-file names from the pkl name:
  • Single-run:  sub-01_…_run-01_decomp_output.pkl → sub-01_…_run-01.otb+
  • Concat:      sub-01_…_run-0_1_2_concat_decomp_output.pkl
                   → sub-01_…_run-01.otb+, run-02.otb+  (auto-detected)
  • _edited.pkl  files are always skipped
  • Existing output files are never overwritten

Arguments
─────────
--decomp       Single decomposition .pkl (single-file mode).
--data         One or more original recording files (single-file mode).
               Pass multiple paths for concat decompositions, in order.
--decomp-dir   Directory of decomposition .pkl files (batch mode).
--data-dir     Directory of original recording files (batch mode).
--config       Channel-config JSON (required, both modes).
--output-dir   Output directory (batch mode).
--output       Output path (single-file mode; default: overwrite --decomp).
--layout       Data-layout YAML — required for .mat/.h5 signal sources.
               Not needed for OTB+ files.

Source types
────────────
"signal"   Force is a column range [start_chan, end_chan) in the main EMG
           data array.  All adapter channels (including aux inputs at high
           indices) are read so they are always accessible.

"aux_file" Force is in auxiliary .sip streams inside an OTB+ archive.
           start_chan / end_chan index into those streams.
"""

import argparse
import copy
import json
import pickle
import sys
import tarfile
from pathlib import Path
from typing import List, Optional

import numpy as np


# ── OTB loaders ───────────────────────────────────────────────────────────────

def _read_otb_emg_all(data_path: Path) -> np.ndarray:
    """Read every adapter channel from an OTB+ file. Returns (samples, channels).

    Mirrors data_loader._read_otb_emg but without any channel filter, so
    auxiliary-input channels at high indices (force transducers, etc.) are
    included at their natural column positions.
    """
    import xml.etree.ElementTree as ET

    with tarfile.open(str(data_path), "r") as tar:
        members  = {m.name: m for m in tar.getmembers()}
        sig_name = next((n for n in members if n.endswith(".sig")), None)
        if sig_name is None:
            raise FileNotFoundError(f"No .sig file in {data_path.name}")

        xml_name  = sig_name.rsplit(".", 1)[0] + ".xml"
        xml_bytes = tar.extractfile(members[xml_name]).read()
        xml_root  = ET.fromstring(xml_bytes)
        device    = xml_root.attrib

        nADbit       = int(device["ad_bits"])
        nchans       = int(device["DeviceTotalChannels"])
        power_supply = 5  # V

        sig_bytes = tar.extractfile(members[sig_name]).read()
        raw = (
            np.frombuffer(sig_bytes, dtype=f"int{nADbit}")
            .reshape(-1, nchans)
            .astype(np.float64)
        )

        adapters = xml_root.findall(".//Adapter")
        active   = []
        for i in range(len(adapters) - 1):
            s_ch = int(adapters[i].attrib["ChannelStartIndex"])
            e_ch = int(adapters[i + 1].attrib["ChannelStartIndex"])
            n_ch = e_ch - s_ch
            if n_ch > 0:
                active.append((i, s_ch, n_ch))

        if not active:
            raise ValueError(f"No active adapters in {data_path.name}")

        total = sum(n for _, _, n in active)
        emg   = np.zeros((raw.shape[0], total))
        col   = 0
        for idx, raw_start, n_ch in active:
            gain  = float(adapters[idx].attrib["Gain"])
            scale = (power_supply * 1000) / (2 ** nADbit * gain)
            emg[:, col : col + n_ch] = raw[:, raw_start : raw_start + n_ch] * scale
            col  += n_ch

    return emg  # (samples, channels)


def _read_otb_sip(data_path: Path) -> np.ndarray:
    """Return .sip auxiliary channels from an OTB+ file as (n_sip, samples)."""
    with tarfile.open(str(data_path), "r") as tar:
        members   = {m.name: m for m in tar.getmembers()}
        sip_names = sorted(n for n in members if n.endswith(".sip"))
        if not sip_names:
            raise FileNotFoundError(f"No .sip channels in {data_path.name}")
        arrays = [
            np.frombuffer(tar.extractfile(members[n]).read(), dtype="float64")
            for n in sip_names
        ]
    min_len = min(len(a) for a in arrays)
    return np.column_stack([a[:min_len] for a in arrays]).T  # (n_sip, samples)


# ── channel extractors ────────────────────────────────────────────────────────

def _load_signal_channels(
    data_paths: List[Path],
    aux_cfg: dict,
    layout_path: Optional[Path],
) -> np.ndarray:
    """Load [start_chan, end_chan) signal channels from one or more data files.

    Multiple files are concatenated along the time axis.
    Returns (channels, samples) or (samples,) for single-channel aux.
    """
    s = int(aux_cfg.get("start_chan", 0))
    e = int(aux_cfg.get("end_chan", s + 1))

    segments = []
    for dp in data_paths:
        suffix = dp.suffix.lower()
        if suffix in (".otb", ".otb+"):
            raw = _read_otb_emg_all(dp).T  # (channels, samples)
        else:
            if layout_path is None:
                raise ValueError(
                    f"--layout is required for non-OTB files ({dp.suffix})"
                )
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from scd_app.io.data_loader import load_field, load_layout
            layout = copy.deepcopy(load_layout(layout_path))
            if "fields" in layout and "emg" in layout["fields"]:
                layout["fields"]["emg"].pop("channels", None)
            raw = load_field(dp, layout, "emg").numpy().T  # (channels, samples)

        if e > raw.shape[0]:
            raise IndexError(
                f"channel range [{s},{e}) out of range "
                f"({raw.shape[0]} adapter channels in {dp.name})"
            )
        segments.append(raw[s:e, :])  # (n_ch, samples_i)

    combined = np.concatenate(segments, axis=-1)  # (n_ch, total_samples)
    return combined.squeeze()


def _load_aux_file_channels(
    data_paths: List[Path],
    aux_cfg: dict,
) -> np.ndarray:
    """Load [start_chan, end_chan) .sip channels, concatenated across files."""
    s = int(aux_cfg.get("start_chan", 0))
    e = int(aux_cfg.get("end_chan", s + 1))

    segments = []
    for dp in data_paths:
        if dp.suffix.lower() not in (".otb", ".otb+"):
            raise ValueError(
                f"aux_file source only supported for OTB+ (got {dp.suffix!r})"
            )
        sip = _read_otb_sip(dp)  # (n_sip, samples)
        if e > sip.shape[0]:
            raise IndexError(
                f"sip range [{s},{e}) out of range ({sip.shape[0]} .sip channels "
                f"in {dp.name})"
            )
        segments.append(sip[s:e, :])

    combined = np.concatenate(segments, axis=-1)
    return combined.squeeze()


# ── concat-stem parser ────────────────────────────────────────────────────────

def _find_source_files_for_concat(
    concat_stem: str,
    data_dir: Path,
    extensions: tuple = (".otb+", ".otb", ".h5", ".hdf5", ".mat"),
) -> Optional[List[Path]]:
    """Reverse-engineer the source file list from a concat pkl stem.

    batch_decompose.py builds the concat stem as:
        commonprefix(stems).rstrip("_") + "_" + "_".join(suffixes) + "_concat"
    where suffixes = [stem[len(prefix):].lstrip("_") for each stem].

    We recover the source stems by:
    1. Stripping "_concat"
    2. Scanning the tail for a sequence of pure-digit tokens (the suffixes)
    3. Reconstructing each source stem as prefix + suffix
    """
    if not concat_stem.endswith("_concat"):
        return None
    no_concat = concat_stem[: -len("_concat")]

    parts = no_concat.split("_")

    # Walk right-to-left collecting pure-digit tokens
    suffix_start = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            suffix_start = i
        else:
            break

    if suffix_start >= len(parts):
        return None  # no trailing digit tokens → can't decode

    prefix   = "_".join(parts[:suffix_start])   # e.g. "sub-01_..._run-0"
    suffixes = parts[suffix_start:]              # e.g. ["1", "2"]

    if len(suffixes) < 2:
        return None  # need at least two source files

    source_files: List[Path] = []
    for suf in suffixes:
        stem = prefix + suf                      # e.g. "sub-01_..._run-01"
        found = None
        for ext in extensions:
            candidate = data_dir / (stem + ext)
            if candidate.exists():
                found = candidate
                break
        if found is None:
            return None   # at least one source file missing → abort
        source_files.append(found)

    return source_files


# ── core processing ───────────────────────────────────────────────────────────

def build_aux_channels(
    data_paths: List[Path],
    aux_cfgs: list,
    layout_path: Optional[Path],
) -> list:
    """Load force data for every aux-channel entry. Returns the save-ready list."""
    result = []
    for a in aux_cfgs:
        name   = a.get("name", "?")
        source = a.get("source", "signal")
        s      = int(a.get("start_chan", 0))
        e      = int(a.get("end_chan", s + 1))
        try:
            sig = (
                _load_aux_file_channels(data_paths, a)
                if source == "aux_file"
                else _load_signal_channels(data_paths, a, layout_path)
            )
        except Exception as ex:
            print(f"    [skip] '{name}': {ex}")
            continue

        arr = np.asarray(sig, dtype=np.float64)
        result.append({
            "data":       arr,
            "meta":       {k: v for k, v in a.items() if k not in ("start_chan", "end_chan")},
            "start_chan": s,
            "end_chan":   e,
        })
        print(f"    [ok]   '{name}'  source={source}  ch=[{s},{e})  shape={arr.shape}")
    return result


def process_one(
    decomp_path: Path,
    data_paths: List[Path],
    config_path: Path,
    layout_path: Optional[Path],
    out_path: Path,
) -> bool:
    """Inject aux channels into one pkl. Returns True on success."""
    with open(decomp_path, "rb") as f:
        decomp = pickle.load(f)

    if decomp.get("aux_channels"):
        print(f"  Warning: already has {len(decomp['aux_channels'])} aux channel(s) — replacing.")

    with open(config_path) as f:
        chan_cfg = json.load(f)
    aux_cfgs = chan_cfg.get("aux_channels", [])
    if not aux_cfgs:
        print("  Warning: no aux_channels in config — skipping.")
        return False

    aux_channels = build_aux_channels(data_paths, aux_cfgs, layout_path)
    if not aux_channels:
        print("  No aux channels loaded — skipping.")
        return False

    decomp["aux_channels"] = aux_channels
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(decomp, f)
    print(f"  Saved -> {out_path.name}  ({len(aux_channels)} aux channel(s))")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill force/aux-channel data into decomposition pkl files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # single-file args
    parser.add_argument("--decomp", help="Single decomposition .pkl to update")
    parser.add_argument(
        "--data", nargs="+",
        help="Original recording file(s). Pass multiple for concat decompositions.",
    )
    parser.add_argument("--output", help="Output path (default: overwrite --decomp)")
    # batch args
    parser.add_argument("--decomp-dir", help="Directory of decomposition .pkl files")
    parser.add_argument("--data-dir",   help="Directory of original recording files")
    parser.add_argument("--output-dir", help="Output directory for updated pkls")
    # shared
    parser.add_argument("--config",  required=True, help="Channel-config JSON")
    parser.add_argument("--layout",  default=None,  help="Data-layout YAML (non-OTB only)")
    args = parser.parse_args()

    config_path = Path(args.config)
    layout_path = Path(args.layout) if args.layout else None

    if not config_path.exists():
        sys.exit(f"Error: --config not found: {config_path}")
    if layout_path and not layout_path.exists():
        sys.exit(f"Error: --layout not found: {layout_path}")

    # ── batch mode ──────────────────────────────────────────────────────────
    if args.decomp_dir:
        if not args.data_dir or not args.output_dir:
            sys.exit("Batch mode requires --decomp-dir, --data-dir, and --output-dir.")

        decomp_dir = Path(args.decomp_dir)
        data_dir   = Path(args.data_dir)
        out_dir    = Path(args.output_dir)

        pkls = sorted(decomp_dir.glob("*.pkl"))
        if not pkls:
            sys.exit(f"No .pkl files found in {decomp_dir}")

        n_ok = n_skip = n_fail = n_exists = 0

        for pkl in pkls:
            stem = pkl.stem

            # Never overwrite existing output
            out_path = out_dir / pkl.name
            if out_path.exists():
                print(f"[skip-exists]  {pkl.name}")
                n_exists += 1
                continue

            # Skip edited files
            if "_edited" in stem:
                print(f"[skip-edited]  {pkl.name}")
                n_skip += 1
                continue

            if not stem.endswith("_decomp_output"):
                print(f"[skip-unknown] {pkl.name} — unexpected name pattern")
                n_skip += 1
                continue

            raw_stem = stem[: -len("_decomp_output")]

            # ── concat pkl ──────────────────────────────────────────────────
            if raw_stem.endswith("_concat"):
                data_paths = _find_source_files_for_concat(raw_stem, data_dir)
                if data_paths is None:
                    print(
                        f"[skip-concat]  {pkl.name} — could not find all source files"
                    )
                    n_skip += 1
                    continue
                label = f"{len(data_paths)} concat files"

            # ── single pkl ──────────────────────────────────────────────────
            else:
                data_path = None
                for ext in (".otb+", ".otb", ".h5", ".hdf5", ".mat"):
                    candidate = data_dir / (raw_stem + ext)
                    if candidate.exists():
                        data_path = candidate
                        break
                if data_path is None:
                    print(f"[skip-no-data] {pkl.name} — no matching data file")
                    n_skip += 1
                    continue
                data_paths = [data_path]
                label = data_path.name

            print(f"[process] {pkl.name}")
            print(f"          data: {label}")
            try:
                ok = process_one(pkl, data_paths, config_path, layout_path, out_path)
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as ex:
                print(f"  ERROR: {ex}")
                n_fail += 1

        print(
            f"\nBatch complete: {n_ok} processed, {n_skip} skipped, "
            f"{n_exists} already existed, {n_fail} failed."
        )
        return

    # ── single-file mode ────────────────────────────────────────────────────
    if not args.decomp or not args.data:
        sys.exit(
            "Provide (--decomp + --data) for single-file mode "
            "or (--decomp-dir + --data-dir + --output-dir) for batch mode."
        )

    decomp_path = Path(args.decomp)
    data_paths  = [Path(p) for p in args.data]
    out_path    = Path(args.output) if args.output else decomp_path

    if not decomp_path.exists():
        sys.exit(f"Error: --decomp not found: {decomp_path}")
    for dp in data_paths:
        if not dp.exists():
            sys.exit(f"Error: --data file not found: {dp}")
    if out_path.exists() and out_path != decomp_path:
        sys.exit(
            f"Output already exists: {out_path}\n"
            "Use --output to choose a different path or delete the existing file."
        )

    print(f"Processing: {decomp_path.name}")
    for dp in data_paths:
        print(f"      data: {dp.name}")
    try:
        ok = process_one(decomp_path, data_paths, config_path, layout_path, out_path)
        sys.exit(0 if ok else 1)
    except Exception as ex:
        sys.exit(f"ERROR: {ex}")


if __name__ == "__main__":
    main()
