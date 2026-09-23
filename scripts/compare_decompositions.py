"""
compare_decompositions.py
─────────────────────────
Load edited and original decomposition files, compare sources, filters,
and timestamps side-by-side.

Usage:
    python scripts/compare_decompositions.py --original original.pkl --edited edited.pkl
    python scripts/compare_decompositions.py --original original.pkl --edited edited.pkl --port 0 --unit 2
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def to_numpy(obj) -> np.ndarray:
    if obj is None:
        return np.array([])
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_filters(data: dict, port_idx: int) -> list:
    """Extract per-unit filters for a port."""
    mu_filters = data.get("mu_filters", [])
    if port_idx >= len(mu_filters) or mu_filters[port_idx] is None:
        return []
    raw = mu_filters[port_idx]
    if isinstance(raw, list):
        return [to_numpy(f) if f is not None else None for f in raw]
    elif isinstance(raw, np.ndarray):
        if raw.ndim >= 2:
            return [raw[i] for i in range(raw.shape[0])]
        return [raw]
    return []


def get_timestamps(data: dict, port_idx: int) -> list:
    dt = data.get("discharge_times", [])
    if port_idx >= len(dt):
        return []
    port_dt = dt[port_idx]
    if isinstance(port_dt, list):
        return [to_numpy(x).flatten().astype(np.int64) for x in port_dt]
    return [to_numpy(port_dt).flatten().astype(np.int64)]


def get_sources(data: dict, port_idx: int) -> list:
    pt = data.get("pulse_trains", [])
    if port_idx >= len(pt):
        return []
    port_pt = pt[port_idx]
    if isinstance(port_pt, list):
        return [to_numpy(x).flatten() for x in port_pt]
    return [to_numpy(port_pt).flatten()]


def get_fsamp(data: dict) -> float:
    return float(data.get("sampling_rate", data.get("fsamp", 2048)))


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_sources_comparison(
    orig_sources: list,
    edit_sources: list,
    orig_timestamps: list,
    edit_timestamps: list,
    fsamp: float,
    port_name: str,
    unit_idx: int | None = None,
):
    """Plot original and edited sources with y-offset, timestamps marked."""
    indices = (
        [unit_idx]
        if unit_idx is not None
        else range(min(len(orig_sources), len(edit_sources)))
    )

    n_units = len(list(indices)) if unit_idx is None else 1
    if n_units == 0:
        print("No units to compare.")
        return

    # Reset indices for iteration
    indices = (
        [unit_idx]
        if unit_idx is not None
        else list(range(min(len(orig_sources), len(edit_sources))))
    )

    fig, ax = plt.subplots(figsize=(18, 3 + 2.5 * n_units))
    fig.suptitle(
        f"Source Comparison — Port: {port_name}", fontsize=14, fontweight="bold"
    )

    offset_step = 0
    y_ticks = []
    y_labels = []

    for rank, mu_idx in enumerate(indices):
        src_orig = orig_sources[mu_idx] if mu_idx < len(orig_sources) else np.zeros(1)
        src_edit = edit_sources[mu_idx] if mu_idx < len(edit_sources) else np.zeros(1)
        ts_orig = (
            orig_timestamps[mu_idx] if mu_idx < len(orig_timestamps) else np.array([])
        )
        ts_edit = (
            edit_timestamps[mu_idx] if mu_idx < len(edit_timestamps) else np.array([])
        )

        # Compute offset: enough to separate units
        amp_orig = np.ptp(src_orig) if len(src_orig) > 1 else 1.0
        amp_edit = np.ptp(src_edit) if len(src_edit) > 1 else 1.0
        offset_step = max(amp_orig, amp_edit) * 1.3

        y_base = -rank * offset_step

        # Plot original (blue)
        t_orig = np.arange(len(src_orig)) / fsamp
        ax.plot(
            t_orig,
            src_orig + y_base,
            color="#2b6cb0",
            alpha=0.7,
            linewidth=0.8,
            label="Original" if rank == 0 else None,
        )

        # Plot edited (red)
        t_edit = np.arange(len(src_edit)) / fsamp
        ax.plot(
            t_edit,
            src_edit + y_base,
            color="#e53e3e",
            alpha=0.7,
            linewidth=0.8,
            label="Edited" if rank == 0 else None,
        )

        # Timestamps — original (blue dots)
        ts_o_valid = ts_orig[(ts_orig >= 0) & (ts_orig < len(src_orig))]
        if len(ts_o_valid) > 0:
            ax.plot(
                ts_o_valid / fsamp,
                src_orig[ts_o_valid] + y_base,
                "o",
                color="#2b6cb0",
                markersize=3,
                alpha=0.6,
            )

        # Timestamps — edited (red dots)
        ts_e_valid = ts_edit[(ts_edit >= 0) & (ts_edit < len(src_edit))]
        if len(ts_e_valid) > 0:
            ax.plot(
                ts_e_valid / fsamp,
                src_edit[ts_e_valid] + y_base,
                "o",
                color="#e53e3e",
                markersize=3,
                alpha=0.6,
            )

        y_ticks.append(y_base)
        y_labels.append(f"MU {mu_idx}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_filters_comparison(
    orig_filters: list,
    edit_filters: list,
    port_name: str,
    unit_idx: int | None = None,
    orig_filters_saved: list | None = None,
):
    """Plot filter vectors side-by-side. If edited file has mu_filters_original,
    also show those."""
    indices = (
        [unit_idx]
        if unit_idx is not None
        else list(range(min(len(orig_filters), len(edit_filters))))
    )

    n_units = len(indices)
    if n_units == 0:
        print("No filters to compare.")
        return

    fig, axes = plt.subplots(n_units, 1, figsize=(14, 3 * n_units), squeeze=False)
    fig.suptitle(
        f"Filter Comparison — Port: {port_name}", fontsize=14, fontweight="bold"
    )

    for rank, mu_idx in enumerate(indices):
        ax = axes[rank, 0]

        f_orig = orig_filters[mu_idx] if mu_idx < len(orig_filters) else None
        f_edit = edit_filters[mu_idx] if mu_idx < len(edit_filters) else None

        if f_orig is not None:
            f_orig = f_orig.flatten()
            ax.plot(
                f_orig, color="#2b6cb0", alpha=0.8, linewidth=1.2, label="Original file"
            )

        if f_edit is not None:
            f_edit = f_edit.flatten()
            ax.plot(
                f_edit, color="#e53e3e", alpha=0.8, linewidth=1.2, label="Edited file"
            )

        if orig_filters_saved is not None and mu_idx < len(orig_filters_saved):
            f_saved = orig_filters_saved[mu_idx]
            if f_saved is not None:
                f_saved = f_saved.flatten()
                ax.plot(
                    f_saved,
                    color="#38a169",
                    alpha=0.6,
                    linewidth=1.0,
                    linestyle="--",
                    label="Saved original (in edited file)",
                )

        # Compute cosine similarity if both exist
        if f_orig is not None and f_edit is not None and len(f_orig) == len(f_edit):
            dot = np.dot(f_orig, f_edit)
            norm = np.linalg.norm(f_orig) * np.linalg.norm(f_edit)
            cos_sim = dot / norm if norm > 0 else 0
            l2_diff = np.linalg.norm(f_orig - f_edit)
            ax.set_title(
                f"MU {mu_idx} — cosine similarity: {cos_sim:.4f}, "
                f"L2 diff: {l2_diff:.6f}",
                fontsize=10,
            )
        else:
            ax.set_title(f"MU {mu_idx}", fontsize=10)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Filter coefficient index")
        ax.set_ylabel("Weight")

    plt.tight_layout()
    plt.show()


def plot_timestamp_comparison(
    orig_timestamps: list,
    edit_timestamps: list,
    fsamp: float,
    port_name: str,
    unit_idx: int | None = None,
):
    """Compare timestamps: IFR overlay + added/removed spike analysis."""
    indices = (
        [unit_idx]
        if unit_idx is not None
        else list(range(min(len(orig_timestamps), len(edit_timestamps))))
    )

    n_units = len(indices)
    if n_units == 0:
        print("No timestamps to compare.")
        return

    fig, axes = plt.subplots(n_units, 2, figsize=(18, 3 * n_units), squeeze=False)
    fig.suptitle(
        f"Timestamp Comparison — Port: {port_name}", fontsize=14, fontweight="bold"
    )

    for rank, mu_idx in enumerate(indices):
        ts_orig = (
            orig_timestamps[mu_idx] if mu_idx < len(orig_timestamps) else np.array([])
        )
        ts_edit = (
            edit_timestamps[mu_idx] if mu_idx < len(edit_timestamps) else np.array([])
        )

        # ── Left: IFR overlay ──
        ax_ifr = axes[rank, 0]

        for ts, color, label in [
            (ts_orig, "#2b6cb0", "Original"),
            (ts_edit, "#e53e3e", "Edited"),
        ]:
            ts_sorted = np.sort(ts)
            if len(ts_sorted) >= 2:
                isi = np.diff(ts_sorted) / fsamp
                ifr = np.where(isi > 0.01, 1.0 / isi, 0.0)
                t_mid = (ts_sorted[:-1] + ts_sorted[1:]) / 2 / fsamp
                ax_ifr.plot(
                    t_mid, ifr, color=color, alpha=0.7, linewidth=1.2, label=label
                )

        ax_ifr.set_title(f"MU {mu_idx} — IFR", fontsize=10)
        ax_ifr.set_xlabel("Time (s)")
        ax_ifr.set_ylabel("Firing rate (Hz)")
        ax_ifr.legend(fontsize=8)
        ax_ifr.grid(True, alpha=0.2)

        # ── Right: Added / Removed analysis ──
        ax_diff = axes[rank, 1]

        set_orig = set(ts_orig.tolist()) if len(ts_orig) > 0 else set()
        set_edit = set(ts_edit.tolist()) if len(ts_edit) > 0 else set()

        added = sorted(set_edit - set_orig)
        removed = sorted(set_orig - set_edit)
        common = sorted(set_orig & set_edit)

        # Bar chart summary
        categories = ["Original", "Edited", "Common", "Added", "Removed"]
        counts = [len(set_orig), len(set_edit), len(common), len(added), len(removed)]
        colors = ["#2b6cb0", "#e53e3e", "#805ad5", "#38a169", "#dd6b20"]

        bars = ax_diff.bar(categories, counts, color=colors, alpha=0.8)
        for bar, count in zip(bars, counts, strict=True):
            ax_diff.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax_diff.set_title(f"MU {mu_idx} — Spike Changes", fontsize=10)
        ax_diff.set_ylabel("Count")
        ax_diff.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()


def print_summary(
    orig_data: dict,
    edit_data: dict,
    port_idx: int,
    port_name: str,
):
    """Print a text summary of differences."""
    orig_ts = get_timestamps(orig_data, port_idx)
    edit_ts = get_timestamps(edit_data, port_idx)
    orig_filt = get_filters(orig_data, port_idx)
    edit_filt = get_filters(edit_data, port_idx)
    fsamp = get_fsamp(orig_data)

    n_orig = len(orig_ts)
    n_edit = len(edit_ts)

    print("\n" + "=" * 70)
    print(f"  COMPARISON SUMMARY — Port: {port_name}")
    print("=" * 70)
    print(f"  Original: {n_orig} units")
    print(f"  Edited:   {n_edit} units")
    print(f"  Fsamp:    {fsamp} Hz")
    print("-" * 70)

    for mu_idx in range(max(n_orig, n_edit)):
        print(f"\n  MU {mu_idx}:")

        # Timestamps
        ts_o = orig_ts[mu_idx] if mu_idx < n_orig else np.array([])
        ts_e = edit_ts[mu_idx] if mu_idx < n_edit else np.array([])
        set_o = set(ts_o.tolist()) if len(ts_o) > 0 else set()
        set_e = set(ts_e.tolist()) if len(ts_e) > 0 else set()

        added = len(set_e - set_o)
        removed = len(set_o - set_e)
        common = len(set_o & set_e)

        print(
            f"    Timestamps: {len(set_o)} → {len(set_e)}  "
            f"(+{added} added, -{removed} removed, {common} unchanged)"
        )

        # IFR
        for label, ts in [("orig", ts_o), ("edit", ts_e)]:
            if len(ts) >= 2:
                isi = np.diff(np.sort(ts)) / fsamp
                ifr = np.where(isi > 0.01, 1.0 / isi, 0.0)
                print(
                    f"    IFR ({label}): mean={ifr.mean():.1f} Hz, "
                    f"std={ifr.std():.1f}, range=[{ifr.min():.1f}, {ifr.max():.1f}]"
                )

        # CoV
        for label, ts in [("orig", ts_o), ("edit", ts_e)]:
            if len(ts) >= 3:
                isi = np.diff(np.sort(ts)).astype(float)
                isi = isi[isi < 5 * np.median(isi)]
                if len(isi) > 1:
                    cov = np.std(isi) / np.mean(isi)
                    print(f"    CoV  ({label}): {cov:.4f}")

        # Filter similarity
        f_o = (
            orig_filt[mu_idx].flatten()
            if mu_idx < len(orig_filt) and orig_filt[mu_idx] is not None
            else None
        )
        f_e = (
            edit_filt[mu_idx].flatten()
            if mu_idx < len(edit_filt) and edit_filt[mu_idx] is not None
            else None
        )

        if f_o is not None and f_e is not None and len(f_o) == len(f_e):
            dot = np.dot(f_o, f_e)
            norm = np.linalg.norm(f_o) * np.linalg.norm(f_e)
            cos_sim = dot / norm if norm > 0 else 0
            l2 = np.linalg.norm(f_o - f_e)
            print(f"    Filter: cosine_sim={cos_sim:.6f}, L2_diff={l2:.6f}")
            if cos_sim > 0.999:
                print("            → Filter UNCHANGED")
            elif cos_sim > 0.95:
                print("            → Filter slightly modified")
            else:
                print("            → Filter SIGNIFICANTLY changed")
        elif f_o is not None and f_e is None:
            print("    Filter: original exists, edited is None")
        elif f_o is None and f_e is not None:
            print("    Filter: original is None, edited exists")

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original and edited SCD decomposition files."
    )
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--edited", required=True, type=Path)
    parser.add_argument("--port", type=int, default=None, help="Port index to plot")
    parser.add_argument("--unit", type=int, default=None, help="Unit index to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    original_path = args.original
    edited_path = args.edited
    port_idx_filter = args.port
    unit_idx_filter = args.unit

    orig = load_pkl(original_path)
    edit = load_pkl(edited_path)

    ports = orig.get("ports", edit.get("ports", []))
    fsamp = get_fsamp(orig)

    print(f"\nPorts: {ports}")
    print(f"Sampling rate: {fsamp} Hz")

    # Check for saved original filters in edited file
    has_orig_filters_in_edit = "mu_filters_original" in edit

    port_indices = (
        [port_idx_filter] if port_idx_filter is not None else list(range(len(ports)))
    )

    for port_idx in port_indices:
        if port_idx >= len(ports):
            print(f"Port index {port_idx} out of range.")
            continue

        port_name = ports[port_idx]

        orig_sources = get_sources(orig, port_idx)
        edit_sources = get_sources(edit, port_idx)
        orig_ts = get_timestamps(orig, port_idx)
        edit_ts = get_timestamps(edit, port_idx)
        orig_filt = get_filters(orig, port_idx)
        edit_filt = get_filters(edit, port_idx)

        # Original filters saved inside edited file
        orig_filt_saved = None
        if has_orig_filters_in_edit:
            raw = edit["mu_filters_original"]
            if port_idx < len(raw) and raw[port_idx] is not None:
                pf = raw[port_idx]
                if isinstance(pf, list):
                    orig_filt_saved = [
                        to_numpy(f) if f is not None else None for f in pf
                    ]
                elif isinstance(pf, np.ndarray) and pf.ndim >= 2:
                    orig_filt_saved = [pf[i] for i in range(pf.shape[0])]

        # Text summary
        print_summary(orig, edit, port_idx, port_name)

        # Plots
        print(f"\nPlotting sources for port '{port_name}'...")
        plot_sources_comparison(
            orig_sources,
            edit_sources,
            orig_ts,
            edit_ts,
            fsamp,
            port_name,
            unit_idx=unit_idx_filter,
        )

        print(f"Plotting filters for port '{port_name}'...")
        plot_filters_comparison(
            orig_filt,
            edit_filt,
            port_name,
            unit_idx=unit_idx_filter,
            orig_filters_saved=orig_filt_saved,
        )

        print(f"Plotting timestamps for port '{port_name}'...")
        plot_timestamp_comparison(
            orig_ts, edit_ts, fsamp, port_name, unit_idx=unit_idx_filter
        )


if __name__ == "__main__":
    main()
