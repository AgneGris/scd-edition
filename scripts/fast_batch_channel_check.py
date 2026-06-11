"""
fast_batch_channel_check.py
───────────────────────────
Like batch_channel_check.py, but pre-populates each new file with the union of
channel rejections from all previously inspected files (per grid, matched by
name).  This removes the need to re-select the same bad channels for every
trial when you need consistent rejections across all recordings.

Colour coding
  Orange  – inherited rejection (flagged in a prior file; you can un-reject)
  Red     – manually rejected in this session
  Active channels are drawn normally in blue.

Clicking a channel cycles:  active → manual-rejected → active
                            inherited → active → manual-rejected → active

Output JSON format is identical to batch_channel_check.py.

Usage
─────
    python scripts/fast_batch_channel_check.py \\
        --channel-config  path/to/channel_config.json \\
        --files           data/*.otb+ \\
        --output          results/channel_rejections.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── colours ───────────────────────────────────────────────────────────────────

COLORS = {
    "background": "#1e1e1e",
    "background_light": "#2d2d2d",
    "background_hover": "#3d3d3d",
    "foreground": "#d4d4d4",
    "text_muted": "#858585",
    "info": "#4ec9b0",
    "info_light": "#9cdcfe",
    "error": "#f44747",
    "inherited": "#FFA500",
    "success": "#4caf50",
    "border": "#404040",
}

# Channel state constants
_ACTIVE = 0
_INHERITED = 1
_MANUAL = 2


# ── Force overlay helpers (mirror batch_channel_check.py) ─────────────────────


def _parse_filename_task(stem: str):
    import re

    _FINGER = {"T": "thumb", "I": "index", "M": "middle", "R": "ring", "L": "little"}
    direction = None
    fingers = []
    mvc_m = re.search(r"mvc-([^_]+)", stem, re.IGNORECASE)
    if mvc_m:
        mvc_str = mvc_m.group(1).lower()
        if "ext" in mvc_str:
            direction = "ext"
        elif "flex" in mvc_str:
            direction = "flex"
    fing_m = re.search(r"fing-([A-Za-z]+)", stem)
    if fing_m:
        fingers = [_FINGER[c] for c in fing_m.group(1).upper() if c in _FINGER]
    return direction, fingers


def _select_aux_for_task(aux_cfgs: list, direction, fingers: list) -> list:
    if not aux_cfgs:
        return []
    labeled = [a for a in aux_cfgs if a.get("unit", "").strip()]
    if not labeled:
        return aux_cfgs
    if direction is None or not fingers:
        return labeled
    matched = [
        a
        for a in labeled
        if any(f in a["unit"].lower() for f in fingers)
        and direction in a["unit"].lower()
    ]
    return matched if matched else labeled


def _compute_inherited_masks(
    grid_list: list, rejections: dict, current_fname: str
) -> List[np.ndarray]:
    """Union of channel rejections from all files except current_fname."""
    inherited: List[np.ndarray] = []
    for port_name, cfg in grid_list:
        n = len(cfg["channels"])
        union = np.zeros(n, dtype=int)
        for fname, grids in rejections.items():
            if fname == current_fname:
                continue
            entry = grids.get(port_name)
            if entry is None:
                continue
            ch_mask = entry["channels"] if isinstance(entry, dict) else entry
            arr = np.array(ch_mask, dtype=int)
            if arr.shape[0] == n:
                union = np.maximum(union, arr)
        inherited.append(union)
    return inherited


# ── GUI ────────────────────────────────────────────────────────────────────────


def _run_channel_check_gui(
    file_paths: List[Path],
    layout: dict,
    grid_configs: dict,
    output_path: Path,
    existing_rejections: dict,
    sampling_rate: int = 10240,
    aux_configs: list = None,
) -> dict:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
    from PyQt5.QtCore import QEventLoop, QTimer
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button, SpanSelector
    from scd_app.io.data_loader import load_field

    app = QApplication.instance() or QApplication(sys.argv)

    rejections: dict = dict(existing_rejections)

    class ChannelCheckWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Fast Batch Channel Inspection")
            self.resize(1600, 1000)
            self.setStyleSheet(f"background-color: {COLORS['background']};")

            central = QWidget()
            self.setCentralWidget(central)
            vbox = QVBoxLayout(central)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(0)

            self.figure = Figure(facecolor=COLORS["background"])
            self.canvas = FigureCanvas(self.figure)
            vbox.addWidget(self.canvas, 1)

            self._current_loop = None
            self.aborted = False
            self.show()

        def closeEvent(self, event):
            self.aborted = True
            if self._current_loop is not None:
                self._current_loop.quit()
            event.accept()

    win = ChannelCheckWindow()

    for file_idx, file_path in enumerate(file_paths):
        fname = file_path.name
        n_files = len(file_paths)
        app.processEvents()

        try:
            emg = load_field(file_path, layout, "emg")
        except Exception as exc:
            print(f"  ERROR loading {fname}: {exc} — skipping.")
            continue

        grid_list = list(grid_configs.items())

        # Compute inherited mask from all already-inspected files
        inherited_masks = _compute_inherited_masks(grid_list, rejections, fname)

        # state_masks: 0=active, 1=inherited, 2=manual
        state_masks: List[np.ndarray] = []
        time_masks_list: List[List] = []
        saved_file = rejections.get(fname, {})

        for i, (port_name, cfg) in enumerate(grid_list):
            n = len(cfg["channels"])
            prev = saved_file.get(port_name)
            if prev is None:
                # New file: pre-populate with inherited
                state = inherited_masks[i].copy()  # 0 or 1
                time_masks_list.append([])
            elif isinstance(prev, list):
                # Old JSON format: load as inherited (we can't distinguish origin)
                state = np.where(np.array(prev, dtype=int) != 0, _INHERITED, _ACTIVE)
                time_masks_list.append([])
            else:
                state = np.where(
                    np.array(prev.get("channels", [0] * n), dtype=int) != 0,
                    _INHERITED,
                    _ACTIVE,
                )
                time_masks_list.append(list(prev.get("time_masks", [])))
            state_masks.append(state)

        _aux_cfgs = aux_configs or []
        _stem = file_path.stem
        _direction, _fingers = _parse_filename_task(_stem)
        aux_to_show = _select_aux_for_task(_aux_cfgs, _direction, _fingers)
        if aux_to_show:
            print(
                f"  [force overlay] {fname}: direction={_direction!r}  "
                f"fingers={_fingers}  channels={[a.get('unit') for a in aux_to_show]}"
            )

        # Count how many channels were pre-inherited
        n_pre = sum(int(np.sum(sm == _INHERITED)) for sm in state_masks)
        if n_pre:
            print(f"  [{fname}] pre-inherited {n_pre} channel rejection(s) from prior files")

        nav = {
            "current": 0,
            "cids": [],
            "buttons": [],
            "time_mask_mode": False,
            "span_selector": None,
            "show_force": bool(aux_to_show),
        }
        event_loop = QEventLoop()

        def draw_grid(
            grid_idx,
            _emg=emg,
            _grid_list=grid_list,
            _state_masks=state_masks,
            _time_masks_list=time_masks_list,
            _sampling_rate=sampling_rate,
            _nav=nav,
            _win=win,
            _loop=event_loop,
            _file_idx=file_idx,
            _fname=fname,
            _n_files=n_files,
            _aux_to_show=aux_to_show,
        ):
            _win.figure.clf()
            port_name, cfg = _grid_list[grid_idx]
            channels = cfg["channels"]
            n_channels = len(channels)
            state = _state_masks[grid_idx]
            is_first = grid_idx == 0
            is_last = grid_idx == len(_grid_list) - 1

            ax = _win.figure.add_axes([0.03, 0.18, 0.94, 0.80])
            ax.set_facecolor(COLORS["background"])

            _win.figure.text(
                0.5,
                0.985,
                f"[{_file_idx + 1}/{_n_files}]  {_fname}   —   "
                f"{port_name}  ({grid_idx + 1}/{len(_grid_list)})",
                ha="center",
                va="center",
                fontsize=18,
                weight="bold",
                color=COLORS["foreground"],
            )
            _win.figure.text(
                0.5,
                0.152,
                "Click = Toggle channel  |  Scroll = Zoom  |  Right-drag = Pan  |  R = Reset",
                ha="center",
                va="center",
                fontsize=14,
                color=COLORS["info"],
            )
            # Legend for colour coding
            _win.figure.text(
                0.5,
                0.170,
                "■ orange = inherited from prior file   ■ red = rejected this session",
                ha="center",
                va="center",
                fontsize=11,
                color=COLORS["text_muted"],
            )

            grid_data = _emg[:, channels].numpy()
            step = max(1, grid_data.shape[0] // 4000)
            disp = grid_data[::step, :]
            max_len = disp.shape[0]

            active_idx = np.where(state == _ACTIVE)[0]
            ref = disp[:, active_idx] if len(active_idx) > 0 else disp
            active_std = float(np.std(ref))
            separation = active_std * 15 if active_std > 0 else 1.0

            _tm = _time_masks_list[grid_idx]
            is_mask_mode = _nav.get("time_mask_mode", False)
            for t_start_s, t_end_s in _tm:
                disp_start = t_start_s * _sampling_rate / step
                disp_end = t_end_s * _sampling_rate / step
                ax.axvspan(
                    disp_start, disp_end, alpha=0.20, color=COLORS["error"], zorder=0
                )

            for ch in range(n_channels):
                y = ch * separation
                if state[ch] == _MANUAL:
                    ax.plot(
                        [0, max_len],
                        [y, y],
                        color=COLORS["error"],
                        alpha=0.25,
                        linewidth=0.8,
                        linestyle="--",
                    )
                elif state[ch] == _INHERITED:
                    ax.plot(
                        [0, max_len],
                        [y, y],
                        color=COLORS["inherited"],
                        alpha=0.25,
                        linewidth=0.8,
                        linestyle="--",
                    )
                else:
                    ax.plot(disp[:, ch] + y, color="#4a9eff", alpha=0.8, linewidth=1.0)

            for ch in range(n_channels):
                if state[ch] == _MANUAL:
                    label_color = COLORS["error"]
                elif state[ch] == _INHERITED:
                    label_color = COLORS["inherited"]
                else:
                    label_color = COLORS["text_muted"]
                ax.text(
                    -max_len * 0.01,
                    ch * separation,
                    str(ch),
                    color=label_color,
                    fontsize=12,
                    ha="right",
                    va="center",
                )

            ax.set_xlim(0, max_len)
            ax.set_ylim(-separation, n_channels * separation)
            ax.axis("off")

            if _aux_to_show and _nav.get("show_force", False):
                total_height = n_channels * separation
                force_color = "#FFD700"
                first_label = True
                import torch as _torch

                for a in _aux_to_show:
                    s = int(a.get("start_chan", 0))
                    e = int(a.get("end_chan", s + 1))
                    n_total_ch = _emg.shape[1]
                    if s >= e or e > n_total_ch:
                        continue
                    raw_full = _emg[:, s:e]
                    if _torch.is_tensor(raw_full):
                        raw_full = raw_full.numpy()
                    raw = raw_full[::step, :].mean(axis=1)
                    sig = raw - raw.mean()
                    peak = max(abs(sig.max()), abs(sig.min()), 1e-9)
                    sig_scaled = sig / peak * (0.35 * total_height) + total_height / 2
                    label = (
                        a.get("unit", a.get("name", "force")) if first_label else "_"
                    )
                    ax.plot(
                        sig_scaled,
                        color=force_color,
                        alpha=0.55,
                        linewidth=1.2,
                        zorder=3,
                        label=label,
                    )
                    first_label = False
                ax.legend(
                    loc="upper right",
                    fontsize=9,
                    facecolor=COLORS["background_light"],
                    labelcolor=force_color,
                    framealpha=0.7,
                )

            n_inherited = int(np.sum(state == _INHERITED))
            n_manual = int(np.sum(state == _MANUAL))
            n_tmask = len(_tm)
            status_parts = []
            if n_inherited:
                status_parts.append(
                    f"{n_inherited} inherited rejection{'s' if n_inherited != 1 else ''}"
                )
            if n_manual:
                status_parts.append(
                    f"{n_manual} manual rejection{'s' if n_manual != 1 else ''}"
                )
            if n_tmask:
                status_parts.append(
                    f"{n_tmask} time region{'s' if n_tmask != 1 else ''} masked"
                )
            _win.figure.text(
                0.5,
                0.124,
                "  |  ".join(status_parts),
                ha="center",
                va="center",
                fontsize=14,
                color=COLORS["error"],
            )
            if is_mask_mode:
                _win.figure.text(
                    0.5,
                    0.096,
                    "TIME MASK MODE: drag to add region  |  Z = undo last  |  T or button = exit",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=COLORS["info"],
                )

            prev_ax = _win.figure.add_axes([0.05, 0.012, 0.09, 0.065])
            mask_ax = _win.figure.add_axes([0.16, 0.012, 0.10, 0.065])
            confirm_ax = _win.figure.add_axes([0.42, 0.012, 0.14, 0.065])
            next_ax = _win.figure.add_axes([0.80, 0.012, 0.15, 0.065])

            prev_btn = Button(
                prev_ax,
                "← Previous",
                color=COLORS["background_light"],
                hovercolor=COLORS["background_hover"],
            )
            prev_btn.label.set_color(
                COLORS["foreground"] if not is_first else COLORS["text_muted"]
            )
            prev_btn.label.set_fontsize(15)

            mask_btn = Button(
                mask_ax,
                "Exit Mask Mode" if is_mask_mode else "Mask Times",
                color=COLORS["error"] if is_mask_mode else COLORS["background_light"],
                hovercolor=COLORS["background_hover"],
            )
            mask_btn.label.set_color("white" if is_mask_mode else COLORS["foreground"])
            mask_btn.label.set_fontsize(14)

            if not is_last:
                next_label = "Next →"
                next_color = COLORS["info"]
            elif _file_idx < _n_files - 1:
                next_label = "Save & Next File →"
                next_color = COLORS["success"]
            else:
                next_label = "Save & Done ✓"
                next_color = COLORS["success"]

            next_btn = Button(
                next_ax,
                next_label,
                color=next_color,
                hovercolor=COLORS["background_hover"],
            )
            next_btn.label.set_color("white")
            next_btn.label.set_fontsize(15)
            next_btn.label.set_weight("bold")

            confirm_btn = Button(
                confirm_ax, "CONFIRM ALL", color="#555555", hovercolor=COLORS["success"]
            )
            confirm_btn.label.set_color(COLORS["foreground"])
            confirm_btn.label.set_fontsize(14)

            check = None
            if _aux_to_show:
                from matplotlib.widgets import CheckButtons

                cb_ax = _win.figure.add_axes([0.28, 0.012, 0.12, 0.065])
                cb_ax.set_facecolor(COLORS["background"])
                check = CheckButtons(
                    cb_ax, ["Show force"], [_nav.get("show_force", True)]
                )
                check.labels[0].set_color(COLORS["foreground"])
                check.labels[0].set_fontsize(11)
                for r in getattr(check, "rectangles", []):
                    r.set_edgecolor(COLORS["text_muted"])
                    r.set_facecolor(COLORS["background_light"])

                def on_force_toggle(
                    _label, _check=check, _nav=_nav, _grid_idx=grid_idx
                ):
                    _nav["show_force"] = _check.get_status()[0]
                    disconnect()
                    draw_grid(_grid_idx)

                check.on_clicked(on_force_toggle)

            istate = {
                "press_event": None,
                "last_scroll_time": 0.0,
                "panning": False,
                "pan_start": None,
                "pan_xlim": None,
                "pan_ylim": None,
            }

            def on_press(ev, _ax=ax, _st=istate):
                if ev.inaxes != _ax:
                    return
                if ev.button == 1:
                    _st["press_event"] = ev
                elif ev.button == 3:
                    _st["panning"] = True
                    _st["pan_start"] = (ev.x, ev.y)
                    _st["pan_xlim"] = _ax.get_xlim()
                    _st["pan_ylim"] = _ax.get_ylim()

            def on_motion(ev, _ax=ax, _st=istate, _canvas=_win.canvas):
                if not _st["panning"] or ev.inaxes != _ax:
                    return
                dx = ev.x - _st["pan_start"][0]
                dy = ev.y - _st["pan_start"][1]
                xlim = _st["pan_xlim"]
                ylim = _st["pan_ylim"]
                bbox = _ax.get_window_extent()
                data_dx = -(dx / bbox.width) * (xlim[1] - xlim[0])
                data_dy = -(dy / bbox.height) * (ylim[1] - ylim[0])
                _ax.set_xlim(xlim[0] + data_dx, xlim[1] + data_dx)
                _ax.set_ylim(ylim[0] + data_dy, ylim[1] + data_dy)
                _canvas.draw_idle()

            _TOGGLE = {_ACTIVE: _MANUAL, _INHERITED: _ACTIVE, _MANUAL: _ACTIVE}

            def on_release(
                ev,
                _ax=ax,
                _st=istate,
                _state=state,
                _n_ch=n_channels,
                _sep=separation,
                _nav=_nav,
                _grid_idx=grid_idx,
            ):
                if ev.button == 3:
                    _st["panning"] = False
                    return
                if _nav.get("time_mask_mode", False):
                    _st["press_event"] = None
                    return
                if ev.button != 1 or _st["press_event"] is None:
                    return
                press = _st["press_event"]
                _st["press_event"] = None
                elapsed_ms = (time.time() - _st["last_scroll_time"]) * 1000
                if elapsed_ms < 300:
                    return
                if abs(ev.x - press.x) > 5 or abs(ev.y - press.y) > 5:
                    return
                if ev.inaxes != _ax or ev.ydata is None:
                    return
                closest, min_d = None, float("inf")
                for ch in range(_n_ch):
                    d = abs(ev.ydata - ch * _sep)
                    if d < _sep * 0.6 and d < min_d:
                        min_d, closest = d, ch
                if closest is None:
                    return
                _state[closest] = _TOGGLE[_state[closest]]
                disconnect()
                draw_grid(_grid_idx)

            def on_scroll(ev, _ax=ax, _st=istate, _canvas=_win.canvas):
                if ev.inaxes != _ax:
                    return
                _st["last_scroll_time"] = time.time()
                xlim, ylim = _ax.get_xlim(), _ax.get_ylim()
                scale = 0.85 if ev.button == "up" else 1.18
                _ax.set_xlim(
                    ev.xdata - (ev.xdata - xlim[0]) * scale,
                    ev.xdata + (xlim[1] - ev.xdata) * scale,
                )
                _ax.set_ylim(
                    ev.ydata - (ev.ydata - ylim[0]) * scale,
                    ev.ydata + (ylim[1] - ev.ydata) * scale,
                )
                _canvas.draw_idle()

            def on_key(
                ev,
                _ax=ax,
                _canvas=_win.canvas,
                _max_len=max_len,
                _sep=separation,
                _total_h=n_channels * separation,
                _nav=_nav,
                _grid_idx=grid_idx,
                _tm=_tm,
            ):
                if ev.key in ("r", "R"):
                    _ax.set_xlim(0, _max_len)
                    _ax.set_ylim(-_sep * 0.5, _total_h + _sep * 0.5)
                    _canvas.draw_idle()
                elif ev.key in ("z", "Z"):
                    if _nav.get("time_mask_mode", False) and _tm:
                        _tm.pop()
                        disconnect()
                        draw_grid(_grid_idx)
                elif ev.key in ("t", "T"):
                    disconnect()
                    _nav["time_mask_mode"] = not _nav.get("time_mask_mode", False)
                    draw_grid(_grid_idx)

            def go_prev(ev, _nav=_nav, _is_first=is_first):
                if not _is_first:
                    disconnect()
                    _nav["current"] -= 1
                    draw_grid(_nav["current"])

            def go_next(ev, _nav=_nav, _is_last=is_last, _loop=event_loop):
                disconnect()
                if not _is_last:
                    _nav["current"] += 1
                    draw_grid(_nav["current"])
                else:
                    QTimer.singleShot(50, _loop.quit)

            def on_confirm(ev, _loop=event_loop):
                disconnect()
                QTimer.singleShot(50, _loop.quit)

            def toggle_mask_mode(_ev, _nav=_nav, _grid_idx=grid_idx):
                disconnect()
                _nav["time_mask_mode"] = not _nav.get("time_mask_mode", False)
                draw_grid(_grid_idx)

            def on_span_select(
                xmin, xmax, _grid_idx=grid_idx, _tm=_tm, _step=step, _fs=_sampling_rate
            ):
                min_span = max(1, int(0.05 * _fs / _step))
                if xmax - xmin < min_span:
                    return
                start_s = round(xmin * _step / _fs, 3)
                end_s = round(xmax * _step / _fs, 3)
                _tm.append([start_s, end_s])
                disconnect()
                draw_grid(_grid_idx)

            span_sel = SpanSelector(
                ax,
                on_span_select,
                "horizontal",
                useblit=False,
                props=dict(alpha=0.20, facecolor=COLORS["error"]),
                button=1,
            )
            span_sel.set_active(is_mask_mode)

            cids = [
                _win.canvas.mpl_connect("button_press_event", on_press),
                _win.canvas.mpl_connect("button_release_event", on_release),
                _win.canvas.mpl_connect("motion_notify_event", on_motion),
                _win.canvas.mpl_connect("scroll_event", on_scroll),
                _win.canvas.mpl_connect("key_press_event", on_key),
            ]
            prev_btn.on_clicked(go_prev)
            next_btn.on_clicked(go_next)
            confirm_btn.on_clicked(on_confirm)
            mask_btn.on_clicked(toggle_mask_mode)

            _nav["cids"] = cids
            _nav["buttons"] = [prev_btn, mask_btn, next_btn, confirm_btn]
            if check is not None:
                _nav["buttons"].append(check)
            _nav["span_selector"] = span_sel
            _win.canvas.draw()

        def disconnect(_nav=nav, _win=win):
            for cid in _nav.get("cids", []):
                _win.canvas.mpl_disconnect(cid)
            for btn in _nav.get("buttons", []):
                btn.disconnect_events()
                try:
                    btn.ax.remove()
                except Exception:
                    pass
            _nav["buttons"] = []
            span = _nav.get("span_selector")
            if span is not None:
                span.set_active(False)
                _nav["span_selector"] = None

        win._current_loop = event_loop
        draw_grid(0)
        event_loop.exec_()
        win._current_loop = None

        if win.aborted:
            print("  Window closed — stopping.")
            return rejections

        # Save: convert 3-state → flat 0/1 mask (state != ACTIVE)
        rejections[fname] = {
            port_name: {
                "channels": (state_masks[i] != _ACTIVE).astype(int).tolist(),
                "time_masks": time_masks_list[i],
            }
            for i, (port_name, _) in enumerate(grid_list)
        }
        _save_json(output_path, rejections)

        n_rej = sum(int(np.sum(sm != _ACTIVE)) for sm in state_masks)
        n_tmask = sum(len(tm) for tm in time_masks_list)
        msg = f"  [{fname}] saved — {n_rej} channel(s) rejected"
        if n_tmask:
            msg += f", {n_tmask} time region(s) masked"
        msg += f" across {len(grid_list)} grid(s)"
        print(msg)

    win.close()
    return rejections


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _setup_from_channel_config(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)

    sampling_rate = data.get("sampling_rate", 10240)

    grid_configs: Dict[str, dict] = {}
    for g in data.get("grids", []):
        channels = list(range(g["start_chan"], g["end_chan"]))
        grid_configs[g["name"]] = {
            "channels": channels,
            "num_channels": len(channels),
            "electrode_type": g.get("config", ""),
            "electrode_class": (
                "surface_grid"
                if g.get("type", "").lower() == "surface"
                else "intramuscular"
            ),
        }

    aux_configs = data.get("aux_channels", [])

    from scd_app.io.data_loader import load_layout

    loader_map = {
        ".otb+": "loader_otb+.yaml",
        ".h5": "loader_h5.yaml",
        ".hdf5": "loader_h5.yaml",
        ".mat": "loader_mat.yaml",
    }
    loader_key = data.get("loader", ".otb+")
    loader_file = loader_map.get(loader_key, "loader_otb+.yaml")
    _here = Path(__file__).resolve().parent.parent
    loader_path = (
        _here / "src" / "scd_app" / "resources" / "loaders_configs" / loader_file
    )
    if not loader_path.exists():
        import importlib.resources as pkg_res

        loader_path = Path(
            str(
                pkg_res.files("scd_app") / "resources" / "loaders_configs" / loader_file
            )
        )
    layout = load_layout(loader_path)

    return grid_configs, layout, sampling_rate, aux_configs


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Fast batch channel inspection with inherited pre-rejections. "
        "Saves per-file rejection masks to a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--channel-config",
        default=None,
        metavar="JSON",
        help="Path to the app's channel_config.json.",
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--layout", default=None)
    ap.add_argument(
        "--files",
        required=False,
        default=None,
        nargs="+",
        help="EMG file path(s), directory, or glob pattern.",
    )
    ap.add_argument(
        "--files-yaml",
        default=None,
        metavar="YAML",
        help="Path to a file-list YAML.",
    )
    ap.add_argument(
        "--output",
        default="channel_rejections.json",
        help="Output JSON path (default: channel_rejections.json).",
    )
    ap.add_argument(
        "--ext",
        default=None,
        help="File extension when --files is a directory (e.g. .h5, .mat, .otb+).",
    )
    args = ap.parse_args()

    if args.files_yaml:
        import yaml

        with open(args.files_yaml) as _fh:
            _fl = yaml.safe_load(_fh)
        _data_dir = Path(_fl["data_dir"])
        if not args.channel_config and _fl.get("channel_config"):
            args.channel_config = _fl["channel_config"]
        if args.output == "channel_rejections.json" and _fl.get("output_rejections"):
            args.output = _fl["output_rejections"]
        _seen: dict = {}
        for _group in _fl.get("decompositions", []):
            for _fname in _group.get("files", []):
                _p = str(_data_dir / _fname)
                if _p not in _seen:
                    _seen[_p] = None
        args.files = list(_seen.keys())

    if not args.files:
        ap.error("Provide either --files or --files-yaml.")

    if args.channel_config:
        grid_configs, layout, sampling_rate, aux_configs = _setup_from_channel_config(
            Path(args.channel_config)
        )
        print(f"Channel config : {args.channel_config}")
        print(f"Fs             : {sampling_rate} Hz")
        print(f"Grids          : {list(grid_configs.keys())}")
        print(f"Aux channels   : {len(aux_configs)}")
    elif args.config and args.layout:
        from scd_app.core.config import ConfigManager
        from scd_app.io.data_loader import load_layout

        mgr = ConfigManager()
        config = mgr.load_session(Path(args.config))
        layout = load_layout(Path(args.layout))
        sampling_rate = config.sampling_frequency
        aux_configs = []
        print(f"Session : {config.name}  |  Fs: {sampling_rate} Hz")
        grid_configs: Dict[str, dict] = {}
        for port in config.ports:
            if not port.enabled:
                continue
            grid_configs[port.name] = {
                "channels": port.electrode.channels,
                "num_channels": len(port.electrode.channels),
                "electrode_type": port.electrode.name,
            }
    else:
        ap.error("Provide either --channel-config OR both --config and --layout.")

    _fmt_ext = {
        "h5": ".h5",
        "mat": ".mat",
        "npy": ".npy",
        "otb": ".otb+",
    }
    default_ext = _fmt_ext.get(layout.get("format", ""), ".h5")
    search_ext = args.ext if args.ext else default_ext

    file_paths: List[Path] = []
    for pat in args.files:
        p = Path(pat)
        if p.is_dir():
            found = sorted(p.glob(f"*{search_ext}"))
            if not found:
                print(f"  Warning: no *{search_ext} files in {p}")
            file_paths.extend(found)
        elif "*" in pat or "?" in pat:
            file_paths.extend(sorted(Path(".").glob(pat)))
        else:
            file_paths.append(p)
    file_paths = sorted(set(file_paths))

    if not file_paths:
        print("No files matched — exiting.")
        sys.exit(1)

    print(f"\nFiles to inspect ({len(file_paths)}):")
    for p in file_paths:
        print(f"  {p}")

    output_path = Path(args.output)

    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(
            f"\nLoaded existing rejections from {output_path} "
            f"({len(existing)} file(s) already done)."
        )

    already_done = [p for p in file_paths if p.name in existing]
    if already_done:
        print(f"\n{len(already_done)} file(s) already inspected:")
        for p in already_done:
            print(f"  {p.name}")
        ans = input("Re-inspect these files? [y/N]: ").strip().lower()
        if ans != "y":
            file_paths = [p for p in file_paths if p.name not in existing]
            if not file_paths:
                print("All files already inspected. Done.")
                sys.exit(0)
            print(f"Skipping {len(already_done)} file(s).")

    print(f"\nStarting inspection for {len(file_paths)} file(s)...\n")
    rejections = _run_channel_check_gui(
        file_paths=file_paths,
        layout=layout,
        grid_configs=grid_configs,
        output_path=output_path,
        existing_rejections=existing,
        sampling_rate=sampling_rate,
        aux_configs=aux_configs,
    )

    print(f"\nAll done. Rejections saved to: {output_path}")

    # Union of rejected channels per grid across all files
    union: Dict[str, set] = {}
    for grids in rejections.values():
        for grid_name, entry in grids.items():
            ch_mask = entry["channels"] if isinstance(entry, dict) else entry
            union.setdefault(grid_name, set()).update(
                i for i, v in enumerate(ch_mask) if v
            )

    print("\nRejected channels (union across all files):")
    for grid_name, indices in union.items():
        if indices:
            print(f"  {grid_name}: {', '.join(str(i) for i in sorted(indices))}")
        else:
            print(f"  {grid_name}: none rejected")


if __name__ == "__main__":
    _src = Path(__file__).resolve().parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    main()
