"""Capture the real Qt interface as the animated README demonstration.

The decomposition-progress frame uses deterministic illustrative sources so
the capture does not run a costly scientific decomposition. Edition and
Visualisation are populated through the same application data path used for a
saved session, with the packaged EMG recording supplying the displayed MUAPs.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from scd_app.core.mu_model import EditMode
from scd_app.examples import bundled_example_config
from scd_app.gui.main_window import MainWindow, _configure_example_on_startup
from scd_app.gui.style.styling import set_style_sheet

WIDTH = 1280
HEIGHT = 720
ACCENT = "#4a9eff"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/scd_app/gui/style/fonts/Figtree-SemiBold.ttf"
    )
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _grab(window: MainWindow, step: str, caption: str) -> Image.Image:
    QApplication.processEvents()
    QTest.qWait(120)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    window.grab().save(buffer, "PNG")
    buffer.close()

    image = Image.open(BytesIO(bytes(data))).convert("RGBA")
    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (28, HEIGHT - 92, WIDTH - 28, HEIGHT - 24)
    draw.rounded_rectangle(box, radius=14, fill=(12, 17, 24, 238))
    draw.rounded_rectangle((42, HEIGHT - 78, 132, HEIGHT - 38), radius=10, fill=ACCENT)
    step_font = _font(19)
    caption_font = _font(25)
    draw.text((87, HEIGHT - 58), step, font=step_font, fill="white", anchor="mm")
    draw.text((154, HEIGHT - 58), caption, font=caption_font, fill="white", anchor="lm")
    return Image.alpha_composite(image, overlay).convert("RGB")


def _illustrative_sources(
    n_samples: int, sampling_rate: int, units: int = 8
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    rng = np.random.default_rng(17)
    all_sources = []
    all_timestamps = []
    missed_peak = int(1.84 * sampling_rate)

    for unit in range(units):
        source = rng.normal(0.0, 0.035, n_samples).astype(np.float32)
        rate = 8.5 + unit * 0.75
        start = int((0.22 + unit * 0.055) * sampling_rate)
        interval = max(1, int(sampling_rate / rate))
        timestamps = np.arange(start, n_samples - 300, interval, dtype=np.int64)
        width = np.arange(-18, 19)
        pulse = (1.2 + unit * 0.035) * np.exp(-0.5 * (width / 5.0) ** 2)
        for timestamp in timestamps:
            left = timestamp - 18
            right = timestamp + 19
            if left >= 0 and right <= n_samples:
                source[left:right] += pulse
        if unit == 0:
            source[missed_peak - 18 : missed_peak + 19] += 1.32 * np.exp(
                -0.5 * (width / 5.0) ** 2
            )
        all_sources.append(source)
        all_timestamps.append(timestamps)
    return all_sources, all_timestamps, missed_peak


def _demo_session(window: MainWindow) -> tuple[dict, int]:
    config = bundled_example_config()
    sampling_rate = int(config["sampling_rate"])
    n_samples = sampling_rate * 4
    emg = window.decomp_tab.emg_data[:n_samples].detach().cpu().numpy().T
    sources, timestamps, missed_peak = _illustrative_sources(n_samples, sampling_rate)
    return (
        {
            "ports": ["Example_Grid"],
            "sampling_rate": sampling_rate,
            "discharge_times": [timestamps],
            "pulse_trains": [sources],
            "mu_filters": [None],
            "skip_filter_recalc": True,
            "plateau_coords": [0, n_samples],
            "data": emg,
            "chans_per_electrode": [64],
            "channel_indices": [np.arange(64)],
            "emg_mask": [np.zeros(64, dtype=np.int8)],
            "electrodes": ["GR10MM0808"],
        },
        missed_peak,
    )


def _with_transitions(stills: list[Image.Image]) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    for index, still in enumerate(stills):
        frames.append(still)
        durations.append(1450 if index < len(stills) - 1 else 2100)
        if index == len(stills) - 1:
            continue
        for blend in (0.25, 0.5, 0.75):
            frames.append(Image.blend(still, stills[index + 1], blend))
            durations.append(90)
    return frames, durations


def capture(output: Path) -> None:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("SCD-Edition demo capture")
    set_style_sheet(app)
    window = MainWindow()
    window.resize(WIDTH, HEIGHT)
    window.show()

    _configure_example_on_startup(window)
    stills = [_grab(window, "01", "Open the configured 64-channel example")]

    window.config_tab._apply_config()
    QApplication.processEvents()
    sources, timestamps, _ = _illustrative_sources(26000, 10240, units=1)
    window.decomp_tab._plot_source_realtime(
        sources[0], timestamps[0], iteration=18, silhouette=0.934
    )
    window.tabs.setCurrentWidget(window.decomp_tab)
    stills.append(_grab(window, "02", "Decompose with live source feedback"))

    session, missed_peak = _demo_session(window)
    window.edition_tab._loaded_path = Path("bundled_example_decomposition.pkl")
    window.edition_tab._load_decomposition_data(session)
    window.edition_tab._update_file_label()
    window.edition_tab.file_loaded.emit()
    window.tabs.setCurrentWidget(window.edition_tab)
    stills.append(_grab(window, "03", "Review every motor unit and its MUAP"))

    window.edition_tab._set_mode(EditMode.ADD)
    window.edition_tab._handle_add_click(missed_peak)
    stills.append(
        _grab(window, "04", "Add a missed spike and update quality instantly")
    )

    window.tabs.setCurrentWidget(window.vis_tab)
    window.vis_tab._inner_tabs.setCurrentIndex(0)
    window.vis_tab.on_tab_activated()
    stills.append(_grab(window, "05", "Inspect population-level discharge behaviour"))

    frames, durations = _with_transitions(stills)
    paletted = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    window.edition_tab._undo_stack.clear()
    window.close()
    print(f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/demo.gif"), help="output GIF path"
    )
    args = parser.parse_args()
    capture(args.output)


if __name__ == "__main__":
    main()
