"""Capture the real Qt interface as the animated README demonstration.

The decomposition-progress frame uses deterministic illustrative sources so
the capture does not run a costly scientific decomposition. Edition and
Visualisation are populated through the same application data path used for a
saved session, with deterministic multichannel waveforms keeping the displayed
MUAP comparisons coherent and reproducible.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSplitter

from scd_app.core.filter_recalculation import snap_to_local_peak
from scd_app.core.mu_model import EditMode
from scd_app.examples import bundled_example_config
from scd_app.gui.main_window import MainWindow, _configure_example_on_startup
from scd_app.gui.style.styling import set_style_sheet

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 1000
GIF_WIDTH = 1120
GIF_HEIGHT = 800
STEP_DURATION_MS = 2400
FINAL_STEP_DURATION_MS = 3200
TRANSITION_DURATION_MS = 140
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


def _grab_window(window: MainWindow) -> Image.Image:
    """Capture a clean, consistently sized image of the application window."""
    QApplication.processEvents()
    QTest.qWait(120)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    window.grab().save(buffer, "PNG")
    buffer.close()

    image = Image.open(BytesIO(bytes(data))).convert("RGB")
    return image.resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.Resampling.LANCZOS)


def _add_caption(image: Image.Image, step: str, caption: str) -> Image.Image:
    """Add the animated-demo caption without changing the clean screenshot."""
    image = image.resize((GIF_WIDTH, GIF_HEIGHT), Image.Resampling.LANCZOS).convert(
        "RGBA"
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = (28, GIF_HEIGHT - 92, GIF_WIDTH - 28, GIF_HEIGHT - 24)
    draw.rounded_rectangle(box, radius=14, fill=(12, 17, 24, 238))
    draw.rounded_rectangle(
        (42, GIF_HEIGHT - 78, 132, GIF_HEIGHT - 38), radius=10, fill=ACCENT
    )
    step_font = _font(19)
    caption_font = _font(25)
    draw.text((87, GIF_HEIGHT - 58), step, font=step_font, fill="white", anchor="mm")
    draw.text(
        (154, GIF_HEIGHT - 58),
        caption,
        font=caption_font,
        fill="white",
        anchor="lm",
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def _save_screenshot(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True, compress_level=9)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", flush=True)


def _show_complete_muap_grid(window: MainWindow) -> None:
    """Give the resizable Edition sidebar enough width for its full 8x8 grid."""
    for splitter in window.edition_tab.findChildren(QSplitter):
        if splitter.orientation() == Qt.Orientation.Horizontal:
            splitter.setSizes([520, WINDOW_WIDTH - 520])
            return


def _illustrative_sources(
    n_samples: int, sampling_rate: int, units: int = 8
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    rng = np.random.default_rng(17)
    all_sources = []
    all_timestamps = []
    missed_peak_nominal = int(1.84 * sampling_rate)
    missed_peak = missed_peak_nominal

    for unit in range(units):
        source = rng.normal(0.0, 0.035, n_samples).astype(np.float32)
        rate = 8.5 + unit * 0.75
        start = int((0.22 + unit * 0.055) * sampling_rate)
        interval = max(1, int(sampling_rate / rate))
        nominal_timestamps = np.arange(start, n_samples - 300, interval, dtype=np.int64)
        width = np.arange(-18, 19)
        pulse = (1.2 + unit * 0.035) * np.exp(-0.5 * (width / 5.0) ** 2)
        for timestamp in nominal_timestamps:
            left = timestamp - 18
            right = timestamp + 19
            if left >= 0 and right <= n_samples:
                source[left:right] += pulse
        if unit == 0:
            missed_pulse = 1.32 * np.exp(-0.5 * (width / 5.0) ** 2)
            source[missed_peak_nominal - 18 : missed_peak_nominal + 19] += missed_pulse
            missed_peak = int(
                snap_to_local_peak(
                    source,
                    np.asarray([missed_peak_nominal]),
                    max_shift=6,
                    square_source=False,
                )[0]
            )
        timestamps = snap_to_local_peak(
            source,
            nominal_timestamps,
            max_shift=6,
            square_source=False,
        )
        all_sources.append(source)
        all_timestamps.append(timestamps)
    return all_sources, all_timestamps, missed_peak


def _illustrative_emg(
    all_timestamps: list[np.ndarray],
    missed_peak: int,
    n_samples: int,
    sampling_rate: int,
    n_channels: int = 64,
) -> np.ndarray:
    """Build stable, unit-specific multichannel MUAPs for documentation."""
    rng = np.random.default_rng(29)
    emg = rng.normal(0.0, 0.012, (n_channels, n_samples)).astype(np.float32)
    half_window = max(1, int(round(0.0125 * sampling_rate)))
    phase = np.linspace(-1.0, 1.0, 2 * half_window, endpoint=False)
    waveform = (
        -0.55 * np.exp(-(((phase + 0.24) / 0.16) ** 2))
        + np.exp(-((phase / 0.11) ** 2))
        - 0.42 * np.exp(-(((phase - 0.27) / 0.18) ** 2))
    )
    waveform /= np.max(np.abs(waveform))

    channel_rows, channel_cols = np.divmod(np.arange(n_channels), 8)
    centers = [(1, 1), (1, 4), (2, 6), (3, 2), (4, 5), (5, 1), (6, 4), (6, 6)]
    for unit, timestamps in enumerate(all_timestamps):
        center_row, center_col = centers[unit % len(centers)]
        distance_sq = (channel_rows - center_row) ** 2 + (
            channel_cols - center_col
        ) ** 2
        spatial = np.exp(-distance_sq / (2.0 * 1.35**2))
        spatial *= 0.75 + unit * 0.035
        events = timestamps
        if unit == 0:
            events = np.sort(np.append(events, missed_peak))
        for timestamp in events:
            left = int(timestamp) - half_window
            right = int(timestamp) + half_window
            if left < 0 or right > n_samples:
                continue
            amplitude = rng.normal(1.0, 0.035)
            emg[:, left:right] += (
                amplitude * spatial[:, np.newaxis] * waveform[np.newaxis, :]
            )
    return emg


def _demo_session() -> tuple[dict, int]:
    config = bundled_example_config()
    sampling_rate = int(config["sampling_rate"])
    n_samples = sampling_rate * 4
    sources, timestamps, missed_peak = _illustrative_sources(n_samples, sampling_rate)
    emg = _illustrative_emg(
        timestamps,
        missed_peak,
        n_samples=n_samples,
        sampling_rate=sampling_rate,
    )
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
        durations.append(
            STEP_DURATION_MS if index < len(stills) - 1 else FINAL_STEP_DURATION_MS
        )
        if index == len(stills) - 1:
            continue
        for blend in (0.25, 0.5, 0.75):
            frames.append(Image.blend(still, stills[index + 1], blend))
            durations.append(TRANSITION_DURATION_MS)
    return frames, durations


def capture(output: Path, screenshots_dir: Path) -> None:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("SCD-Edition demo capture")
    set_style_sheet(app)
    window = MainWindow()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.show()

    _configure_example_on_startup(window)
    window.config_tab.path_edit.setText("examples/emg.mat")
    window.config_tab.output_dir_edit.setText("scd-edition-output")
    window.tabs.setCurrentWidget(window.config_tab)
    config_image = _grab_window(window)
    _save_screenshot(config_image, screenshots_dir / "configuration.png")
    stills = [
        _add_caption(config_image, "01", "Open the configured 64-channel example")
    ]

    window.config_tab._apply_config()
    QApplication.processEvents()
    sources, timestamps, _ = _illustrative_sources(26000, 10240, units=1)
    window.decomp_tab._plot_source_realtime(
        sources[0], timestamps[0], iteration=18, silhouette=0.934
    )
    window.tabs.setCurrentWidget(window.decomp_tab)
    decomposition_image = _grab_window(window)
    _save_screenshot(decomposition_image, screenshots_dir / "decomposition.png")
    stills.append(
        _add_caption(decomposition_image, "02", "Decompose with live source feedback")
    )

    session, missed_peak = _demo_session()
    window.edition_tab._loaded_path = Path("bundled_example_decomposition.pkl")
    window.edition_tab._load_decomposition_data(session)
    window.edition_tab._update_file_label()
    window.edition_tab.file_loaded.emit()
    window.tabs.setCurrentWidget(window.edition_tab)
    _show_complete_muap_grid(window)
    edition_image = _grab_window(window)
    stills.append(
        _add_caption(edition_image, "03", "Review every motor unit and its MUAP")
    )

    inspection_sample = int(session["discharge_times"][0][0][4])
    window.edition_tab._inspect_spike_muap(inspection_sample)
    inspection_image = _grab_window(window)
    _save_screenshot(inspection_image, screenshots_dir / "edition.png")
    stills.append(
        _add_caption(
            inspection_image,
            "04",
            "Compare each discharge with or without earlier units",
        )
    )

    window.edition_tab._set_mode(EditMode.ADD)
    window.edition_tab._handle_add_click(missed_peak)
    stills.append(
        _add_caption(
            _grab_window(window),
            "05",
            "Add a missed spike and update quality instantly",
        )
    )

    window.tabs.setCurrentWidget(window.vis_tab)
    window.vis_tab._inner_tabs.setCurrentIndex(0)
    window.vis_tab.on_tab_activated()
    visualisation_image = _grab_window(window)
    _save_screenshot(visualisation_image, screenshots_dir / "visualisation.png")
    stills.append(
        _add_caption(
            visualisation_image,
            "06",
            "Inspect population-level discharge behaviour",
        )
    )

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
    window.edition_tab._set_dirty(False)
    window.close()
    print(
        f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/demo.gif"), help="output GIF path"
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("docs/screenshots"),
        help="directory for clean README screenshots",
    )
    args = parser.parse_args()
    capture(args.output, args.screenshots_dir)


if __name__ == "__main__":
    main()
