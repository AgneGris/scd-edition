# SCD Edition 🔧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A PyQt5 GUI application for editing and visualising EMG decomposition results from the [swarm-contrastive-decomposition](https://github.com/AgneGris/swarm-contrastive-decomposition) package.

SCD Edition sits downstream of the decomposition pipeline. After running SCD to extract motor units from high-density surface/intramuscular EMG, this editor lets you visually inspect each unit's source signal, spike-triggered average, and firing behaviour — then manually correct mistakes before exporting clean data.

## Features ✨

### Configuration & Data Loading
Configure decomposition sessions by selecting the data format, sampling rate, and input/output directories. Supported file formats include `.mat`, `.npy`, `.csv`, `.h5`, and `.otb+`. Load a single file or queue multiple files for batch processing. Define the channel layout by adding electrode grids (surface or intramuscular, with pre-defined electrode configurations) and auxiliary channels visually on a channel allocation bar. Configurations can be saved to and loaded from JSON files.

### Full-Length Source Recovery
When loading a decomposition file into the Edition tab, the application automatically integrates the saved **peel-off sequence**. Instead of limiting your view to the original decomposition time window (the plateau), the editor re-preprocesses the *entire* raw EMG signal. It then sequentially replays the peel-off of each motor unit and applies the saved spatial filters. This fully reconstructs the source signals and detects spike timestamps across the entire length of the recording, giving you a complete view of the unit's firing behaviour.

### Decomposition
Run the Swarm Contrastive Decomposition algorithm directly from the GUI. Features include:
- **Global Parameters:** Adjust SIL threshold, iterations, MUAP window, clamping, fitness metric (CoV/SIL), peel-off, and swarm mode.
- **Per-Grid Parameters:** Set extension factor, high-pass, low-pass, and notch filter (50/60 Hz with optional harmonics) independently per grid.
- **Batch Processing:** Queue multiple EMG files for sequential decomposition. Choose whether channel rejection is performed per file or once on the first file and shared across the batch.
- **Manual Channel Rejection:** Interactively select and mask noisy channels before decomposition. Channels are displayed at full resolution with a time axis at the bottom. Click a channel to toggle rejection (rejected channels are shown as faint dashed lines). Scroll to zoom the time axis, Shift+Scroll to zoom the channel amplitude axis, right-drag to pan, and R to reset the view.
- **Time Window Selection:** Choose specific time segments (plateaus) for decomposition to reduce processing time and focus on steady-state contractions. Set start/end times manually or by clicking on the signal plot.
- **Real-Time Visualisation:** Watch the decomposition progress as sources and timestamps are found at each iteration. Stop the decomposition at any time.

### Spike Editing
Click directly on the source signal to add or remove spikes. Every edit shows a **real-time preview** on the MUAP plot before you commit: the candidate spike's waveform is overlaid on the existing spike-triggered average across all channels so you can judge whether it belongs to that motor unit. Press Enter to confirm or Escape to cancel. All edits are stored in an undo stack (`Ctrl+Z`, up to 100 actions).

- **Add mode (A)**: click near a peak and the editor snaps to the nearest local maximum within the visible amplitude range, avoiding locations where a spike already exists
- **Delete mode (D)**: click near an existing spike and the editor selects the closest one for removal
- **Add in Selection mode (Ctrl+A)**: drag a region on the signal to add all detected peaks within it. Click the button again or press `Esc` to exit
- **Delete in Selection mode (Ctrl+D)**: drag a region on the signal to delete all spikes within it. Click the button again or press `Esc` to exit
- **View mode (V)**: default navigation mode; you can still quick-edit with `Ctrl+Click` (add) and `Alt+Click` (delete)

### Scrolling/Zoom options
There are three options for scrolling or zooming around the visualised spikes:
- **Scroll only**: This will zoom in and out of the plotting window in ONLY the x-direction
- **Shift + scroll**: This will scroll horizontally (along the x-dimension)
- **Ctrl + scroll**: This will zoom in and out of the plotting window in both x- and y-directions

### Visualisation
Three synchronised plots update whenever you switch unit or edit spikes:

- **Source signal** — the spatial filter output with spike locations marked as circles. The x-axis is shared with the discharge rate plot so zooming one zooms both. The plateau region used for decomposition is highlighted.
- **MUAP (Motor Unit Action Potentials)** — spike-triggered average for every EMG channel, displayed either stacked or in a grid layout matching the electrode's physical geometry.
- **Instantaneous discharge rate** — inter-spike intervals converted to firing rate (pps).

### Quality Metrics
A properties panel displays metrics for the currently selected motor unit, updated live after every spike edit:

- **Firing properties:** spike count, mean discharge rate (Hz), CoV of inter-spike intervals (%), minimum ISI (ms)
- **Quality scores:** SIL (Silhouette index), PNR (Peak-to-Noise Ratio, dB)
- **MUAP features:** max peak-to-peak amplitude (µV), max waveform length, peak and median frequency (Hz)
- **Reliability badge:** shown as RELIABLE or UNRELIABLE based on quality thresholds
- **Duplicate warning:** automatically flags possible duplicate motor units with similarity percentages

### Filter Recalculation
After substantial manual edits to a unit's spikes, the original spatial filter may no longer be optimal. **Recalculate Filters** uses your edited spike train to compute a new Spike-Triggered Average (STA) filter. Crucially, this process is fully integrated with the **peel-off sequence**:
1. It replays the peel-off of all previously extracted units up to the target unit.
2. It computes the new filter using the isolated, residual EMG and your manually corrected timestamps.
3. It re-applies this optimized filter to extract an updated, full-length source signal and a newly thresholded set of timestamps.

*(Note: Recalculation requires at least 2 spikes within the plateau region).*

### Unit Management
- **Flag Unit (X)** — toggles a visual flag on the current unit, marking it for later removal. Detected duplicates are automatically flagged.

### Export
- **Save Edited Data (Ctrl+S)** — writes the full edited state (sources, timestamps, metadata, recalculated filters, and properties) to a `.pkl` file.

## 🚧 Under Development

The following areas are not yet fully implemented or are planned for future releases:

- **Multi-file comparison** — no way to load two decompositions side by side for the same recording
- **Automated quality control** — no automatic flagging of units with low SIL, irregular discharge rates, or physiologically implausible firing patterns beyond the basic reliable flag.

## Installation 🛠️

```bash
pip install git+https://github.com/AgneGris/scd-edition.git
```

All dependencies are installed automatically.

### From Source (development)

```bash
git clone https://github.com/AgneGris/scd-edition
cd scd-edition
pip install -e .
```

### Using uv 
For Windows/Linux, CUDA-enabled PyTorch build is recommended. Otherwise use --extra cpu
```bash
uv sync --extra cuda
.venv\Scripts\Activate.ps1
```
For macOS 
```bash
uv sync --extra cpu
.venv\Scripts\Activate.ps1
```

## Usage 🚀

### Launch the GUI

```bash
# Using the entry point
scd-edition

# Or run directly
python -m scd_app.gui.main_window
```

### Typical Workflow

1. **Configuration:** Use Tab 1 to define the data format, sampling rate, input files, and electrode grids. Apply the configuration.
2. **Decomposition:** In Tab 2, set parameters, select channels/time windows, and run the decomposition.
3. **Edition:** The results automatically load into Tab 3. Browse units using the port/unit dropdowns or arrow keys, edit spikes using modes/selection windows, monitor quality metrics, recalculate filters if needed, and flag units for deletion.
4. **Save:** Save the edited decomposition as a `.pkl` file.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+1` | Switch to Configuration tab |
| `Ctrl+2` | Switch to Decomposition tab |
| `Ctrl+3` | Switch to Edition tab |
| `V` | View mode |
| `A` | Add mode |
| `Ctrl+A` | Add in selection mode |
| `D` | Delete mode |
| `Ctrl+D` | Delete in selection mode |
| `R` | Toggle ROI |
| `Shift+A` | Add spikes in ROI |
| `Shift+D` | Delete spikes in ROI |
| `F` | Recalculate Filter |
| `X` | Flag unit |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Save |
| `Up/Down` | Next/Previous MU |
| `Home` | Reset View |
| `Scroll` | Zoom X-Dimension |
| `Shift+Scroll` | Scroll horizontally |
| `Ctrl+Scroll` | Zoom |

## Citation

If you use this software, please cite:

```bibtex
@article{grison2024particle,
  title={A particle swarm optimised independence estimator for blind source separation of neurophysiological time series},
  author={Grison, Agnese and Clarke, Alexander Kenneth and Muceli, Silvia and Ib{\'a}{\~n}ez, Jaime and Kundu, Aritra and Farina, Dario},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024},
  publisher={IEEE}
}

@article{grison2025unlocking,
  title={Unlocking the full potential of high-density surface EMG: novel non-invasive high-yield motor unit decomposition},
  author={Grison, Agnese and Mendez Guerra, Irene and Clarke, Alexander Kenneth and Muceli, Silvia and Ib{\'a}{\~n}ez, Jaime and Farina, Dario},
  journal={The Journal of Physiology},
  volume={603},
  number={8},
  pages={2281--2300},
  year={2025},
  publisher={Wiley Online Library}
}
```

## Contact

**Agnese Grison**
📧 agnese.grison@outlook.it
