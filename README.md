# SCD Edition 🔧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A PyQt5 GUI application for editing and visualising EMG decomposition results from the [swarm-contrastive-decomposition](https://github.com/AgneGris/swarm-contrastive-decomposition) package.

SCD Edition sits downstream of the decomposition pipeline. After running SCD to extract motor units from high-density surface/intramuscular EMG, this editor lets you visually inspect each unit's source signal, spike-triggered average, and firing behaviour — then manually correct mistakes before exporting clean data.

## Features ✨

### Spike Editing
Click directly on the source signal to add or remove spikes. Every edit shows a **real-time preview** on the MUAP plot before you commit: the candidate spike's waveform is overlaid on the existing spike-triggered average across all channels so you can judge whether it belongs to that motor unit. Press Enter to confirm or Escape to cancel. All edits are stored in an undo stack (`Ctrl+Z`).

- **Add mode (A)**: click near a peak and the editor snaps to the nearest local maximum within the visible amplitude range, avoiding locations where a spike already exists
- **Delete mode (D)**: click near an existing spike and the editor selects the closest one for removal
- **View mode (V)**: default navigation mode; you can still quick-edit with `Ctrl+Click` (add) and `Alt+Click` (delete)

### ROI Selection
Toggle a draggable/resizable rectangle on the source plot (`R`). Once positioned, **ROI Add** finds all peaks inside the box and adds them as spikes, while **ROI Delete** removes any existing spikes that fall within the region. Useful for bulk-correcting a noisy segment or filling in a missed burst.

### Visualisation
Three synchronised plots update whenever you switch unit or edit spikes:

- **Source signal** — the spatial filter output with spike locations marked as circles. The x-axis is shared with the discharge rate plot so zooming one zooms both.
- **MUAP (Motor Unit Action Potentials)** — spike-triggered average (±20 ms window) for every EMG channel, stacked vertically with the highest-amplitude channel highlighted. During previews, the candidate spike's single-trial waveform is overlaid in green (add) or red (delete).
- **Instantaneous discharge rate** — inter-spike intervals converted to firing rate (pps) with mean, median, and coefficient of variation displayed in the title. A dashed line marks the median rate.

### Quality Metrics
The **SIL (Silhouette)** score is recalculated every time the spike train changes and displayed next to the unit selector. This gives immediate feedback on whether your edits are improving or degrading separation quality.

### Duplicate Detection
Compares all pairs of motor units to find duplicates (units that share a high proportion of spike times). Detected duplicates are automatically **flagged** — shown in red in the unit dropdown — so you can review them before deletion. Uses a configurable time tolerance based on the sampling frequency.

### Filter Recalculation
After substantial manual edits, the original spatial filter may no longer be optimal. **Recalculate Filters** recomputes the separation vector from the whitened EMG using the current spike train, then re-applies it and re-thresholds to produce updated source signals and timestamps. Requires at least 5 spikes.

### Unit Management
- **Flag Unit** — toggles a visual flag (red highlight) on the current unit, marking it for later removal
- **Delete Flagged** — permanently removes all flagged units from the dataset and re-indexes the remaining ones

### Export
- **Save Edited Data** — writes the full edited state (sources, timestamps, metadata) to a `.pkl` file
- **Convert to EMGlab (.eaf)** — exports spike trains and filtered EMG to EMGlab format for use with external analysis tools

## 🚧 Under Development

The following areas are not yet fully implemented or are planned for future releases:

- **Batch processing** — currently limited to one file at a time; no queue or scripted batch mode
- **Session persistence** — edit history is lost when you close the application; there is no auto-save or session recovery
- **Configurable duplicate detection parameters** — thresholds are set internally in `DuplicateDetector` and cannot yet be adjusted from the GUI
- **Multi-file comparison** — no way to load two decompositions side by side for the same recording
- **Automated quality control** — no automatic flagging of units with low SIL, irregular discharge rates, or physiologically implausible firing patterns

## Installation 🛠️

### From Source

```bash
git clone https://github.com/AgneGris/scd-edition
cd scd-edition
pip install -e .
```

### Dependencies

The package automatically installs:
- `swarm-contrastive-decomposition` (SCD package)
- `PyQt5`
- `pyqtgraph`
- `numpy`, `scipy`, `scikit-learn`
- `torch`, `mat73`

## Usage 🚀

### Launch the GUI

```bash
# Using the entry point
scd-edition

# Or run directly
python -m scd_edition.app
```

### Typical Workflow

```python
import scd

# 1. Decompose
dictionary, timestamps = scd.train("emg.mat", config_name="surface")
scd.save_results("decomposition.pkl", dictionary)

# 2. Open the editor for manual review
#    scd-edition → Load decomposition.pkl + emg.mat
```

Then inside the GUI:

1. **Load Data** — select the decomposition `.pkl` and the original EMG `.mat` or `.npy`
2. **Browse units** — step through each motor unit with the dropdown
3. **Edit spikes** — switch mode (`V`/`A`/`D`) or use modifier clicks; preview appears on the MUAP plot, confirm with Enter
4. **Bulk edits** — toggle ROI (`R`), position the box, then ROI Add or ROI Delete
5. **Quality control** — check SIL, remove duplicates, recalculate filters after major changes
6. **Save** — export as `.pkl` for further analysis or `.eaf` for EMGlab

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | View mode |
| `A` | Add mode |
| `D` | Delete mode |
| `R` | Toggle ROI |
| `Enter` | Confirm preview |
| `Escape` | Cancel preview |
| `Ctrl+Z` | Undo |
| `Ctrl+Click` | Quick-add (any mode) |
| `Alt+Click` | Quick-delete (any mode) |

## Project Structure 📁

```
scd-edition/
├── scd_edition/
│   ├── __init__.py
│   ├── app.py                 # Entry point
│   ├── core/
│   │   ├── data_handler.py    # Data loading, saving, and channel management
│   │   ├── spike_editor.py    # Spike add/delete, peak finding, undo stack, filter recalculation
│   │   └── analysis.py        # SIL calculation, discharge rate, duplicate detection
│   ├── gui/
│   │   ├── main_window.py     # Main window, plots, and all user interaction
│   │   └── styles.py          # Plot colours and theme constants
│   └── export/
│       └── emglab.py          # EMGlab .eaf format writer
└── pyproject.toml
```

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
