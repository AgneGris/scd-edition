# SCD Edition 🔧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A PyQt5 GUI application for editing and visualising EMG decomposition results from the [swarm-contrastive-decomposition](https://github.com/AgneGris/swarm-contrastive-decomposition) package.

SCD Edition sits downstream of the decomposition pipeline. After running SCD to extract motor units from high-density surface/intramuscular EMG, this editor lets you visually inspect each unit's source signal, spike-triggered average, and firing behaviour — then manually correct mistakes before exporting clean data.

## Features ✨

### Configuration & Data Loading
Configure decomposition sessions by selecting the data format, sampling rate, and input/output directories. Define the channel layout by adding electrode grids (e.g., surface, intramuscular) and auxiliary channels (e.g., force, torque) visually on a channel allocation bar. Configurations can be saved to and loaded from JSON files.

### Full-Length Source Recovery
When loading a decomposition file into the Edition tab, the application automatically integrates the saved **peel-off sequence**. Instead of limiting your view to the original decomposition time window (the plateau), the editor re-preprocesses the *entire* raw EMG signal. It then sequentially replays the peel-off of each motor unit and applies the saved spatial filters. This fully reconstructs the source signals and detects spike timestamps across the entire length of the recording, giving you a complete view of the unit's firing behaviour.

### Decomposition
Run the Swarm Contrastive Decomposition algorithm directly from the GUI. Features include:
- **Global & Per-Grid Parameters:** Adjust SIL threshold, iterations, and filter settings (low-pass, high-pass, notch) per grid.
- **Batch Processing:** Queue multiple EMG files for sequential decomposition.
- **Manual Channel Rejection:** Interactively select and mask noisy channels before decomposition.
- **Time Window Selection:** Choose specific time segments (plateaus) for decomposition to reduce processing time and focus on steady-state contractions.
- **Real-Time Visualisation:** Watch the decomposition progress as sources and timestamps are found at each iteration.

### Spike Editing
Click directly on the source signal to add or remove spikes. Every edit shows a **real-time preview** on the MUAP plot before you commit: the candidate spike's waveform is overlaid on the existing spike-triggered average across all channels so you can judge whether it belongs to that motor unit. Press Enter to confirm or Escape to cancel. All edits are stored in an undo stack (`Ctrl+Z`).

- **Add mode (A)**: click near a peak and the editor snaps to the nearest local maximum within the visible amplitude range, avoiding locations where a spike already exists
- **Delete mode (D)**: click near an existing spike and the editor selects the closest one for removal
- **View mode (V)**: default navigation mode; you can still quick-edit with `Ctrl+Click` (add) and `Alt+Click` (delete)

### ROI Selection
Toggle a draggable/resizable rectangle on the source plot (`R`). Once positioned, **ROI Add** finds all peaks inside the box and adds them as spikes, while **ROI Delete** removes any existing spikes that fall within the region. Useful for bulk-correcting a noisy segment or filling in a missed burst.

### Visualisation
Three synchronised plots update whenever you switch unit or edit spikes:

- **Source signal** — the spatial filter output with spike locations marked as circles. The x-axis is shared with the discharge rate plot so zooming one zooms both. The plateau region used for decomposition is highlighted.
- **MUAP (Motor Unit Action Potentials)** — spike-triggered average for every EMG channel, displayed either stacked or in a grid layout matching the electrode's physical geometry.
- **Instantaneous discharge rate** — inter-spike intervals converted to firing rate (pps).

### Quality Metrics
Quality metrics, including Discharge Rate, Coefficient of Variation (CoV), Silhouette (SIL) score, and Pulse-to-Noise Ratio (PNR), are calculated using `motor_unit_toolbox`. These metrics automatically update when spike trains are edited, providing immediate feedback on unit reliability.

### Filter Recalculation
After substantial manual edits to a unit's spikes, the original spatial filter may no longer be optimal. **Recalculate Filters** uses your edited spike train to compute a new Spike-Triggered Average (STA) filter. Crucially, this process is fully integrated with the **peel-off sequence**:
1. It replays the peel-off of all previously extracted units up to the target unit.
2. It computes the new filter using the isolated, residual EMG and your manually corrected timestamps. 
3. It re-applies this optimized filter to extract an updated, full-length source signal and a newly thresholded set of timestamps. 

*(Note: Recalculation requires at least 2 spikes within the plateau region).*

### Unit Management
- **Flag Unit** — toggles a visual flag on the current unit, marking it for later removal. Detected duplicates are automatically flagged.

### Export
- **Save Edited Data** — writes the full edited state (sources, timestamps, metadata, recalculated filters, and properties) to a `.pkl` file.

## 🚧 Under Development

The following areas are not yet fully implemented or are planned for future releases:

- **Multi-file comparison** — no way to load two decompositions side by side for the same recording
- **Automated quality control** — no automatic flagging of units with low SIL, irregular discharge rates, or physiologically implausible firing patterns beyond the basic reliable flag.

## Installation 🛠️

### From Source

```bash
git clone [https://github.com/AgneGris/scd-edition](https://github.com/AgneGris/scd-edition)
cd scd-edition
pip install -e .
```

### Dependencies

The package requires:
- `swarm-contrastive-decomposition`
- `motor_unit_toolbox` (install via `pip install git+https://github.com/imendezguerra/motor_unit_toolbox.git`)
- `PyQt5`
- `pyqtgraph`
- `numpy`, `scipy`, `scikit-learn`
- `torch`, `mat73`, `matplotlib`

## Usage 🚀

### Launch the GUI

```bash
# Using the entry point
scd-edition

# Or run directly
python -m scd_edition.app
```

### Typical Workflow

1. **Configuration:** Use Tab 1 to define the data format, sampling rate, input files, and electrode grids. Apply the configuration.
2. **Decomposition:** In Tab 2, set parameters, select channels/time windows, and run the decomposition.
3. **Edition:** The results automatically load into Tab 3. Browse units, edit spikes using modes/ROI, monitor quality metrics, recalculate filters if needed, and flag units for deletion.
4. **Save:** Save the edited decomposition as a `.pkl` file.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | View mode |
| `A` | Add mode |
| `D` | Delete mode |
| `R` | Toggle ROI |
| `Shift+A` | Add spikes in ROI |
| `Shift+D` | Delete spikes in ROI |
| `F` | Recalculate Filter |
| `X` | Flag unit |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Up/Down` | Next/Previous MU |
| `Home` | Reset View |

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
📧 agnese.grison@outlook.it -->