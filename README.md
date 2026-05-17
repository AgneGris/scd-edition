# SCD Edition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A graphical application for decomposing high-density surface or intramuscular EMG recordings into individual motor unit spike trains, editing them manually, and visualising population-level discharge behaviour.

Built on the [Swarm Contrastive Decomposition (SCD)](https://github.com/AgneGris/swarm-contrastive-decomposition) algorithm.

---

## Table of Contents

1. [Installation](#installation)
2. [Launching the app](#launching-the-app)
3. [Complete workflow](#complete-workflow)
   - [Tab 1 — Configuration](#tab-1--configuration)
   - [Tab 2 — Decomposition](#tab-2--decomposition)
   - [Tab 3 — Edition](#tab-3--edition)
   - [Tab 4 — Visualisation](#tab-4--visualisation)
4. [Saving and loading](#saving-and-loading)
5. [Keyboard shortcuts](#keyboard-shortcuts)
6. [File formats](#file-formats)
7. [Force channel setup](#force-channel-setup)
8. [Citation](#citation)

---

## Installation

### From PyPI (recommended)

```bash
pip install git+https://github.com/AgneGris/scd-edition.git
```

All dependencies install automatically.

### From source with uv (recommended for development)

[uv](https://github.com/astral-sh/uv) manages the virtual environment and dependencies automatically.

**Windows / Linux — CUDA-enabled (recommended if you have an NVIDIA GPU):**
```bash
uv sync --extra cuda
.venv\Scripts\Activate.ps1
```

**macOS or CPU-only:**
```bash
uv sync --extra cpu
source .venv/bin/activate
```

### From source with pip

```bash
git clone https://github.com/AgneGris/scd-edition
cd scd-edition
pip install -e .
```

**Requirements:** Python 3.10 or later. A CUDA-enabled GPU is recommended for fast decomposition but not required — a CPU fallback is available.

---

## Launching the app

```bash
scd-edition
```

Or, if the entry point is not on your PATH:

```bash
python -m scd_app.gui.main_window
```

The application opens with four tabs along the top. Work left to right: configure → decompose → edit → visualise.

---

## Complete workflow

### Tab 1 — Configuration

This tab is where you tell the application about your recording before doing anything else.

#### 1. Select your input file

Click **Select Input File** and choose your EMG recording. Supported formats:

| Extension | Format |
|-----------|--------|
| `.otb+`   | OTBiolab+ (Quattrocento amplifier) |
| `.mat`    | MATLAB |
| `.h5`     | HDF5 |
| `.npy`    | NumPy array |
| `.csv`    | Comma-separated values |

You can also queue **multiple files** for batch processing using **Add Files**. All files in the queue will be decomposed sequentially with the same configuration.

#### 2. Set the sampling rate

Enter the sampling rate of your recording in Hz (e.g. `2048` for a Quattrocento at 2 kHz).

#### 3. Add electrode grids

Click **+ Add Grid** for each electrode array in your recording. For each grid, specify:

- **Name** — a label you choose (e.g. `Biceps`, `FDI`)
- **Muscle** — optional, for your own reference
- **Type** — Surface or Intramuscular
- **Electrode model** — select from the dropdown (e.g. `GR10MM0808` for a 64-channel 8×8 surface grid)
- **Channel start / end** — which channels in the file correspond to this grid (0-indexed, end is exclusive). The colour bar at the top shows how channels are allocated across all grids.

#### 4. Add force / auxiliary channels (optional)

If your recording includes force or other analogue channels, click **+ Add Aux Channel** for each one. Specify:

- **Name** — e.g. `Middle Ext`
- **Unit label** — e.g. `Middle Ext` (used to auto-select the correct channel when visualising named tasks)
- **Source** — `Signal` if the force data is stored as regular channels in the EMG file; `Aux file` if it is stored in a separate `.sip` stream inside an OTB+ archive
- **Channel start / end** — channel indices within the file (or sip stream)
- **MVC (mV)** — the maximum voluntary contraction value **in millivolts**. This is used to normalise force to %MVC in the visualisation. See [Force channel setup](#force-channel-setup) for how to find this value.

#### 5. Save / load your configuration

Click **Save Config** to export the full configuration (file path, grids, aux channels, parameters) to a JSON file. Click **Load Config** to restore a previously saved configuration. The input file is not overwritten when loading a config — the file you selected in step 1 is preserved.

#### 6. Apply

Click **Apply Configuration**. This validates all settings and prepares the decomposition tab. If anything is wrong (e.g. channel indices out of range) an error will appear here.

---

### Tab 2 — Decomposition

#### Global parameters

These apply to all grids:

| Parameter | What it does |
|-----------|--------------|
| **SIL Threshold** | Minimum silhouette score for a source to be accepted as a motor unit. Higher = stricter (fewer but more reliable MUs). Default 0.9. |
| **Iterations** | Maximum number of optimisation steps per source. More iterations → longer runtime but potentially more MUs found. |
| **MUAP Window (ms)** | Duration of the spike-triggered average window used for peel-off. |
| **Fitness** | Optimisation criterion: `SIL` (silhouette) or `CoV` (coefficient of variation of ISI). |
| **Peel-off** | Whether to subtract each found motor unit from the signal before searching for the next. Recommended: on. |
| **Swarm mode** | Enables the particle swarm optimiser. Recommended: on. |
| **Adapt clamp** | Adaptive clamping during whitening. Recommended: on. |

#### Per-grid parameters

These can be set independently for each electrode grid by selecting the grid from the dropdown:

| Parameter | What it does |
|-----------|--------------|
| **SIL Threshold** | Per-grid SIL acceptance criterion (overrides global for this grid). |
| **Extension Factor** | Number of delayed copies of each channel used to extend the observation space. Larger values capture more motor unit information but increase computation time. Typical: 10–30 for surface, 20–40 for intramuscular. |
| **High-pass (Hz)** | High-pass filter cutoff. Use ≥10 Hz for surface, ≥20 Hz for intramuscular. |
| **Low-pass (Hz)** | Low-pass filter cutoff. 4400 Hz is typical for surface at 10 kHz; adjust to ~half the Nyquist of your sampling rate. |
| **Notch filter** | Remove power-line interference: `None`, `50 Hz` (Europe), or `60 Hz` (Americas). |
| **Notch harmonics** | Also remove harmonics (100, 150 Hz etc.) when notch is active. |

#### Batch processing options

When multiple files are queued:

- **Shared rejection** — perform channel rejection on the first file only, then apply the same rejection mask to all subsequent files. Useful when recording conditions are stable across files.
- **Per-file rejection** — perform channel rejection independently for each file.

#### Starting decomposition

1. Click **Start Decomposition**.
2. A signal plot appears. Noisy or broken channels are shown as dashed lines. **Click any channel to toggle rejection** (rejected channels are excluded from decomposition).
   - Scroll to zoom the time axis
   - Shift+Scroll to pan
   - Right-drag to pan
   - `R` to reset the view
3. Optionally set a **time window** (plateau) to decompose only a steady-state segment. Click on the plot to set start and end points, or enter times manually.
4. Click **Confirm** to start the actual decomposition. Progress and found sources are shown in real time.
5. Click **Stop** at any time to halt early. Results from completed grids are saved.

When decomposition finishes, the results are automatically saved to a `.pkl` file in the configured output folder, and the Edition tab opens.

---

### Tab 3 — Edition

This is the main editing environment. It shows one motor unit at a time.

#### Navigating between units

- Use the **Port** dropdown to switch between electrode grids.
- Use the **Unit** dropdown or the **Up/Down arrow keys** to switch between motor units within a port.
- The **properties panel** on the right updates automatically:
  - **Spike count, mean discharge rate, CoV ISI, minimum ISI**
  - **SIL and PNR** — quality scores. A unit is **RELIABLE** if SIL ≥ 0.8 and PNR ≥ 30 dB.
  - **MUAP amplitude, waveform length, peak and median frequency**
  - **Duplicate warning** — if the current unit is very similar to another unit in the same port

#### Source signal plot

The large plot on the left shows the squared source signal (the spatial filter output) with spike locations marked as orange circles. A shaded region shows the plateau used for decomposition; outside this region the signal is reconstructed by replaying the peel-off sequence over the full recording.

**Scrolling / zooming:**
- **Scroll** — zoom in/out along the time axis
- **Shift+Scroll** — scroll horizontally
- **Ctrl+Scroll** — zoom in/out in both axes
- `Home` — reset view to show the full signal

If force channels are configured, a force trace is overlaid on the source plot. The right y-axis shows % MVC, scaled automatically to the actual force range.

#### Editing modes

| Mode | How to activate | What it does |
|------|----------------|--------------|
| **View** | `V` or button | Navigate without editing. Quick-edit with `Ctrl+Click` (add) or `Alt+Click` (delete). |
| **Add** | `A` or button | Click near a peak to add a spike. Snaps to nearest local maximum. |
| **Delete** | `D` or button | Click near an existing spike to remove the closest one. |
| **Add in Selection** | `Ctrl+A` or button | Drag a box to add all peaks found within it. |
| **Delete in Selection** | `Ctrl+D` or button | Drag a box to remove all spikes within it. |

Every edit shows a **live preview**: before you commit, the candidate spike's waveform is overlaid on the spike-triggered average (MUAP) plot so you can judge whether it looks like the unit's template. Press **Enter** to confirm or **Escape** to cancel.

Press `Ctrl+Z` to undo (up to 100 steps) and `Ctrl+Y` to redo.

#### ROI (Region of Interest)

Press `R` to toggle a draggable region on the signal. Then:
- `Shift+A` — add spikes within the ROI
- `Shift+D` — delete spikes within the ROI

#### MUAP plot

Shows the spike-triggered average for every EMG channel. Toggle between **stacked** and **grid** layout (the grid layout reflects the physical electrode geometry). Click a channel to open a pop-out window with a larger view.

#### Quality actions

| Button | Shortcut | Effect |
|--------|----------|--------|
| **Recalculate Filter** | `F` | Re-estimates the spatial filter from your edited spike train, then re-computes the source signal and re-detects timestamps. Requires ≥2 spikes in the plateau region. |
| **Auto-edit** | — | Automatically removes obvious outlier spikes based on physiological firing rate limits. |
| **Remove outliers** | — | Removes spikes with very short or very long ISIs. |
| **Flag unit** | `X` | Marks the unit for deletion. Duplicates are auto-flagged. |
| **Flag within-port duplicates** | — | Compares all units in the current port and flags pairs with high cross-correlation. |
| **Flag cross-port duplicates** | — | Same, across all ports. |
| **Delete All Flagged MUs** | — | Permanently removes all flagged units from the session (cannot be undone). |

#### Saving

`Ctrl+S` saves the current state to a `.pkl` file. The first save opens a dialog; subsequent saves to the same file happen silently. The saved file contains the edited spike trains, spatial filters, raw EMG, force data, and all metadata needed to reload and continue editing later.

---

### Tab 4 — Visualisation

This tab shows population-level summaries for all motor units in the loaded decomposition. It updates automatically whenever you switch to it or edit spikes in Tab 3.

The left panel lists all motor units by port. Click a unit to toggle it on/off. Use **All** / **None** to show or hide all units at once. The **Sort** dropdown reorders units by recruitment threshold, index, or mean discharge rate.

If force channels are configured, they appear as overlays on the time-domain plots. Click the legend (top-right of each plot) to toggle individual channels. The right y-axis shows % MVC, automatically scaled to the actual force range of the recording.

#### Sub-tabs

**Raster** — each row is one motor unit; vertical marks show individual spike times. Force is overlaid at the bottom, scaled to its actual %MVC range. The full force recording is shown even if only part of the signal was decomposed.

**IDR (Instantaneous Discharge Rate)** — smoothed discharge rate (pps) over time for each unit. Useful for checking that firing rates are physiologically plausible and that units track the force task.

**CST (Cumulative Spike Train)** — sum of all discharge rate traces. Approximates the neural drive to the muscle.

**Quality** — SIL and PNR bar charts for all units, with dashed threshold lines (SIL ≥ 0.8, PNR ≥ 32 dB). Units above the threshold are considered reliable.

**DR vs Force** — scatter plot of each motor unit's recruitment force (%MVC at first spike) against its mean discharge rate during the plateau. A regression line is drawn when three or more units are present. This plot requires at least one active force channel.

---

## Saving and loading

Decomposition results are stored as `.pkl` (Python pickle) files. Each file contains:

- Raw EMG data (all channels, full recording)
- Spike timestamps for every motor unit
- Spatial filters (original and edited)
- Force / auxiliary channel data (full recording)
- Channel layout, rejection mask, electrode geometry
- Peel-off sequence (for filter recalculation)
- Quality metrics

To **reload** a decomposition: in Tab 3, click **Load Decomposition** and select the `.pkl` file. The app will ask whether to re-run peel-off replay on the full signal (recommended for the first load) or to use the stored timestamps as-is (faster; appropriate when reloading a previously edited file).

Saved files can be reloaded in any order and remain fully editable.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+1` | Switch to Configuration tab |
| `Ctrl+2` | Switch to Decomposition tab |
| `Ctrl+3` | Switch to Edition tab |
| `V` | View mode |
| `A` | Add mode |
| `Ctrl+A` | Add in Selection mode |
| `D` | Delete mode |
| `Ctrl+D` | Delete in Selection mode |
| `R` | Toggle ROI |
| `Shift+A` | Add spikes in ROI |
| `Shift+D` | Delete spikes in ROI |
| `F` | Recalculate Filter |
| `X` | Flag unit |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Save |
| `Up / Down` | Next / previous motor unit |
| `Home` | Reset view |
| `Scroll` | Zoom time axis |
| `Shift+Scroll` | Scroll horizontally |
| `Ctrl+Scroll` | Zoom both axes |

---

## File formats

### Input

| Format | Notes |
|--------|-------|
| `.otb+` | OTBiolab+ archive (Quattrocento). EMG channels and auxiliary `.sip` channels (force, angle) are both supported. |
| `.mat` | MATLAB v5 and v7.3 (HDF5-based). The field containing the EMG matrix is configurable via `resources/loaders_configs/loader_mat.yaml`. |
| `.h5` | HDF5. Field path configurable via `resources/loaders_configs/loader_h5.yaml`. |
| `.npy` | NumPy array, shape `(channels, samples)` or `(samples, channels)` — the longer axis is assumed to be time. |
| `.csv` | Rows = samples, columns = channels. |

### Output

`.pkl` files are standard Python pickle files. They can be opened in Python with:

```python
import pickle
with open("my_decomp.pkl", "rb") as f:
    data = pickle.load(f)

# Key fields:
data["ports"]           # list of port names (one per electrode grid)
data["discharge_times"] # list[list[np.ndarray]] — spike timestamps in samples
data["sampling_rate"]   # int — sampling frequency in Hz
data["data"]            # np.ndarray — raw EMG, shape (channels, samples)
data["aux_channels"]    # list of dicts — force/aux data and metadata
```

---

## Force channel setup

Force channels allow you to overlay the force trace on source signal and discharge rate plots, and to normalise force to %MVC.

### Step 1 — Find the MVC value

The MVC value you enter must be in the **same units as the force signal stored in the file**.

**For OTBiolab+ recordings (.otb+):**
The Quattrocento ADC stores force channels internally in **millivolts (mV)** after its analogue-to-digital conversion. OTBiolab+ displays force in **Volts** on screen (e.g. "MVC = 0.049 V"). To get the correct value for the config:

```
MVC in mV = OTBiolab+ displayed value × 1000
```

Example: if OTBiolab+ shows `MVC = 0.049 V` for Middle Extension → enter `49` in the config.

You can find the displayed MVC value by opening the `.otb+` file in OTBiolab+ and reading the scale shown next to the force channel.

**For other formats:** use whatever MVC value is in the same units as the raw signal values in your file. You can check what the signal amplitude looks like by loading a decomposition and reading the console output — when data loads, the application prints the force channel min, max, and net amplitude so you can verify the units.

### Step 2 — Add the channel in the config

In Tab 1, click **+ Add Aux Channel** and fill in:

- **Name** — descriptive label (e.g. `Middle Ext`)
- **Unit label** — must match exactly if you want auto-selection by filename (e.g. `Middle Ext` will be auto-enabled for files with `mvc-15ext_fing-M` in the name)
- **Source** — `Signal` (channel embedded in the EMG array) or `Aux file` (OTB+ `.sip` stream)
- **Channel start / end** — 0-based index of the force channel
- **MVC (mV)** — value from Step 1

### Step 3 — Verify

After running decomposition, switch to Tab 4 and open the **DR vs Force** sub-tab. The right y-axis should show percentages consistent with the task (e.g. 0–20% for a 15% MVC contraction). If the values look wrong (e.g. showing 10,000%), check that the MVC value is in mV, not in Volts.

---

## Under development

- **Multi-file comparison** — loading two decompositions side by side for the same recording
- **Automated quality control** — automatic flagging beyond the basic SIL/PNR thresholds

---

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
agnese.grison@outlook.it
