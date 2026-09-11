# SCD Edition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A graphical application for researchers and engineers to decompose high-density surface or intramuscular EMG recordings into individual motor unit spike trains, edit them manually, and visualise population-level discharge behaviour.

![SCD Edition demo](docs/demo.gif)

Built on the [Swarm Contrastive Decomposition (SCD)](https://github.com/AgneGris/swarm-contrastive-decomposition) algorithm.
The desktop interface uses the official [Qt for Python (PySide6)](https://doc.qt.io/qtforpython-6/) bindings.

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
9. [License](#license)

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/AgneGris/scd-edition.git
```

All dependencies install automatically.

On Windows, this route may install a CPU-only PyTorch build. Use the CUDA-enabled uv route below to ensure NVIDIA GPU support.

### From source with uv (recommended for development)

[uv](https://github.com/astral-sh/uv) manages the virtual environment and dependencies automatically.

Clone the repository, then choose the command for your platform:

```bash
git clone https://github.com/AgneGris/scd-edition
cd scd-edition
```

On Windows, install and require a uv-managed Python to avoid DLL conflicts with Conda or Anaconda:

```powershell
uv python install 3.13
```

**Windows — CUDA-enabled (recommended if you have an NVIDIA GPU):**
```bash
uv sync --python 3.13 --managed-python --extra cuda
.venv\Scripts\Activate.ps1
```

**Linux — CUDA-enabled:**
```bash
sudo apt-get install libegl1 libxkbcommon-x11-0 libxcb-cursor0
uv sync --extra cuda
source .venv/bin/activate
```

**macOS or CPU-only Linux:**
```bash
uv sync --extra cpu
source .venv/bin/activate
```

On CPU-only Linux, install the same Qt runtime packages shown in the CUDA-enabled Linux example before launching the GUI. Package names may differ on distributions that do not use `apt`.

**CPU-only Windows:**
```bash
uv sync --python 3.13 --managed-python --extra cpu
.venv\Scripts\Activate.ps1
```

### From source with pip

```bash
git clone https://github.com/AgneGris/scd-edition
cd scd-edition
pip install -e .
```

## Usage 🚀

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
| `.otb4`   | OTBiolab 4 (Novecento+ amplifier) |
| `.rhs`    | Intan RHS2000 (Stim/Recording Controller) |
| `.mat`    | MATLAB |
| `.h5`     | HDF5 |
| `.npy`    | NumPy array |
| `.csv`    | Comma-separated values |

You can also queue **multiple files** for batch processing using **Add Files**. All files in the queue will be decomposed sequentially with the same configuration.

#### 2. Set the sampling rate

Enter the sampling rate of your recording in Hz (e.g. `2048` for a Quattrocento at 2 kHz). For formats that carry it in the file (`.otb4`, `.rhs`) it is filled in automatically.

**Decimate by** (next to the sampling rate) reduces the rate the decomposition works at by an integer factor, for any file format. The loader low-pass filters (zero-phase FIR) before subsampling, so nothing aliases, and the delivered rate is shown beside the box. Use it for recordings sampled far above the amplifier bandwidth — e.g. a 20 kHz Intan file band-limited at 500 Hz decimated by 5 to 4 kHz — so that extension factors and MUAP windows stay meaningful in milliseconds and runtime drops accordingly. Loader presets can seed the box (`decimate:` in the loader YAML); the value is saved with the configuration. The factor must divide the sampling rate exactly.

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
- **Source** — `Signal` if force is stored as regular channels in the EMG array; `Auxiliary stream` for OTB+ `.sip` or Novecento+ external/AUX tracks
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
  - **SIL and PNR** — quality scores.
  - **Reliability badge** — a unit is automatically **RELIABLE** when *every*
    quality criterion passes, i.e. when every metric in the panel is green:
    SIL ≥ 0.9, PNR ≥ 30 dB, CoV ISI ≤ 40 %, discharge rate 3–40 Hz and
    at least 10 spikes. Hover the badge to see which criteria a unit fails.
    Left-click the badge (or press `T`) to override the verdict by hand — an
    overridden badge is italic and marked *(manual)*, and the unit is shown as
    `✓*` / `✗*` in the unit dropdown. Right-click (or press `Shift+T`) to go
    back to the automatic verdict. Manual verdicts are saved with the session.
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
| **Flag within-port duplicates** | — | Compares all units in each port and flags the lower-quality unit of every pair whose rate of agreement is above threshold. A summary dialog reports the pairs found, their RoA scores and which units were flagged. |
| **Flag cross-port duplicates** | — | Same, across all ports, with the same summary dialog. |
| **Delete All Flagged MUs** | — | Permanently removes all flagged units from the session (cannot be undone). Afterwards the view jumps back to the first unit of the first port. |

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

**Quality** — SIL and PNR bar charts for all units, with dashed threshold lines at the reliability thresholds (SIL ≥ 0.9, PNR ≥ 30 dB).

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

To **reload** a decomposition: in Tab 3, click **Load Decomposition** and select the `.pkl` file. The app automatically recognises both SCD Edition files and raw `*_scddict.pkl` output from `swarm-contrastive-decomposition`. Raw SCD output is converted into a one-grid Edition session; the muscle label in the filename is used as the grid name, and any companion `*_scdcommit.txt` is retained as provenance.

SCD Edition files that contain the original EMG can replay peel-off and recalculate filters. Raw `swarm-contrastive-decomposition` output does not contain the original EMG, so its motor-unit spikes and source signals remain editable and can be saved from Edition, but filter/MUAP recalculation is unavailable unless the original EMG is supplied in an Edition file.

For compatible SCD Edition files, the app may ask whether to re-run peel-off replay on the full signal (recommended for the first load) or to use the stored timestamps as-is (faster; appropriate when reloading a previously edited file).

Saved Edition files can be reloaded in any order and remain fully editable.

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
| `T` | Toggle the reliability verdict of the current unit |
| `Shift+T` | Reset reliability to the automatic verdict |
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
| `.otb4` | OTBiolab 4 archive (Novecento+). EMG grids, external/AUX channels, sampling rate, and acquisition metadata are discovered from the embedded track metadata. |
| `.rhs` | Intan RHS2000 recording. Enabled amplifier channels (mV), board ADC inputs (V, as `aux`) and the sampling rate are read from the file header. Intan recordings are usually heavily oversampled relative to the amplifier bandwidth; use **Decimate by** (e.g. 5 for 20 kHz → 4 kHz with a 500 Hz band limit). |
| `.mat` | MATLAB v5 and v7.3 (HDF5-based). The field containing the EMG matrix is configurable via `src/scd_app/resources/loaders_configs/loader_mat.yaml`. |
| `.h5` | HDF5. Field path configurable via `src/scd_app/resources/loaders_configs/loader_h5.yaml`. |
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

**For Intan recordings (.rhs):** board ADC inputs are converted to Volts
(312.5 µV/bit); enter MVC in Volts.

**For Novecento+ recordings (.otb4):** external/AUX channels are converted
using their per-track ADC metadata and retain the unit declared in the file
(the supplied Novecento+ examples declare Volts). Enter MVC in that same unit.

**For other formats:** use whatever MVC value is in the same units as the raw signal values in your file. You can check what the signal amplitude looks like by loading a decomposition and reading the console output — when data loads, the application prints the force channel min, max, and net amplitude so you can verify the units.

### Step 2 — Add the channel in the config

In Tab 1, click **+ Add Aux Channel** and fill in:

- **Name** — descriptive label (e.g. `Middle Ext`)
- **Unit label** — must match exactly if you want auto-selection by filename (e.g. `Middle Ext` will be auto-enabled for files with `mvc-15ext_fing-M` in the name)
- **Source** — `Signal` (channel embedded in the EMG array) or `Auxiliary stream` (OTB+ `.sip` or Novecento+ external/AUX track)
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
  volume={72},
  number={1},
  pages={227--237},
  year={2025},
  doi={10.1109/TBME.2024.3446806},
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
  doi={10.1113/JP287913},
  publisher={Wiley Online Library}
}
```

## License

SCD Edition is open-source software licensed under the [BSD 3-Clause License](LICENSE).

Third-party packages, including PySide6 and Qt, retain their own licences.

## Contact

**Agnese Grison**  
agnese.grison@outlook.it
