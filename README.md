# SCD Edition 🔧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A PyQt5 GUI application for editing and visualizing EMG decomposition results from the [swarm-contrastive-decomposition](https://github.com/AgneGris/swarm-contrastive-decomposition) package.

## Features ✨

- **Spike Editing**: Add and delete spikes with visual preview
- **ROI Selection**: Bulk add/delete spikes within a region of interest
- **Quality Metrics**: Real-time SIL (Silhouette) calculation
- **Duplicate Detection**: Automatic detection of duplicate and subset motor units
- **Filter Recalculation**: Recalculate source filters after spike editing
- **Export**: Save to pickle or EMGlab (.eaf) format

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

### Workflow

1. **Load Data**: Click "Load Data" and select:
   - Decomposition file (`.pkl` from SCD)
   - EMG file (`.mat`)

2. **Edit Spikes**:
   - **View Mode (V)**: Navigate and inspect
   - **Add Mode (A)**: Click to preview → Enter to confirm
   - **Delete Mode (D)**: Click to preview → Enter to confirm
   - **ROI Mode (R)**: Draw region → ROI Add/Delete buttons

3. **Quality Control**:
   - "Remove Duplicates" to flag duplicate units
   - "Recalculate Filters" after major edits
   - "Flag Unit" to mark units for removal

4. **Save**: 
   - "Save Edited Data" → `.pkl`
   - "Convert to EMGlab" → `.eaf`

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

## Project Structure 📁

```
scd-edition/
├── scd_edition/
│   ├── __init__.py
│   ├── app.py              # Entry point
│   ├── core/
│   │   ├── data_handler.py # Data loading/saving
│   │   ├── spike_editor.py # Spike editing logic
│   │   └── analysis.py     # Quality metrics
│   ├── gui/
│   │   ├── main_window.py  # Main window
│   │   └── styles.py       # Theming
│   └── export/
│       └── emglab.py       # EMGlab format
└── pyproject.toml
```

## Citation

If you use this software, please cite the SCD paper:

```bibtex
@article{grison2024particle,
  author={Grison, Agnese and Clarke, Alexander Kenneth and Muceli, Silvia and Ibáñez, Jaime and Kundu, Aritra and Farina, Dario},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={A Particle Swarm Optimised Independence Estimator for Blind Source Separation of Neurophysiological Time Series}, 
  year={2024},
  doi={10.1109/TBME.2024.3446806}
}
```

## Contact

**Agnese Grison**  
📧 agnese.grison16@imperial.ac.uk