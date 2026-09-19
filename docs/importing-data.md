# Importing recordings without a built-in loader

Use the simplest route that preserves the original signal values and sampling
rate. SCD Edition never needs arbitrary Python embedded in a loader profile.

## MATLAB, HDF5, NumPy, CSV, or text

In **Configuration**, click **Choose recording…** and select the file. When the
selected data format cannot locate an EMG matrix, the inspector opens
automatically. You can reopen it with **Inspect arrays…** beside **Data Format**.
It lists numeric arrays with their shapes and data types. Choose the EMG matrix,
confirm whether samples or channels are the first axis, and enter the native
sampling rate. The preview shows the resulting sample and channel counts before
anything is loaded for decomposition.

For MATLAB structs, nested fields are shown with dot paths such as
`recording.signal`. HDF5 datasets use paths such as `recording/emg`. Sampling
rates named `sampling_rate_hz`, `sampling_frequency`, `sampling_rate`, `fsamp`,
or `fs` are detected when possible, but should still be checked against the
acquisition settings.

After configuring the electrode grids, use **Save Config**. The resulting JSON
keeps the selected array path, orientation, sampling rate, and channel layout;
load it after selecting another file with the same structure to reuse the
import profile. The data file itself is never modified.

CSV files must contain a numeric matrix with samples in rows and channels in
columns. A single non-numeric header row is accepted. `.txt` files use
whitespace-separated values. NumPy files must contain one numeric array and are
opened with pickle disabled.

## Proprietary formats

Use the manufacturer's SDK or documented parser to decode the file, then write
the result to the portable SCD HDF5 format. The helper protects existing files
unless `overwrite=True` is explicitly passed:

```python
import numpy as np

from scd_app.io import write_portable_recording

# Replace this with the vendor SDK. Keep the EMG sample-first.
emg = np.asarray(vendor_recording.emg, dtype=np.float32)  # samples × channels
aux = np.asarray(vendor_recording.force, dtype=np.float32)  # optional

write_portable_recording(
    "participant-01.scd.h5",
    emg,
    sampling_rate_hz=vendor_recording.sampling_rate,
    aux=aux,
    emg_unit="mV",
    source_format="Vendor model and SDK version",
    source_file="participant-01.vendor-extension",
)
```

The portable file is selected automatically by the **Portable SCD (.h5)**
loader. Its schema is deliberately small:

| Location | Required | Meaning |
|---|---:|---|
| `/emg` | yes | Numeric `(samples, channels)` matrix |
| root attribute `sampling_rate_hz` | yes | Native sampling frequency |
| root attribute `emg_unit` | yes | Unit of `/emg`, normally `mV` |
| `/aux` | no | Numeric `(samples, channels)` auxiliary matrix |
| `/timestamps` | no | One timestamp per sample |
| root attributes `source_format`, `source_file` | no | Conversion provenance |

Keep the original acquisition file as the source of record. The converted file
is an analysis derivative, so document the SDK/parser version and verify channel
order, units, sample count, sampling rate, and a few traces against the vendor
software before decomposition.

## Already decomposed elsewhere

If the Swarm Contrastive Decomposition package can load the recording, call
`scd.save_results(..., save_data=True)` and open its `.pkl` directly in the
Edition tab. This bypasses raw-data configuration while retaining the signal
needed for MUAPs, filter recalculation, and spike re-detection.

## When a real loader is worthwhile

For a format used repeatedly across a lab or community, a maintained built-in
loader is preferable to repeated conversion. Open an issue with a public or
synthetic sample, format documentation, expected channel order and units, and
sampling-rate metadata. Proprietary SDK integration should remain explicit and
installed by the user; loader YAML files describe data and do not execute code.
