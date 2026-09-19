# SCD Edition quick-start

This walkthrough uses the example recording bundled with every installation.
It contains 64 channels and ten seconds of EMG sampled at 10,240 Hz.

## 1. Install and open the example

```bash
pip install scd-edition
scd-edition --example
```

The Configuration tab opens with the MATLAB loader, `emg` variable, sampling
rate and a 64-channel surface grid already selected. The output directory is
`scd-edition-output` under the directory from which you launched the app.

## 2. Apply the configuration

Click **Apply Configuration**. The Decomposition tab loads the recording and
shows the settings for `Example_Grid`.

For a quick functional check, retain the defaults and select a short interval
before running. For a scientific analysis, choose parameters appropriate for
the acquisition, muscle and contraction rather than treating the example
settings as universal defaults.

## 3. Decompose

Click **Start Decomposition**. Confirm the channel-rejection view, choose the
full recording or a time interval, then click **Confirm & Run**. CUDA is used
automatically when a compatible PyTorch installation and NVIDIA driver are
available; otherwise the calculation runs on the CPU.

## 4. Review and edit

Completed results open in the Edition tab. Select a motor unit, zoom the source
plot and use **Add** or **Delete** to correct spikes. Press `Ctrl+Z` to undo.
The right-hand panel recomputes quality and MUAP properties after edits.

## 5. Inspect the population

Open the Visualisation tab for the raster, instantaneous discharge rate,
cumulative spike train and quality summaries. Press `Ctrl+S` in Edition to save
an editable `.pkl` session.

## Inspect the example in Python

The companion [`tutorial.ipynb`](tutorial.ipynb) locates the installed data,
loads it through the same loader as the GUI, and plots several channels.

To use a recording whose MATLAB/HDF5 layout is not recognised, choose the
recording normally; the array inspector opens automatically. You can reopen it
with **Inspect arrays…** beside **Data Format**. For proprietary formats, follow the
[data-import and portable-conversion guide](importing-data.md).
