# Example dataset

This folder contains a small, self-contained dataset for exploring SCD Edition
without using your own recording.

## Files

- [`emg.mat`](emg.mat) is a 10-second, 64-channel surface EMG recording sampled
  at 10,240 Hz. The MATLAB variable is named `emg` and has shape
  102,401 samples × 64 channels.
- [`emg_decomp_output.pkl`](emg_decomp_output.pkl) contains a decomposition of
  that recording with ten motor units, including their discharge times, source
  signals, filters and quality values.

## Using the example

To inspect the completed decomposition, launch SCD Edition, open the
**Edition** tab, click **Load**, and select `emg_decomp_output.pkl`. You can
review and edit its source signals and spike trains immediately.

The decomposition file does not embed a second copy of the raw signal. This
keeps it compact, but MUAP display and filter recalculation are unavailable
when it is opened by itself. The matching `emg.mat` is provided alongside it
for workflows that need the original recording.

To start from the raw recording instead, run:

```bash
scd-edition --example
```

This opens the application with the same recording's sampling rate, channel
layout and MATLAB variable already configured. The application command uses
the packaged copy of the recording so it also works after installation.

The same recording and decomposition are also used as demonstration data in
the Swarm-Contrastive Decomposition project.

Pickle files can execute code when opened. Only load decomposition files from
sources you trust.
