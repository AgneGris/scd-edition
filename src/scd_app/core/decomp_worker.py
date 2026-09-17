"""
Decomposition Worker - Manages EMG signal decomposition (via SCD).
"""

import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from scd.config.structures import Config
from scd.models.scd import SwarmContrastiveDecomposition
from scd.processing.preprocess import replace_bad_channels_with_noise


class DecompositionWorker(QThread):
    """Worker thread to run the SCD decomposition algorithm."""

    # NB: not named `finished` — that would shadow QThread.finished, and this one is
    # emitted from inside run() while the thread is still alive.
    decomposition_finished = Signal(dict)
    stopped = Signal(dict)  # emitted instead of decomposition_finished when user stops
    error = Signal(str)
    progress = Signal(str)
    electrode_completed = Signal(int, int)
    source_found = Signal(object, object, int, float)

    def __init__(
        self,
        emg_data: torch.Tensor,
        grid_configs: dict,
        rejected_channels: list[np.ndarray],
        plateau_coords: np.ndarray,
        sampling_rate: int,
        save_path: Path,
        aux_configs: list[dict] | None = None,
        emg_file_path: Path | None = None,
        data_layout: dict | None = None,
    ):
        super().__init__()
        self.emg_data = emg_data
        self.grid_configs = grid_configs
        self.rejected_channels = rejected_channels
        self.plateau_coords = plateau_coords
        self.sampling_rate = sampling_rate
        self.save_path = save_path
        self.aux_configs = aux_configs or []
        self.emg_file_path = emg_file_path
        self.data_layout = data_layout
        self._aux_data_cache = {}
        self._is_running = True
        self._partial_results = None  # (results_dict, total_mus) after each grid

    def run(self):
        try:
            self.progress.emit("Starting decomposition...")

            print(f"[DecompositionWorker] Sampling rate: {self.sampling_rate} Hz")
            print(
                f"[DecompositionWorker] EMG data shape: {tuple(self.emg_data.shape)} (samples x channels)"
            )
            print(f"[DecompositionWorker] Grids: {list(self.grid_configs.keys())}")

            results = {
                "pulse_trains": [],
                "discharge_times": [],
                "mu_filters": [],
                "ports": [],
                "w_mat": [],  # list, one entry per grid
                "peel_off_sequence": [],  # list, one entry per grid
                "preprocessing_config": [],  # list, one entry per grid
            }

            total_mus = 0

            for grid_idx, (port_name, config) in enumerate(self.grid_configs.items()):
                if not self._is_running:
                    break

                self.progress.emit(
                    f"Processing {port_name} ({grid_idx + 1}/{len(self.grid_configs)})..."
                )

                # Extract data for this grid
                channels = config["channels"]
                n_total = self.emg_data.shape[1]
                print(
                    f"  [{port_name}] emg_data shape: {tuple(self.emg_data.shape)}, "
                    f"channels [{channels[0]}..{channels[-1]}] ({len(channels)} ch)"
                )
                bad_ch_idx = [c for c in channels if c >= n_total]
                if bad_ch_idx:
                    raise IndexError(
                        f"{port_name}: channel indices {bad_ch_idx} are out of range "
                        f"for EMG data with {n_total} channels."
                    )
                grid_data = self.emg_data[:, channels]  # (time, channels)

                # Replace rejected channels with baseline noise
                rejected = self.rejected_channels[grid_idx]
                # Guard: if mask was carried over from a different config, trim/pad it
                if len(rejected) != len(channels):
                    print(
                        f"  [{port_name}] Warning: rejection mask length ({len(rejected)}) "
                        f"!= n_channels ({len(channels)}), resetting mask."
                    )
                    rejected = np.zeros(len(channels), dtype=int)
                bad_channels = np.where(rejected == 1)[0]
                if len(bad_channels) > 0:
                    # Delegated to SCD so the decomposition and
                    # filter_recalculation._replace_bad_channels fill the
                    # channels identically. Fixed seed inside, so the noise is
                    # reproducible when sources are recomputed on load.
                    replace_bad_channels_with_noise(grid_data, bad_channels.tolist())

                # Slice to selected time window
                start_sample = int(self.plateau_coords[0])
                end_sample = int(self.plateau_coords[1])
                grid_data = grid_data[start_sample:end_sample, :]

                scd_config = self._create_scd_config(config["params"])
                print(f"\n--- Decomposition config for {port_name} ---")
                for field, value in vars(scd_config).items():
                    print(f"  {field}: {value}")
                print("---")

                dictionary, timestamps = self._decompose_grid(grid_data, scd_config)

                if dictionary and "filters" in dictionary:
                    results["pulse_trains"].append(dictionary["source"])
                    # Convert CUDA tensors to numpy so the PKL is portable
                    cpu_timestamps = (
                        [
                            (
                                t.detach().cpu().numpy()
                                if torch.is_tensor(t)
                                else np.asarray(t)
                            )
                            for t in timestamps
                        ]
                        if isinstance(timestamps, list)
                        else timestamps
                    )
                    results["discharge_times"].append(cpu_timestamps)
                    results["mu_filters"].append(dictionary["filters"])
                    results["ports"].append(port_name)
                    results["w_mat"].append(dictionary.get("w_mat"))
                    results["peel_off_sequence"].append(
                        dictionary.get("peel_off_sequence", [])
                    )
                    prep_cfg = dict(dictionary.get("preprocessing_config", {}))
                    # Patch in square_sources_spike_det — SCD does not include it in
                    # _capture_preprocessing_config but defaults it to True, so we
                    # record it explicitly so filter_recalculation can replay faithfully.
                    prep_cfg.setdefault(
                        "square_sources_spike_det",
                        bool(scd_config.square_sources_spike_det),
                    )
                    results["preprocessing_config"].append(prep_cfg)

                    n_mus = len(timestamps) if isinstance(timestamps, list) else 1
                    total_mus += n_mus
                    self.progress.emit(f"{port_name}: {n_mus} MUs found")

                else:
                    results["pulse_trains"].append(np.array([]))
                    results["discharge_times"].append([])
                    results["mu_filters"].append(np.array([]))
                    results["ports"].append(port_name)
                    results["w_mat"].append(None)
                    results["peel_off_sequence"].append([])
                    results["preprocessing_config"].append({})
                    self.progress.emit(f"{port_name}: 0 MUs found")

                self.electrode_completed.emit(grid_idx + 1, len(self.grid_configs))

                # Stash after every completed grid for partial save on stop
                self._partial_results = (results, total_mus)

            # Stopped early by user
            if not self._is_running:
                self.stopped.emit(
                    {
                        "path": str(self.save_path),
                        "n_units": total_mus,
                    }
                )
                return

            # Normal completion
            self.progress.emit("Saving results...")
            self._save_results(results)

            self.decomposition_finished.emit(
                {
                    "status": "success",
                    "path": str(self.save_path),
                    "n_units": total_mus,
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(str(e))

    def _create_notch_params(self, params: dict) -> tuple[int, float, bool] | None:
        """Create notch_params tuple from params dict."""
        notch_freq = self._parse_notch(params["notch_filter"])
        if notch_freq is None:
            return None
        return (notch_freq, 2.0, params["notch_harmonics"])

    def _create_scd_config(self, params: dict) -> Config:
        """Create SCD Config object from GUI parameters."""
        return Config(
            device="cuda" if torch.cuda.is_available() else "cpu",
            sampling_frequency=self.sampling_rate,
            start_time=0,
            end_time=-1,
            # Decomposition parameters
            acceptance_silhouette=params["sil_threshold"],
            max_iterations=params["iterations"],
            extension_factor=params["extension_factor"],
            # Filter parameters
            low_pass_cutoff=int(params["lowpass_hz"]),
            high_pass_cutoff=int(params["highpass_hz"]),
            notch_params=self._create_notch_params(params),
            # Algorithm parameters
            adapt_clamp=params["clamp"],
            use_coeff_var_fitness=(params["fitness"] == "CoV"),
            # Additional parameters
            peel_off=params["peel_off"],
            peel_off_window_size_ms=params["muap_window_ms"],
            peel_off_repeats=params.get("peel_off_repeats", True),
            swarm=params["swarm"],
            fixed_exponent=params.get("fixed_exponent", 2),
            bad_channels=None,
            remove_bad_fr=False,
        )

    def _parse_notch(self, notch_str: str) -> int | None:
        """Parse notch filter string to frequency."""
        if notch_str == "50":
            return 50
        elif notch_str == "60":
            return 60
        return None

    def _decompose_grid(self, grid_data: torch.Tensor, config):
        """Run SCD decomposition on a single grid."""
        grid_data = grid_data.to(device=config.device, dtype=torch.float32)

        def on_source_found(source, timestamps, iteration, silhouette):
            self.source_found.emit(source, timestamps, iteration, silhouette)

        model = SwarmContrastiveDecomposition()
        timestamps, dictionary = model.run(
            grid_data,
            config,
            source_callback=on_source_found,
        )
        return dictionary, timestamps

    def _save_results(self, results: dict):
        # 1. Channel counts, actual indices, electrode info and the user-set
        #    decomposition parameters per port.
        chans_per_electrode = []
        channel_indices = []  # actual absolute channel indices per port
        electrodes = []
        decomposition_params = []  # user-modifiable params, one dict per port
        for port_name in results["ports"]:
            if port_name in self.grid_configs:
                cfg = self.grid_configs[port_name]
                chs = list(cfg.get("channels", []))
                chans_per_electrode.append(len(chs))
                channel_indices.append(chs)
                electrodes.append(cfg.get("electrode_type"))
                decomposition_params.append(dict(cfg.get("params") or {}))
            else:
                chans_per_electrode.append(64)
                channel_indices.append(None)
                electrodes.append(None)
                decomposition_params.append({})

        # 2. Raw EMG — always (channels, samples)
        if torch.is_tensor(self.emg_data):
            data_np = self.emg_data.detach().cpu().numpy()
        else:
            data_np = np.asarray(self.emg_data)
        if data_np.shape[0] > data_np.shape[1]:
            data_np = data_np.T

        # 3. De-whitened filters (one list entry per grid)
        dewhitened_filters = []
        for i, filters in enumerate(results["mu_filters"]):
            w_mat = results["w_mat"][i]  # now safely indexed — it's a list
            if (
                isinstance(filters, np.ndarray)
                and filters.size > 0
                and w_mat is not None
                and isinstance(w_mat, np.ndarray)
                and w_mat.size > 0
            ):
                try:
                    dewhitened_filters.append(filters @ w_mat)
                except Exception as e:
                    print(f"Warning: de-whitening failed for grid {i}: {e}")
                    dewhitened_filters.append(None)
            else:
                dewhitened_filters.append(None)

        acquisition_metadata = {}
        if self.emg_file_path is not None and self.data_layout:
            try:
                from scd_app.io.data_loader import load_metadata

                acquisition_metadata = load_metadata(
                    self.emg_file_path, self.data_layout
                )
            except Exception as ex:
                print(f"  [metadata] Could not preserve acquisition metadata: {ex}")

        # 4. Aux channels — three source types:
        #    "signal"     → slice from full EMG array (channels, samples)
        #    "aux_file"   → read the acquisition format's canonical aux field
        #    "data_field" → read a named field out of the data file itself
        #                   (e.g. a MATLAB struct field holding calibrated force)
        #    Each saved entry: {"data": np.ndarray (samples,), "meta": dict,
        #                       "start_chan": int, "end_chan": int}
        #    data is NOT time-cropped; the full recording is preserved so force
        #    traces can be inspected outside the plateau window.
        aux_channels_saved = []
        if self.aux_configs:
            full_np = data_np  # already (channels, samples) — all time
            n_total_ch = full_np.shape[0]
            for a in self.aux_configs:
                source = a.get("source", "signal")
                if source == "data_field":
                    if self.emg_file_path is None:
                        print(
                            f"  [aux] Skipping '{a.get('name', '?')}': "
                            "data_field source requires emg_file_path"
                        )
                        continue
                    entry = self._load_data_field_channel(self.emg_file_path, a)
                    if entry is not None:
                        aux_channels_saved.append(entry)
                    continue

                if source == "aux_file":
                    if self.emg_file_path is None:
                        print(
                            f"  [aux] Skipping '{a.get('name', '?')}': "
                            "aux_file source requires emg_file_path"
                        )
                        continue
                    entry = self._load_aux_file_channel(self.emg_file_path, a)
                    if entry is not None:
                        s = int(a.get("start_chan", 0))
                        e = int(a.get("end_chan", s + 1))
                        channel_meta = acquisition_metadata.get("aux_channels", [])[s:e]
                        if channel_meta:
                            entry["channel_metadata"] = channel_meta
                            if len(channel_meta) == 1:
                                entry["physical_unit"] = channel_meta[0].get("unit")
                        aux_channels_saved.append(entry)
                    continue

                # source == "signal": slice from the EMG data array
                s, e = int(a.get("start_chan", 0)), int(a.get("end_chan", 0))
                if s >= e or e > n_total_ch:
                    print(
                        f"  [aux] Skipping '{a.get('name', '?')}': "
                        f"channel range [{s},{e}) out of range ({n_total_ch} ch)"
                    )
                    continue
                sig = full_np[s:e, :].squeeze()  # (samples,) for single-ch aux
                entry = dict(a.items())  # flat copy of full aux config
                entry["data"] = sig
                aux_channels_saved.append(entry)
            print(f"  [aux] Saved {len(aux_channels_saved)} aux channel(s).")

        # 5. Build save dict
        save_dict = {
            "version": 1.1,
            # User-modifiable decomposition parameters, one dict per port.  Everything the
            # user can set in the Decomposition tab: sil_threshold, iterations,
            # extension_factor, highpass_hz, lowpass_hz, notch_filter, notch_harmonics,
            # peel_off, peel_off_repeats, muap_window_ms, fitness, swarm, fixed_exponent,
            # clamp.  Saved so a decomposition can be reproduced or audited from its output
            # alone (preprocessing_config only records what SCD derived from these).
            "decomposition_params": decomposition_params,
            "aux_configs": [dict(a) for a in self.aux_configs],
            # Decomposition results
            "pulse_trains": results["pulse_trains"],
            "discharge_times": results["discharge_times"],
            "mu_filters": results["mu_filters"],
            "dewhitened_filters": dewhitened_filters,
            "ports": results["ports"],
            # Whitening matrices (one per grid) — needed for de-whitening later
            "w_mat": results["w_mat"],
            "peel_off_sequence": results[
                "peel_off_sequence"
            ],  # list[list], one per port
            "preprocessing_config": results[
                "preprocessing_config"
            ],  # list[dict], one per port
            # Metadata
            "sampling_rate": self.sampling_rate,
            "acquisition_metadata": acquisition_metadata,
            "plateau_coords": (
                self.plateau_coords.tolist()
                if hasattr(self.plateau_coords, "tolist")
                else list(self.plateau_coords)
            ),
            "data": data_np,  # (channels, samples)
            "chans_per_electrode": chans_per_electrode,
            "channel_indices": channel_indices,  # list[list[int]], one per port
            "emg_mask": [
                m.tolist() if isinstance(m, np.ndarray) else m
                for m in self.rejected_channels
            ],
            "electrodes": electrodes,
            "aux_channels": aux_channels_saved,  # list of {data, meta, start_chan, end_chan}
        }

        # 6. Write
        save_path_obj = Path(self.save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "wb") as f:
            pickle.dump(save_dict, f)
            print(f"File saved successfully: {self.save_path}")

    def _load_aux_file_channel(self, file_path: Path, aux_config: dict) -> dict | None:
        """Load one channel from the format's canonical auxiliary field."""
        from scd_app.io.data_loader import load_field

        file_path = Path(file_path)
        layout = copy.deepcopy(self.data_layout) if self.data_layout else None
        if not layout or "aux" not in layout.get("fields", {}):
            fmt = self._FORMAT_BY_SUFFIX.get(file_path.suffix.lower())
            if not fmt:
                print(
                    f"  [aux] '{aux_config.get('name', '?')}': cannot determine "
                    f"format for {file_path.suffix!r}"
                )
                return None
            layout = {
                "name": f"aux:{file_path.suffix.lower()}",
                "format": fmt,
                "fields": {
                    "aux": {
                        "path": "aux",
                        "channels": None,
                        "orientation": "samples_first",
                    }
                },
            }

        cache_key = (str(file_path), layout.get("format"))
        aux_np = self._aux_data_cache.get(cache_key)
        if aux_np is None:
            try:
                aux_np = load_field(file_path, layout, "aux").numpy()
                self._aux_data_cache[cache_key] = aux_np
            except Exception as ex:
                print(
                    f"  [aux] Failed to read auxiliary stream from "
                    f"{file_path.name}: {ex}"
                )
                return None

        s = int(aux_config.get("start_chan", 0))
        e = int(aux_config.get("end_chan", s + 1))
        if s >= e or e > aux_np.shape[1]:
            print(
                f"  [aux] Skipping '{aux_config.get('name', '?')}': "
                f"range [{s},{e}) out of range ({aux_np.shape[1]} aux channels)"
            )
            return None

        sig = aux_np[:, s:e].squeeze()
        entry = dict(aux_config.items())  # flat copy
        entry["data"] = sig
        return entry

    # Data-file formats understood by the loader, keyed by file extension. Used
    # only when no layout was supplied (e.g. a worker driven outside the GUI).
    _FORMAT_BY_SUFFIX = {
        ".mat": "mat",
        ".h5": "h5",
        ".hdf5": "h5",
        ".npy": "npy",
        ".otb": "otb",
        ".otb+": "otb",
        ".otb4": "otb4",
        ".rhs": "rhs",
    }

    def _load_data_field_channel(
        self, file_path: Path, aux_config: dict
    ) -> dict | None:
        """Load one aux channel from a named field inside the data file.

        Unlike the "signal" source this does not slice the EMG array, so it can
        reach values stored alongside it — e.g. a calibrated force trace in a
        MATLAB struct field. Dot notation walks structs (``signal.path``).
        """
        from scd_app.io.data_loader import load_field

        name = aux_config.get("name", "?")
        field_path = str(aux_config.get("field_path", "")).strip()
        if not field_path:
            print(f"  [aux] Skipping '{name}': data_field source needs a field_path")
            return None

        file_path = Path(file_path)
        fmt = None
        if self.data_layout:
            fmt = self.data_layout.get("format")
        if not fmt:
            fmt = self._FORMAT_BY_SUFFIX.get(file_path.suffix.lower())
        if not fmt:
            print(
                f"  [aux] Skipping '{name}': cannot determine format "
                f"for {file_path.suffix!r}"
            )
            return None

        # Minimal single-field layout; no channel slicing, auto orientation so a
        # (1, samples) row vector comes back the same way as (samples, 1).
        layout = {
            "name": f"aux:{field_path}",
            "format": fmt,
            "fields": {
                "aux": {
                    "path": field_path,
                    "fallback_keys": [],
                    "channels": None,
                    "orientation": "auto",
                }
            },
        }

        try:
            sig = load_field(file_path, layout, "aux").numpy()
        except Exception as ex:
            print(f"  [aux] Failed to read '{field_path}' from {file_path.name}: {ex}")
            return None

        sig = np.asarray(sig, dtype=float).squeeze()
        if sig.ndim > 1:
            print(
                f"  [aux] Skipping '{name}': field '{field_path}' is "
                f"{sig.shape}, expected a single trace"
            )
            return None
        if sig.size == 0:
            print(f"  [aux] Skipping '{name}': field '{field_path}' is empty")
            return None

        entry = dict(aux_config.items())  # flat copy
        entry["data"] = sig
        print(f"  [aux] '{name}': read {field_path} -> {sig.shape[0]} samples")
        return entry

    def stop(self):
        self._is_running = False
