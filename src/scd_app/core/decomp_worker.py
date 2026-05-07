"""
Decomposition Worker - Manages EMG signal decomposition (via SCD).
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pickle

from PyQt5.QtCore import QThread, pyqtSignal

import torch
from scd.config.structures import Config
from scd.models.scd import SwarmContrastiveDecomposition


class DecompositionWorker(QThread):
    """Worker thread to run the SCD decomposition algorithm."""

    finished = pyqtSignal(dict)
    stopped = pyqtSignal(dict)  # emitted instead of finished when user stops
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    electrode_completed = pyqtSignal(int, int)
    source_found = pyqtSignal(object, object, int, float)

    def __init__(
        self,
        emg_data: torch.Tensor,
        grid_configs: dict,
        rejected_channels: List[np.ndarray],
        plateau_coords: np.ndarray,
        sampling_rate: int,
        save_path: Path,
        aux_configs: Optional[List[dict]] = None,
        emg_file_path: Optional[Path] = None,
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
                    good_channels = np.where(rejected == 0)[0]
                    noise_std = (
                        grid_data[:, good_channels].std().item()
                        if len(good_channels) > 0
                        else 1e-6
                    )
                    # Use a fixed seed (matching filter_recalculation._replace_bad_channels)
                    # so that the noise is reproducible when sources are recomputed on load.
                    gen = torch.Generator()
                    gen.manual_seed(42)
                    noise = (
                        torch.randn(
                            grid_data.shape[0], len(bad_channels), generator=gen
                        )
                        * noise_std
                    )
                    grid_data[:, bad_channels] = noise

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

            self.finished.emit(
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

    def _create_notch_params(self, params: dict) -> Optional[Tuple[int, float, bool]]:
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
            clamp_percentile=params["clamp"],
            use_coeff_var_fitness=(params["fitness"] == "CoV"),
            # Additional parameters
            peel_off=params["peel_off"],
            peel_off_window_size_ms=params["muap_window_ms"],
            peel_off_repeats=params.get("peel_off_repeats", True),
            swarm=params["swarm"],
            fixed_exponent=3,
            bad_channels=None,
            remove_bad_fr=False,
        )

    def _parse_notch(self, notch_str: str) -> Optional[int]:
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
        # 1. Channel counts, actual indices, and electrode info per port
        chans_per_electrode = []
        channel_indices = []  # actual absolute channel indices per port
        electrodes = []
        for port_name in results["ports"]:
            if port_name in self.grid_configs:
                cfg = self.grid_configs[port_name]
                chs = list(cfg.get("channels", []))
                chans_per_electrode.append(len(chs))
                channel_indices.append(chs)
                electrodes.append(cfg.get("electrode_type"))
            else:
                chans_per_electrode.append(64)
                channel_indices.append(None)
                electrodes.append(None)

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

        # 4. Aux channels — two source types:
        #    "signal"   → slice from full EMG array (channels, samples)
        #    "aux_file" → read .sip streams from the OTB+ archive
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
                if source == "aux_file":
                    if self.emg_file_path is None:
                        print(
                            f"  [aux] Skipping '{a.get('name', '?')}': "
                            "aux_file source requires emg_file_path"
                        )
                        continue
                    entry = self._load_aux_file_channel(self.emg_file_path, a)
                    if entry is not None:
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
                aux_channels_saved.append(
                    {
                        "data": sig,
                        "meta": {
                            k: v
                            for k, v in a.items()
                            if k not in ("start_chan", "end_chan")
                        },
                        "start_chan": s,
                        "end_chan": e,
                    }
                )
            print(f"  [aux] Saved {len(aux_channels_saved)} aux channel(s).")

        # 5. Build save dict
        save_dict = {
            "version": 1.0,
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

    def _load_aux_file_channel(self, file_path: Path, aux_config: dict) -> Optional[dict]:
        """Load one aux channel from an OTB+ .sip stream."""
        import tarfile

        file_path = Path(file_path)
        if file_path.suffix.lower() not in (".otb", ".otb+"):
            print(
                f"  [aux] '{aux_config.get('name', '?')}': "
                f"aux_file source only supported for OTB+ files, got {file_path.suffix!r}"
            )
            return None

        try:
            with tarfile.open(str(file_path), "r") as tar:
                members = {m.name: m for m in tar.getmembers()}
                sip_names = sorted(n for n in members if n.endswith(".sip"))
                if not sip_names:
                    print(f"  [aux] No .sip channels found in {file_path.name}")
                    return None
                arrays = [
                    np.frombuffer(
                        tar.extractfile(members[n]).read(), dtype="float64"
                    )
                    for n in sip_names
                ]
            min_len = min(len(a) for a in arrays)
            # (n_sip, samples) — channels-first to match the EMG convention
            aux_np = np.column_stack([a[:min_len] for a in arrays]).T
        except Exception as ex:
            print(f"  [aux] Failed to read .sip from {file_path.name}: {ex}")
            return None

        s = int(aux_config.get("start_chan", 0))
        e = int(aux_config.get("end_chan", s + 1))
        if s >= e or e > aux_np.shape[0]:
            print(
                f"  [aux] Skipping '{aux_config.get('name', '?')}': "
                f"sip range [{s},{e}) out of range ({aux_np.shape[0]} sip channels)"
            )
            return None

        sig = aux_np[s:e, :].squeeze()
        return {
            "data": sig,
            "meta": {
                k: v
                for k, v in aux_config.items()
                if k not in ("start_chan", "end_chan")
            },
            "start_chan": s,
            "end_chan": e,
        }

    def stop(self):
        self._is_running = False
