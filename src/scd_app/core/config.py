"""
Configuration management for SCD-Edition.
Handles session settings, electrode presets, and serialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json


@dataclass
class ElectrodeConfig:
    """Configuration for a single electrode array."""
    name: str
    type: str  # "surface_grid", "thin_film", "intramuscular"
    channels: List[int]  # Channel indices (0-based)
    rows: int = 1
    cols: int = 1
    spacing_mm: float = 4.0
    
    @property
    def n_channels(self) -> int:
        return len(self.channels)
    
    def validate(self):
        if not self.channels:
            raise ValueError(f"Electrode '{self.name}' has no channels defined")


@dataclass
class FilterConfig:
    """EMG filter configuration."""
    highpass_hz: float = 10.0
    lowpass_hz: float = 4400.0
    order: int = 4
    notch_hz: Optional[float] = None


@dataclass 
class DecompositionConfig:
    """Decomposition algorithm parameters."""
    extension_factor: Optional[int] = None
    acceptance_sil: float = 0.85
    max_iterations: int = 200
    use_cv_fitness: bool = True
    remove_bad_fr: bool = True
    min_fr_hz: float = 2.0
    max_fr_hz: float = 100.0
    peel_off_window_ms: float = 50.0


@dataclass
class PortConfig:
    name: str
    electrode: ElectrodeConfig
    filter: FilterConfig
    decomposition: DecompositionConfig
    muscle: str = ""
    enabled: bool = True


@dataclass
class SessionConfig:
    """Complete session configuration."""
    name: str
    sampling_frequency: int = 2048
    ports: List[PortConfig] = field(default_factory=list)
    input_dir: str = "data/input"
    emg_paths: List[str] = field(default_factory=list)
    output_dir: str = "data/output"
    
    # Edition settings
    auto_save: bool = True
    undo_levels: int = 50
    
    # Analysis thresholds
    roa_threshold: float = 0.3
    subset_threshold: float = 0.8
    
    def get_enabled_ports(self) -> List[PortConfig]:
        return [p for p in self.ports if p.enabled]


class ConfigManager:
    """Manages loading, saving, and validating configurations."""
    
    ELECTRODE_PRESETS = {
        "Grid 8x8 (64ch)": {"type": "surface_grid", "rows": 8, "cols": 8, "spacing_mm": 4.0},
        "Grid 4x8 (32ch)": {"type": "surface_grid", "rows": 8, "cols": 4, "spacing_mm": 4.0},
        "Linear 1x16 (16ch)": {"type": "linear", "rows": 16, "cols": 1, "spacing_mm": 5.0},
        "Grid 13x5 (64ch)": {"type": "surface_grid", "rows": 13, "cols": 5, "spacing_mm": 8.0},
        "Thin Film 8x8": {"type": "thin_film", "rows": 8, "cols": 8, "spacing_mm": 2.0},
    }
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
    
    def create_default_session(self, name: str = "New Session") -> SessionConfig:
        return SessionConfig(name=name)
    
    def add_port_from_preset(self, session: SessionConfig, port_name: str, electrode_preset: str, channel_start: int) -> PortConfig:
        if electrode_preset not in self.ELECTRODE_PRESETS:
            raise ValueError(f"Unknown electrode preset: {electrode_preset}")
        
        preset = self.ELECTRODE_PRESETS[electrode_preset]
        n_channels = preset["rows"] * preset["cols"]
        channels = list(range(channel_start, channel_start + n_channels))
        
        electrode = ElectrodeConfig(
            name=electrode_preset,
            type=preset["type"],
            channels=channels,
            rows=preset["rows"],
            cols=preset["cols"],
            spacing_mm=preset["spacing_mm"],
        )
        electrode.validate()
        
        port = PortConfig(
            name=port_name,
            electrode=electrode,
            filter=FilterConfig(),
            decomposition=DecompositionConfig()
        )
        
        session.ports.append(port)
        return port
    
    # === Serialization Methods ===

    def load_session(self, path: Path) -> SessionConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            if path.suffix.lower() in ['.json']:
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        return self._parse_session(data)
    
    def save_session(self, config: SessionConfig, path: Path):
        data = self._serialize_session(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _parse_session(self, data: Dict[str, Any]) -> SessionConfig:
        ports = []
        for port_data in data.get("ports", []):
            ports.append(self._parse_port(port_data))
        
        return SessionConfig(
            name=data.get("name", "unnamed"),
            sampling_frequency=data.get("sampling_frequency", 2048),
            ports=ports,
            input_dir=data.get("input_dir", ""),
            output_dir=data.get("output_dir", ""),
            auto_save=data.get("auto_save", True),
            undo_levels=data.get("undo_levels", 50),
        )
    
    def _parse_port(self, data: Dict[str, Any]) -> PortConfig:
        el_data = data["electrode"]
        electrode = ElectrodeConfig(
            name=el_data.get("name", "unnamed"),
            type=el_data.get("type", "unknown"),
            channels=el_data.get("channels", []),
            rows=el_data.get("rows", 1),
            cols=el_data.get("cols", 1),
            spacing_mm=el_data.get("spacing_mm", 4.0),
        )
        return PortConfig(
            name=data["name"],
            electrode=electrode,
            filter=FilterConfig(),
            decomposition=DecompositionConfig(),
            enabled=data.get("enabled", True),
        )
    
    def _serialize_session(self, config: SessionConfig) -> Dict[str, Any]:
        return {
            "name": config.name,
            "sampling_frequency": config.sampling_frequency,
            "input_dir": config.input_dir,
            "output_dir": config.output_dir,
            "auto_save": config.auto_save,
            "undo_levels": config.undo_levels,
            "ports": [self._serialize_port(p) for p in config.ports],
        }
    
    def _serialize_port(self, port: PortConfig) -> Dict[str, Any]:
        return {
            "name": port.name,
            "enabled": port.enabled,
            "electrode": {
                "name": port.electrode.name,
                "type": port.electrode.type,
                "channels": port.electrode.channels,
                "rows": port.electrode.rows,
                "cols": port.electrode.cols,
                "spacing_mm": port.electrode.spacing_mm,
            },
            # Serialize filter/decomp if modified
            "filter": {
                "highpass_hz": port.filter.highpass_hz,
                "lowpass_hz": port.filter.lowpass_hz,
                "order": port.filter.order,
                "notch_hz": port.filter.notch_hz,
            },
            "decomposition": {
                "extension_factor": port.decomposition.extension_factor,
            }
        }