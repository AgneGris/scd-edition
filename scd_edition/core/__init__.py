"""Core modules for scd-edition."""

from scd_edition.core.data_handler import DataHandler
from scd_edition.core.spike_editor import SpikeEditor, EditState
from scd_edition.core.analysis import SpikeTrain, QualityMetrics, DuplicateDetector

__all__ = [
    "DataHandler",
    "SpikeEditor",
    "EditState",
    "SpikeTrain",
    "QualityMetrics",
    "DuplicateDetector",
]