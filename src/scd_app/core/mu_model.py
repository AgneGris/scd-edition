from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from scd_app.core.mu_properties import MUProperties


class EditMode(Enum):
    VIEW = "view"
    ADD = "add"
    DELETE = "delete"


@dataclass
class MotorUnit:
    id: int
    timestamps: np.ndarray
    source: np.ndarray
    port_name: str = ""
    mu_filter: np.ndarray | None = None
    enabled: bool = True
    # Persisted user/general deletion flag. Duplicate suggestions remain in
    # their scan-specific roles and are combined by ``flagged_for_deletion``.
    flagged_duplicate: bool = False
    reviewed: bool = False
    props: MUProperties | None = field(default=None, repr=False)

    notes: str = ""

    # Duplicate detection roles — set by toolbar buttons, not persisted
    within_duplicate_role: str | None = None  # "keep" | "delete" | None
    cross_duplicate_role: str | None = None  # "keep" | "delete" | None
    # Partner tuples: (port_name, mu_id, roa_score)
    within_duplicate_partners: list[tuple[str, int, float]] = field(
        default_factory=list
    )
    cross_duplicate_partners: list[tuple[str, int, float]] = field(default_factory=list)

    @property
    def flagged_for_deletion(self) -> bool:
        """Whether any manual or duplicate-scan reason marks this unit."""
        return (
            self.flagged_duplicate
            or self.within_duplicate_role == "delete"
            or self.cross_duplicate_role == "delete"
        )


@dataclass
class UndoAction:
    description: str
    port_name: str
    mu_idx: int
    old_timestamps: np.ndarray | None = None
    new_timestamps: np.ndarray | None = None
    old_source: np.ndarray | None = None
    old_filter: np.ndarray | None = None
    new_source: np.ndarray | None = None
    new_filter: np.ndarray | None = None
