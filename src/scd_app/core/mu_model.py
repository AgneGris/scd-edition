from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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
    mu_filter: Optional[np.ndarray] = None
    enabled: bool = True
    flagged_duplicate: bool = False
    props: Optional[MUProperties] = field(default=None, repr=False)

    # Duplicate detection roles — set by toolbar buttons, not persisted
    within_duplicate_role: Optional[str] = None   # "keep" | "delete" | None
    cross_duplicate_role: Optional[str] = None    # "keep" | "delete" | None
    # Partner tuples: (port_name, mu_id, roa_score)
    within_duplicate_partners: List[Tuple[str, int, float]] = field(default_factory=list)
    cross_duplicate_partners: List[Tuple[str, int, float]] = field(default_factory=list)


@dataclass
class UndoAction:
    description: str
    port_name: str
    mu_idx: int
    old_timestamps: Optional[np.ndarray] = None
    new_timestamps: Optional[np.ndarray] = None
    old_source: Optional[np.ndarray] = None
    old_filter: Optional[np.ndarray] = None
    new_source: Optional[np.ndarray] = None
    new_filter: Optional[np.ndarray] = None