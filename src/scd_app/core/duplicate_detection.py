"""Duplicate motor-unit detection independent of the GUI.

The scan functions update duplicate roles, partner lists, and deletion flags on
the supplied motor units. They return structured results for presentation and
audit logging by callers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from scd_app._vendor.motor_unit_toolbox import spike_comp as _tb_spike_comp
from scd_app.core.constants import ROA_THRESHOLD
from scd_app.core.mu_model import MotorUnit
from scd_app.core.mu_properties import build_spike_train_matrix

logger = logging.getLogger(__name__)

DUPLICATE_DETECTION_AVAILABLE = True

DuplicateKind = Literal["within", "cross"]
DuplicatePair = tuple[str, int, str, int, float]
AgreementComputer = Callable[..., tuple[np.ndarray, Any]]


@dataclass(frozen=True)
class DuplicateScanResult:
    """Domain result returned by a within- or cross-port duplicate scan."""

    pairs: list[DuplicatePair] = field(default_factory=list)
    flagged_by_port: dict[str, list[int]] = field(default_factory=dict)
    n_compared: int = 0
    skipped_ports: list[str] = field(default_factory=list)
    failed_ports: list[str] = field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return sum(len(unit_ids) for unit_ids in self.flagged_by_port.values())


def motor_unit_quality_key(motor_unit: MotorUnit) -> tuple[float, float, int, int]:
    """Return a sort key where a higher tuple means higher unit quality."""
    if motor_unit.props is None:
        return (-float("inf"), -float("inf"), 0, -motor_unit.id)
    silhouette = (
        motor_unit.props.sil if not np.isnan(motor_unit.props.sil) else -float("inf")
    )
    stability = (
        motor_unit.props.muap_template_stability
        if not np.isnan(motor_unit.props.muap_template_stability)
        else -float("inf")
    )
    return (silhouette, stability, motor_unit.props.n_spikes, -motor_unit.id)


def clear_duplicate_roles(
    ports: Mapping[str, Sequence[MotorUnit]], kind: DuplicateKind
) -> None:
    """Clear one scan kind without changing manual or other-scan flags."""
    for motor_units in ports.values():
        for motor_unit in motor_units:
            setattr(motor_unit, f"{kind}_duplicate_role", None)
            setattr(motor_unit, f"{kind}_duplicate_partners", [])


def _duplicate_keeper_keys(
    ports: Mapping[str, Sequence[MotorUnit]],
) -> dict[int, tuple[bool, float, float, int, int]]:
    """Snapshot keeper priority before this scan adds duplicate suggestions.

    All units still participate in comparisons, but an existing deletion flag
    must lose to an unflagged duplicate so a scan cannot schedule both units for
    deletion. Quality decides only when their existing flag state is equal.
    """
    return {
        id(motor_unit): (
            not motor_unit.flagged_for_deletion,
            *motor_unit_quality_key(motor_unit),
        )
        for motor_units in ports.values()
        for motor_unit in motor_units
    }


def scan_within_port_duplicates(
    ports: Mapping[str, Sequence[MotorUnit]],
    sampling_rate: float,
    *,
    threshold: float = ROA_THRESHOLD,
    agreement_computer: AgreementComputer | None = None,
) -> DuplicateScanResult:
    """Find duplicate pairs within each port and flag lower-quality units."""
    clear_duplicate_roles(ports, "within")
    keeper_keys = _duplicate_keeper_keys(ports)
    compute_agreement = agreement_computer or _tb_spike_comp.rate_of_agreement_full

    pairs: list[DuplicatePair] = []
    skipped_ports = []
    failed_ports = []
    n_compared = 0

    for port_name, motor_units in ports.items():
        if len(motor_units) < 2:
            if motor_units:
                skipped_ports.append(port_name)
            continue
        n_compared += len(motor_units)

        n_samples = max(len(motor_unit.source) for motor_unit in motor_units)
        spike_matrix = build_spike_train_matrix(
            [motor_unit.timestamps for motor_unit in motor_units], n_samples
        )
        try:
            agreement, _ = compute_agreement(
                spike_trains_ref=spike_matrix,
                spike_trains_test=spike_matrix,
                fs=int(round(sampling_rate)),
            )
        except Exception as exc:
            logger.warning("Within-port RoA failed for %s: %s", port_name, exc)
            failed_ports.append(port_name)
            continue

        for first_index in range(len(motor_units)):
            for second_index in range(first_index + 1, len(motor_units)):
                score = float(
                    max(
                        agreement[first_index, second_index],
                        agreement[second_index, first_index],
                    )
                )
                if score < threshold:
                    continue
                first = motor_units[first_index]
                second = motor_units[second_index]
                pairs.append((port_name, first.id, port_name, second.id, score))
                first.within_duplicate_partners.append((port_name, second.id, score))
                second.within_duplicate_partners.append((port_name, first.id, score))

        for motor_unit in motor_units:
            if not motor_unit.within_duplicate_partners:
                continue
            partner_ids = {
                unit_id
                for _port_name, unit_id, _score in motor_unit.within_duplicate_partners
            }
            partners = [
                candidate for candidate in motor_units if candidate.id in partner_ids
            ]
            best_partner = max(partners, key=lambda unit: keeper_keys[id(unit)])
            if keeper_keys[id(motor_unit)] >= keeper_keys[id(best_partner)]:
                motor_unit.within_duplicate_role = "keep"
            else:
                motor_unit.within_duplicate_role = "delete"

    flagged_by_port = {
        port_name: [
            motor_unit.id
            for motor_unit in motor_units
            if motor_unit.within_duplicate_role == "delete"
        ]
        for port_name, motor_units in ports.items()
    }
    return DuplicateScanResult(
        pairs=pairs,
        flagged_by_port=flagged_by_port,
        n_compared=n_compared,
        skipped_ports=skipped_ports,
        failed_ports=failed_ports,
    )


def scan_cross_port_duplicates(
    ports: Mapping[str, Sequence[MotorUnit]],
    sampling_rate: float,
    *,
    threshold: float = ROA_THRESHOLD,
    agreement_computer: AgreementComputer | None = None,
) -> DuplicateScanResult:
    """Find duplicate pairs across ports and flag lower-quality units."""
    clear_duplicate_roles(ports, "cross")
    keeper_keys = _duplicate_keeper_keys(ports)
    compute_agreement = agreement_computer or _tb_spike_comp.rate_of_agreement_full

    port_names = list(ports)
    pairs: list[DuplicatePair] = []
    failed_ports = []
    n_compared = sum(len(ports[port_name]) for port_name in port_names)

    for first_port_index in range(len(port_names)):
        for second_port_index in range(first_port_index + 1, len(port_names)):
            first_port = port_names[first_port_index]
            second_port = port_names[second_port_index]
            first_units = ports[first_port]
            second_units = ports[second_port]
            if not first_units or not second_units:
                continue

            n_samples = max(
                max(len(motor_unit.source) for motor_unit in first_units),
                max(len(motor_unit.source) for motor_unit in second_units),
            )
            first_spike_matrix = build_spike_train_matrix(
                [motor_unit.timestamps for motor_unit in first_units], n_samples
            )
            second_spike_matrix = build_spike_train_matrix(
                [motor_unit.timestamps for motor_unit in second_units], n_samples
            )
            try:
                agreement, _ = compute_agreement(
                    spike_trains_ref=first_spike_matrix,
                    spike_trains_test=second_spike_matrix,
                    fs=int(round(sampling_rate)),
                )
            except Exception as exc:
                logger.warning(
                    "Cross-port RoA failed for %s vs %s: %s",
                    first_port,
                    second_port,
                    exc,
                )
                failed_ports.append(f"{first_port} ↔ {second_port}")
                continue

            first_count, second_count = agreement.shape[:2]
            for first_index in range(min(len(first_units), first_count)):
                for second_index in range(min(len(second_units), second_count)):
                    score = float(agreement[first_index, second_index])
                    if score < threshold:
                        continue
                    first = first_units[first_index]
                    second = second_units[second_index]
                    pairs.append((first_port, first.id, second_port, second.id, score))
                    first.cross_duplicate_partners.append(
                        (second_port, second.id, score)
                    )
                    second.cross_duplicate_partners.append(
                        (first_port, first.id, score)
                    )

    for motor_units in ports.values():
        for motor_unit in motor_units:
            if not motor_unit.cross_duplicate_partners:
                continue
            partners = []
            for partner_port, partner_id, _score in motor_unit.cross_duplicate_partners:
                for candidate in ports.get(partner_port, []):
                    if candidate.id == partner_id:
                        partners.append(candidate)
                        break
            if not partners:
                continue
            best_partner = max(partners, key=lambda unit: keeper_keys[id(unit)])
            if keeper_keys[id(motor_unit)] >= keeper_keys[id(best_partner)]:
                if motor_unit.cross_duplicate_role != "delete":
                    motor_unit.cross_duplicate_role = "keep"
            else:
                motor_unit.cross_duplicate_role = "delete"

    flagged_by_port = {
        port_name: [
            motor_unit.id
            for motor_unit in motor_units
            if motor_unit.cross_duplicate_role == "delete"
        ]
        for port_name, motor_units in ports.items()
    }
    return DuplicateScanResult(
        pairs=pairs,
        flagged_by_port=flagged_by_port,
        n_compared=n_compared,
        failed_ports=failed_ports,
    )
