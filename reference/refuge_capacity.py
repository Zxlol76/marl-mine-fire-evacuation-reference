"""Refuge chamber capacity and reservation management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class CapacityStatus:
    capacity: int
    occupancy: int
    reserved_total: int
    reserved_effective: int
    effective_remaining: int


class RefugeCapacityManager:
    """Maintain immediate reservations and completed chamber entries."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        self.capacities = {
            str(chamber_id): max(0, int(capacity))
            for chamber_id, capacity in capacities.items()
        }
        self.occupancy = {chamber_id: 0 for chamber_id in self.capacities}
        self.reserved = {chamber_id: 0 for chamber_id in self.capacities}
        self.agent_reservation: dict[str, Optional[str]] = {}

    def status(
        self,
        chamber_id: str,
        *,
        holder_agent_id: str | None = None,
    ) -> CapacityStatus:
        chamber_id = self._require_chamber(chamber_id)
        capacity = self.capacities[chamber_id]
        occupancy = self.occupancy[chamber_id]
        reserved_total = self.reserved[chamber_id]
        reserved_effective = reserved_total

        if holder_agent_id is not None:
            held_chamber = self.agent_reservation.get(str(holder_agent_id))
            if held_chamber == chamber_id:
                reserved_effective = max(0, reserved_effective - 1)

        return CapacityStatus(
            capacity=capacity,
            occupancy=occupancy,
            reserved_total=reserved_total,
            reserved_effective=reserved_effective,
            effective_remaining=capacity - occupancy - reserved_effective,
        )

    def reserve(self, agent_id: str, chamber_id: str) -> bool:
        """Reserve one crew-capacity unit for an agent."""

        agent_id = str(agent_id)
        chamber_id = self._require_chamber(chamber_id)
        previous = self.agent_reservation.get(agent_id)

        if previous == chamber_id:
            return True
        if self.status(chamber_id).effective_remaining <= 0:
            return False

        if previous is not None:
            self.release(agent_id)
        self.reserved[chamber_id] += 1
        self.agent_reservation[agent_id] = chamber_id
        return True

    def release(self, agent_id: str) -> str | None:
        """Release the reservation currently held by an agent."""

        agent_id = str(agent_id)
        chamber_id = self.agent_reservation.pop(agent_id, None)
        if chamber_id is None:
            return None
        self.reserved[chamber_id] = max(0, self.reserved[chamber_id] - 1)
        return chamber_id

    def enter(self, agent_id: str, chamber_id: str | None = None) -> bool:
        """Convert a reservation into occupancy when the crew arrives."""

        agent_id = str(agent_id)
        held_chamber = self.agent_reservation.get(agent_id)
        target = held_chamber if chamber_id is None else self._require_chamber(chamber_id)
        if target is None:
            return False

        status = self.status(target, holder_agent_id=agent_id)
        owns_reservation = held_chamber == target
        if not owns_reservation and status.effective_remaining <= 0:
            return False

        if held_chamber is not None:
            self.release(agent_id)
        self.occupancy[target] += 1
        return True

    def _require_chamber(self, chamber_id: str) -> str:
        chamber_id = str(chamber_id)
        if chamber_id not in self.capacities:
            raise KeyError(f"Unknown refuge chamber: {chamber_id}")
        return chamber_id
