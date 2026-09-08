"""Directional smoke-front propagation on a roadway graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .physical_models import smoke_spread_velocity


@dataclass(frozen=True)
class Roadway:
    roadway_id: str
    start_node: str
    end_node: str
    length: float
    slope_start_to_end: float = 0.0
    slope_end_to_start: float = 0.0
    ventilation_start_to_end: float = 0.0

    def direction_from(self, node_id: str) -> int:
        if node_id == self.start_node:
            return 1
        if node_id == self.end_node:
            return -1
        raise ValueError(f"Node {node_id!r} is not on roadway {self.roadway_id!r}.")

    def destination(self, direction: int) -> str:
        return self.end_node if direction > 0 else self.start_node


@dataclass
class SmokeFront:
    roadway_id: str
    position: float
    direction: int


@dataclass
class SmokeState:
    time: float = 0.0
    covered_intervals: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    active_fronts: list[SmokeFront] = field(default_factory=list)

    def is_position_covered(self, roadway_id: str, distance: float) -> bool:
        return any(
            lower <= float(distance) <= upper
            for lower, upper in self.covered_intervals.get(roadway_id, [])
        )


class DirectionalSmokeNetwork:
    """Advance smoke fronts using roadway slope and signed ventilation."""

    def __init__(
        self,
        roadways: Iterable[Roadway],
        *,
        wind_scale: float = 1.0,
    ) -> None:
        self.roadways = {roadway.roadway_id: roadway for roadway in roadways}
        self.wind_scale = float(wind_scale)
        self.connected: dict[str, list[str]] = {}
        for roadway in self.roadways.values():
            if roadway.length <= 0:
                raise ValueError("Roadway length must be positive.")
            self.connected.setdefault(roadway.start_node, []).append(
                roadway.roadway_id
            )
            self.connected.setdefault(roadway.end_node, []).append(
                roadway.roadway_id
            )
        self.state = SmokeState()
        self._activated_directions: set[tuple[str, int]] = set()

    def ignite(self, roadway_id: str, distance_from_start: float) -> None:
        roadway = self.roadways[roadway_id]
        position = min(max(float(distance_from_start), 0.0), roadway.length)
        self._add_interval(roadway_id, position, position)
        self._activate_front(roadway_id, position, 1)
        self._activate_front(roadway_id, position, -1)

    def update(self, elapsed_seconds: float) -> SmokeState:
        elapsed_seconds = max(0.0, float(elapsed_seconds))
        pending = [
            (front, elapsed_seconds) for front in self.state.active_fronts
        ]
        next_fronts: list[SmokeFront] = []
        transitions = 0

        while pending:
            front, remaining_time = pending.pop()
            roadway = self.roadways[front.roadway_id]
            speed = self.directional_speed(roadway, front.direction)
            if speed <= 0.0 or remaining_time <= 0.0:
                next_fronts.append(front)
                continue

            endpoint = roadway.length if front.direction > 0 else 0.0
            available_distance = abs(endpoint - front.position)
            travel_distance = speed * remaining_time

            if travel_distance < available_distance:
                new_position = front.position + front.direction * travel_distance
                self._add_interval(
                    roadway.roadway_id, front.position, new_position
                )
                next_fronts.append(
                    SmokeFront(
                        roadway_id=front.roadway_id,
                        position=new_position,
                        direction=front.direction,
                    )
                )
                continue

            self._add_interval(roadway.roadway_id, front.position, endpoint)
            time_used = available_distance / speed if speed > 0 else remaining_time
            time_left = max(0.0, remaining_time - time_used)
            reached_node = roadway.destination(front.direction)

            for next_roadway_id in self.connected.get(reached_node, []):
                if next_roadway_id == front.roadway_id:
                    continue
                next_roadway = self.roadways[next_roadway_id]
                next_direction = next_roadway.direction_from(reached_node)
                activation = (next_roadway_id, next_direction)
                if activation in self._activated_directions:
                    continue
                self._activated_directions.add(activation)
                start_position = 0.0 if next_direction > 0 else next_roadway.length
                pending.append(
                    (
                        SmokeFront(
                            roadway_id=next_roadway_id,
                            position=start_position,
                            direction=next_direction,
                        ),
                        time_left,
                    )
                )
                transitions += 1
                if transitions > 10_000:
                    raise RuntimeError("Smoke-front transition limit exceeded.")

        self.state.active_fronts = self._deduplicate_fronts(next_fronts)
        self.state.time += elapsed_seconds
        return self.state

    def directional_speed(self, roadway: Roadway, direction: int) -> float:
        if direction > 0:
            slope = roadway.slope_start_to_end
            ventilation = roadway.ventilation_start_to_end
        else:
            slope = roadway.slope_end_to_start
            ventilation = -roadway.ventilation_start_to_end
        inclination = math.atan(float(slope))
        return smoke_spread_velocity(
            inclination,
            float(ventilation) * self.wind_scale,
        )

    def _activate_front(
        self,
        roadway_id: str,
        position: float,
        direction: int,
    ) -> None:
        key = (roadway_id, 1 if direction > 0 else -1)
        if key in self._activated_directions:
            return
        self._activated_directions.add(key)
        self.state.active_fronts.append(
            SmokeFront(roadway_id, float(position), key[1])
        )

    def _add_interval(self, roadway_id: str, start: float, end: float) -> None:
        lower, upper = sorted((float(start), float(end)))
        intervals = self.state.covered_intervals.setdefault(roadway_id, [])
        intervals.append((lower, upper))
        intervals.sort()

        merged: list[tuple[float, float]] = []
        for current_lower, current_upper in intervals:
            if not merged or current_lower > merged[-1][1]:
                merged.append((current_lower, current_upper))
            else:
                previous_lower, previous_upper = merged[-1]
                merged[-1] = (
                    previous_lower,
                    max(previous_upper, current_upper),
                )
        self.state.covered_intervals[roadway_id] = merged

    @staticmethod
    def _deduplicate_fronts(front_list: list[SmokeFront]) -> list[SmokeFront]:
        selected: dict[tuple[str, int], SmokeFront] = {}
        for front in front_list:
            key = (front.roadway_id, front.direction)
            previous = selected.get(key)
            if previous is None:
                selected[key] = front
            elif front.direction > 0 and front.position > previous.position:
                selected[key] = front
            elif front.direction < 0 and front.position < previous.position:
                selected[key] = front
        return list(selected.values())
