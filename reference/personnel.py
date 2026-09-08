"""Personnel movement along a selected sequence of roadway segments."""

from __future__ import annotations

from dataclasses import dataclass, field

from .physical_models import WorkerState, advance_worker_state
from .smoke_network import SmokeState


@dataclass(frozen=True)
class RouteSegment:
    roadway_id: str
    length: float
    directional_gradient: float
    enter_at_start: bool = True

    @property
    def direction(self) -> int:
        return 1 if self.enter_at_start else -1

    @property
    def entry_position(self) -> float:
        return 0.0 if self.enter_at_start else float(self.length)

    @property
    def exit_position(self) -> float:
        return float(self.length) if self.enter_at_start else 0.0


@dataclass
class Evacuee:
    agent_id: str
    route: list[RouteSegment]
    route_index: int = 0
    distance_from_start: float | None = None
    worker_state: WorkerState = field(default_factory=WorkerState)
    using_self_rescuer: bool = False
    cumulative_smoke_exposure: float = 0.0
    elapsed_time: float = 0.0
    final_state: str = "Active"

    def __post_init__(self) -> None:
        if not self.route:
            raise ValueError("An evacuee route must contain at least one segment.")
        if self.distance_from_start is None:
            self.distance_from_start = self.route[0].entry_position

    @property
    def current_segment(self) -> RouteSegment:
        return self.route[self.route_index]


class PersonnelMovementModel:
    """Advance evacuees using gradient, smoke, stamina, and equipment effects."""

    def __init__(
        self,
        *,
        smoke_exposure_limit: float = 1800.0,
        running_speed: float = 2.56,
        jogging_speed: float = 1.81,
        stamina_threshold: float = 30.0,
        base_consumption_rate: float = 0.5,
        self_rescuer_penalty: float = 1.1,
        smoke_speed_reduction: float = 0.68,
    ) -> None:
        self.smoke_exposure_limit = float(smoke_exposure_limit)
        self.running_speed = float(running_speed)
        self.jogging_speed = float(jogging_speed)
        self.stamina_threshold = float(stamina_threshold)
        self.base_consumption_rate = float(base_consumption_rate)
        self.self_rescuer_penalty = float(self_rescuer_penalty)
        self.smoke_speed_reduction = float(smoke_speed_reduction)

    def advance(
        self,
        evacuee: Evacuee,
        smoke_state: SmokeState,
        elapsed_seconds: float,
    ) -> Evacuee:
        if evacuee.final_state != "Active":
            return evacuee

        remaining_time = max(0.0, float(elapsed_seconds))
        while remaining_time > 1e-9 and evacuee.final_state == "Active":
            segment = evacuee.current_segment
            position = float(evacuee.distance_from_start or 0.0)
            in_smoke = smoke_state.is_position_covered(
                segment.roadway_id, position
            )
            evacuee.using_self_rescuer = bool(
                evacuee.using_self_rescuer or in_smoke
            )

            baseline = (
                self.running_speed
                if evacuee.worker_state.running
                else self.jogging_speed
            )
            preview_state = advance_worker_state(
                evacuee.worker_state,
                directional_gradient=segment.directional_gradient,
                in_smoke=in_smoke,
                wearing_self_rescuer=evacuee.using_self_rescuer,
                time_step=0.0,
                baseline_speed=baseline,
                running_speed=self.running_speed,
                jogging_speed=self.jogging_speed,
                stamina_threshold=self.stamina_threshold,
                base_consumption_rate=self.base_consumption_rate,
                self_rescuer_penalty=self.self_rescuer_penalty,
                smoke_speed_reduction=self.smoke_speed_reduction,
            )
            speed = max(preview_state.speed, 1e-9)
            distance_to_exit = abs(segment.exit_position - position)
            time_to_exit = distance_to_exit / speed
            step_time = min(remaining_time, time_to_exit)

            if in_smoke:
                exposure_remaining = (
                    self.smoke_exposure_limit
                    - evacuee.cumulative_smoke_exposure
                )
                if exposure_remaining <= 0.0:
                    evacuee.final_state = "SmokeTimeout"
                    break
                step_time = min(step_time, exposure_remaining)

            evacuee.worker_state = advance_worker_state(
                evacuee.worker_state,
                directional_gradient=segment.directional_gradient,
                in_smoke=in_smoke,
                wearing_self_rescuer=evacuee.using_self_rescuer,
                time_step=step_time,
                baseline_speed=baseline,
                running_speed=self.running_speed,
                jogging_speed=self.jogging_speed,
                stamina_threshold=self.stamina_threshold,
                base_consumption_rate=self.base_consumption_rate,
                self_rescuer_penalty=self.self_rescuer_penalty,
                smoke_speed_reduction=self.smoke_speed_reduction,
            )
            move_distance = evacuee.worker_state.speed * step_time
            new_position = position + segment.direction * move_distance
            if segment.direction > 0:
                new_position = min(new_position, segment.length)
            else:
                new_position = max(new_position, 0.0)

            evacuee.distance_from_start = new_position
            evacuee.elapsed_time += step_time
            remaining_time -= step_time
            if in_smoke:
                evacuee.cumulative_smoke_exposure += step_time
                if (
                    evacuee.cumulative_smoke_exposure
                    >= self.smoke_exposure_limit
                ):
                    evacuee.final_state = "SmokeTimeout"
                    break

            if abs(new_position - segment.exit_position) <= 1e-8:
                if evacuee.route_index + 1 >= len(evacuee.route):
                    evacuee.final_state = "Arrived"
                    break
                evacuee.route_index += 1
                evacuee.distance_from_start = (
                    evacuee.current_segment.entry_position
                )

        return evacuee
