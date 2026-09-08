"""Reference implementations of the smoke-spread and worker-movement models."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final


_SMOKE_COEFFICIENTS: Final[dict[str, tuple[float, ...]]] = {
    "A": (-455.52, 1014.76, 215.01, -171.83, -1003.05, 456.23),
    "B": (214.59, -725.95, -100.35, 119.71, 723.09, -213.46),
    "C": (-20.79, 81.14, 9.76, -13.30, -80.96, 20.79),
}


def _coefficient_function(
    inclination_radians: float,
    coefficients: tuple[float, ...],
) -> float:
    """Evaluate the polynomial-trigonometric coefficient function."""

    c0, c1, c2, c3, c4, c5 = coefficients
    theta = float(inclination_radians)
    return (
        c0
        + c1 * theta
        + c2 * theta**2
        + c3 * theta**3
        + c4 * math.sin(theta)
        + c5 * math.cos(theta)
    )


def smoke_spread_velocity(
    inclination_radians: float,
    ventilation_velocity: float,
) -> float:
    """Calculate smoke-front velocity in metres per second.

    The inclination angle follows the travel direction and must be supplied in
    radians. The ventilation velocity uses the same signed direction
    convention.
    """

    theta = float(inclination_radians)
    ventilation = float(ventilation_velocity)
    coefficient_a = _coefficient_function(theta, _SMOKE_COEFFICIENTS["A"])
    coefficient_b = _coefficient_function(theta, _SMOKE_COEFFICIENTS["B"])
    coefficient_c = _coefficient_function(theta, _SMOKE_COEFFICIENTS["C"])
    velocity = (
        coefficient_a
        + coefficient_b * ventilation
        + coefficient_c * ventilation**2
    )
    return max(0.0, velocity)


def gradient_adjusted_speed(
    baseline_speed: float,
    directional_gradient: float,
) -> float:
    """Apply the Tobler-type directional-gradient relationship."""

    return float(baseline_speed) * math.exp(
        -3.5 * abs(float(directional_gradient) + 0.05)
    )


def stamina_after_step(
    stamina: float,
    directional_gradient: float,
    time_step: float,
    *,
    base_consumption_rate: float = 0.5,
    wearing_self_rescuer: bool = False,
    self_rescuer_penalty: float = 1.1,
) -> float:
    """Update stamina using slope and self-rescuer penalty coefficients."""

    slope_penalty = 1.0 + 5.0 * max(float(directional_gradient), 0.0)
    equipment_penalty = float(self_rescuer_penalty) if wearing_self_rescuer else 1.0
    consumption = (
        float(time_step)
        * float(base_consumption_rate)
        * slope_penalty
        * equipment_penalty
    )
    return max(0.0, float(stamina) - consumption)


def integrated_worker_speed(
    baseline_speed: float,
    directional_gradient: float,
    *,
    in_smoke: bool,
    smoke_speed_reduction: float = 0.68,
    minimum_speed: float = 0.2,
) -> float:
    """Combine directional-gradient and binary smoke-speed effects."""

    speed = gradient_adjusted_speed(baseline_speed, directional_gradient)
    if in_smoke:
        speed -= float(smoke_speed_reduction)
    return max(float(minimum_speed), speed)


@dataclass(frozen=True)
class WorkerState:
    """Worker state used by the movement-model update."""

    stamina: float = 100.0
    running: bool = True
    speed: float = 0.0


def advance_worker_state(
    state: WorkerState,
    *,
    directional_gradient: float,
    in_smoke: bool,
    wearing_self_rescuer: bool,
    time_step: float,
    baseline_speed: float | None = None,
    running_speed: float = 2.56,
    jogging_speed: float = 1.81,
    stamina_threshold: float = 30.0,
    base_consumption_rate: float = 0.5,
    self_rescuer_penalty: float = 1.1,
    smoke_speed_reduction: float = 0.68,
) -> WorkerState:
    """Advance stamina, speed mode, and movement speed by one time step."""

    if state.running:
        next_stamina = stamina_after_step(
            state.stamina,
            directional_gradient,
            time_step,
            base_consumption_rate=base_consumption_rate,
            wearing_self_rescuer=wearing_self_rescuer,
            self_rescuer_penalty=self_rescuer_penalty,
        )
        next_running = bool(next_stamina > float(stamina_threshold))
        if not next_running:
            next_stamina = float(stamina_threshold)
    else:
        next_stamina = float(state.stamina)
        next_running = False
    selected_baseline = (
        float(baseline_speed)
        if baseline_speed is not None
        else (float(running_speed) if next_running else float(jogging_speed))
    )
    next_speed = integrated_worker_speed(
        selected_baseline,
        directional_gradient,
        in_smoke=in_smoke,
        smoke_speed_reduction=smoke_speed_reduction,
    )
    return replace(
        state,
        stamina=next_stamina,
        running=next_running,
        speed=next_speed,
    )
