"""Event-triggered replanning state management for evacuation agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReplanReason(str, Enum):
    ROADWAY_CLOSURE = "roadway_closure"


@dataclass(frozen=True)
class ReplanningConfig:
    initial_decision_window_steps: int = 20
    replan_window_steps: int = 10
    cooldown_steps: int = 8
    switch_token_limit: int = 1


@dataclass(frozen=True)
class PlanningObservation:
    closure_revision: int


@dataclass(frozen=True)
class ReplanDecision:
    triggered: bool
    reason: ReplanReason | None
    initial_decision_allowed: bool
    route_refresh_allowed: bool
    target_switch_allowed: bool
    window_steps: int


@dataclass
class AgentPlanningState:
    goal: Optional[dict] = None
    route: Optional[list] = None
    initial_decision_locked: bool = False
    replan_window_left: int = 0
    switch_tokens: int = 0
    last_trigger_step: int = -10**9
    last_closure_revision: int = 0


class EventReplanningManager:

    def __init__(
        self,
        agent_ids: list[str],
        config: ReplanningConfig | None = None,
    ) -> None:
        self.config = config or ReplanningConfig()
        self.states = {
            str(agent_id): AgentPlanningState()
            for agent_id in agent_ids
        }
        self.step_count = 0

    def reset(self, closure_revision: int = 0) -> None:
        self.step_count = 0
        for agent_id in self.states:
            self.states[agent_id] = AgentPlanningState(
                last_closure_revision=int(closure_revision)
            )

    def commit_initial_plan(
        self,
        agent_id: str,
        *,
        goal: dict,
        route: list,
    ) -> bool:
        state = self._state(agent_id)
        if not self.initial_decision_open(agent_id):
            return False
        state.goal = dict(goal)
        state.route = list(route)
        state.initial_decision_locked = True
        return True

    def assign_plan_at_deadline(
        self,
        agent_id: str,
        *,
        goal: dict,
        route: list,
    ) -> bool:
        state = self._state(agent_id)
        if (
            state.initial_decision_locked
            or self.step_count
            < max(1, int(self.config.initial_decision_window_steps))
        ):
            return False
        state.goal = dict(goal)
        state.route = list(route)
        state.initial_decision_locked = True
        return True

    def initial_decision_open(self, agent_id: str) -> bool:
        state = self._state(agent_id)
        return (
            not state.initial_decision_locked
            and self.step_count
            < max(1, int(self.config.initial_decision_window_steps))
        )

    def macro_action_allowed(self, agent_id: str) -> bool:
        state = self._state(agent_id)
        return (
            self.initial_decision_open(agent_id)
            or state.replan_window_left > 0
        )

    def observe(
        self,
        agent_id: str,
        observation: PlanningObservation,
    ) -> ReplanDecision:
        state = self._state(agent_id)
        if not state.initial_decision_locked:
            state.last_closure_revision = int(observation.closure_revision)
            return self._decision(state, None, False)
        reason = self._detect_reason(state, observation)

        if reason is None:
            return self._decision(state, None, False)

        cooldown_complete = (
            self.step_count - state.last_trigger_step
            >= max(0, self.config.cooldown_steps)
        )
        if not cooldown_complete:
            return self._decision(state, reason, False)

        state.replan_window_left = max(
            state.replan_window_left,
            max(1, int(self.config.replan_window_steps)),
        )
        state.last_trigger_step = self.step_count
        state.switch_tokens = min(
            max(1, int(self.config.switch_token_limit)),
            state.switch_tokens + 1,
        )
        state.last_closure_revision = int(observation.closure_revision)
        return self._decision(state, reason, True)

    def consume_target_switch(
        self,
        agent_id: str,
        *,
        new_goal: dict,
        new_route: list,
    ) -> bool:
        state = self._state(agent_id)
        if state.replan_window_left <= 0 or state.switch_tokens <= 0:
            return False
        state.switch_tokens -= 1
        state.goal = dict(new_goal)
        state.route = list(new_route)
        return True

    def refresh_route(self, agent_id: str, route: list) -> bool:
        state = self._state(agent_id)
        if state.replan_window_left <= 0:
            return False
        state.route = list(route)
        return True

    def advance_step(self) -> None:
        self.step_count += 1
        for state in self.states.values():
            state.replan_window_left = max(
                0, state.replan_window_left - 1
            )

    def _detect_reason(
        self,
        state: AgentPlanningState,
        observation: PlanningObservation,
    ) -> ReplanReason | None:
        if observation.closure_revision != state.last_closure_revision:
            return ReplanReason.ROADWAY_CLOSURE
        return None

    def _decision(
        self,
        state: AgentPlanningState,
        reason: ReplanReason | None,
        triggered: bool,
    ) -> ReplanDecision:
        return ReplanDecision(
            triggered=triggered,
            reason=reason,
            initial_decision_allowed=(
                not state.initial_decision_locked
                and self.step_count
                < max(1, int(self.config.initial_decision_window_steps))
            ),
            route_refresh_allowed=state.replan_window_left > 0,
            target_switch_allowed=(
                state.replan_window_left > 0 and state.switch_tokens > 0
            ),
            window_steps=state.replan_window_left,
        )

    def _state(self, agent_id: str) -> AgentPlanningState:
        agent_id = str(agent_id)
        if agent_id not in self.states:
            raise KeyError(f"Unknown agent: {agent_id}")
        return self.states[agent_id]
