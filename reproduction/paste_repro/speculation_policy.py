"""Non-neural confidence calibration and authority-first speculation policy.

The policy deliberately separates three questions:

* a grouped-session pattern table estimates exact-candidate probability;
* a load-dependent shadow price turns probability into expected net utility;
* a global selector spends scarce speculative starts on the highest utility
  candidates, without assigning one candidate to every task.

It contains no embedding, neural model, or gradient-trained component.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Callable, Hashable


def query_bucket(value: int) -> str:
    if value <= 1:
        return "q1"
    if value == 2:
        return "q2"
    if value <= 4:
        return "q3-4"
    if value <= 9:
        return "q5-9"
    return "q10+"


def streak_bucket(value: int) -> str:
    if value == 1:
        return "s1"
    if value == 2:
        return "s2"
    return "s3+"


def sequence_bucket(value: int) -> str:
    if value == 1:
        return "w1"
    if value == 2:
        return "w2"
    if value <= 4:
        return "w3-4"
    return "w5+"


def rank_bucket(value: int) -> str:
    return str(value) if value in {1, 2, 3, 4, 5} else "6+"


@dataclass(frozen=True)
class CandidatePattern:
    """Causal, discrete features available before the next tool decision."""

    session_id: str
    decision_id: str
    url: str
    position: int
    query_count: int
    search_streak: int
    search_sequence: int
    candidate_count: int
    current_count: int
    repeated_current: bool
    source_rank: int
    current: bool
    was_visited: bool
    search_age: int
    appearances: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.decision_id:
            raise ValueError("session_id and decision_id must be non-empty")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("url must be an exact absolute HTTP(S) URL")
        for name in (
            "position",
            "search_streak",
            "search_sequence",
            "candidate_count",
            "source_rank",
            "appearances",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.query_count < 0 or self.current_count < 0 or self.search_age < 0:
            raise ValueError(
                "query_count, current_count, and search_age must be non-negative"
            )

    @property
    def visit_key(self) -> tuple[str, str, str]:
        return (
            query_bucket(self.query_count),
            streak_bucket(self.search_streak),
            sequence_bucket(self.search_sequence),
        )

    @property
    def conditional_key(self) -> tuple[Any, ...]:
        return (
            self.position,
            self.repeated_current,
            rank_bucket(self.source_rank),
            self.current,
            self.was_visited,
        )

    @property
    def hard_abstain(self) -> bool:
        return self.query_count >= 10 and self.search_streak == 2


@dataclass(frozen=True)
class LabeledCandidatePattern:
    pattern: CandidatePattern
    next_tool_visit: bool
    exact_match: bool

    def __post_init__(self) -> None:
        if self.exact_match and not self.next_tool_visit:
            raise ValueError("an exact candidate match requires next_tool_visit")


def _shrunk_rate(
    hits: float,
    count: float,
    prior: float,
    strength: float,
) -> float:
    return (hits + strength * prior) / (count + strength)


def _group_rates(
    rows: Sequence[LabeledCandidatePattern],
    *,
    key: Callable[[LabeledCandidatePattern], Hashable],
    label: Callable[[LabeledCandidatePattern], bool],
    prior: Callable[[Hashable], float],
    strength: float,
) -> dict[Hashable, float]:
    counts: dict[Hashable, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        group = key(row)
        counts[group][0] += int(label(row))
        counts[group][1] += 1
    return {
        group: _shrunk_rate(hits, count, prior(group), strength)
        for group, (hits, count) in counts.items()
    }


class CountPatternCalibrator:
    """Hierarchical empirical-Bayes table with deterministic backoff."""

    def __init__(self, rows: Sequence[LabeledCandidatePattern]) -> None:
        examples = tuple(rows)
        if not examples:
            raise ValueError("calibrator requires labeled candidate patterns")

        windows: dict[tuple[str, str], LabeledCandidatePattern] = {}
        for row in examples:
            window_key = (
                row.pattern.session_id,
                row.pattern.decision_id,
            )
            prior = windows.setdefault(window_key, row)
            if prior.next_tool_visit != row.next_tool_visit:
                raise ValueError("one decision has inconsistent next-tool labels")
        window_rows = tuple(windows.values())
        visit_hits = sum(row.next_tool_visit for row in window_rows)
        self.visit_global = (visit_hits + 1.0) / (len(window_rows) + 2.0)
        self.visit_query = _group_rates(
            window_rows,
            key=lambda row: query_bucket(row.pattern.query_count),
            label=lambda row: row.next_tool_visit,
            prior=lambda _: self.visit_global,
            strength=20.0,
        )
        self.visit_detail = _group_rates(
            window_rows,
            key=lambda row: row.pattern.visit_key,
            label=lambda row: row.next_tool_visit,
            prior=lambda group: self.visit_query.get(group[0], self.visit_global),
            strength=20.0,
        )

        visit_candidates = tuple(row for row in examples if row.next_tool_visit)
        conditional_hits = sum(row.exact_match for row in visit_candidates)
        self.conditional_global = (conditional_hits + 1.0) / (
            len(visit_candidates) + 2.0
        )
        self.conditional_position = _group_rates(
            visit_candidates,
            key=lambda row: row.pattern.position,
            label=lambda row: row.exact_match,
            prior=lambda _: self.conditional_global,
            strength=12.0,
        )
        self.conditional_detail = _group_rates(
            visit_candidates,
            key=lambda row: row.pattern.conditional_key,
            label=lambda row: row.exact_match,
            prior=lambda group: self.conditional_position.get(
                group[0], self.conditional_global
            ),
            strength=15.0,
        )

        direct_global = (sum(row.exact_match for row in examples) + 1.0) / (
            len(examples) + 2.0
        )
        self.direct_position = _group_rates(
            examples,
            key=lambda row: row.pattern.position,
            label=lambda row: row.exact_match,
            prior=lambda _: direct_global,
            strength=20.0,
        )
        self.example_count = len(examples)
        self.window_count = len(window_rows)
        self.visit_candidate_count = len(visit_candidates)

    def visit_probability(self, pattern: CandidatePattern) -> float:
        query_prior = self.visit_query.get(
            query_bucket(pattern.query_count), self.visit_global
        )
        return self.visit_detail.get(pattern.visit_key, query_prior)

    def conditional_probability(self, pattern: CandidatePattern) -> float:
        position_prior = self.conditional_position.get(
            pattern.position, self.conditional_global
        )
        return self.conditional_detail.get(
            pattern.conditional_key, position_prior
        )

    def exact_probability(self, pattern: CandidatePattern) -> float:
        if pattern.hard_abstain:
            return 0.0
        return self.visit_probability(pattern) * self.conditional_probability(
            pattern
        )

    def rank_only_probability(self, pattern: CandidatePattern) -> float:
        return self.direct_position.get(
            pattern.position,
            self.visit_global * self.conditional_global,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "hierarchical_empirical_bayes_discrete_patterns",
            "neural_model": False,
            "example_count": self.example_count,
            "window_count": self.window_count,
            "visit_candidate_count": self.visit_candidate_count,
            "visit_global": self.visit_global,
            "conditional_global": self.conditional_global,
            "hard_abstain": "query_count>=10 and search_streak==2",
            "smoothing_strengths": {
                "visit_query": 20.0,
                "visit_detail": 20.0,
                "conditional_position": 12.0,
                "conditional_detail": 15.0,
                "direct_position": 20.0,
            },
        }


@dataclass(frozen=True)
class UtilityCandidate:
    pattern: CandidatePattern
    exact_probability: float
    estimated_service_s: float
    lead_remaining_s: float
    task_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "exact_probability",
            "estimated_service_s",
            "lead_remaining_s",
            "task_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.exact_probability <= 1.0:
            raise ValueError("exact_probability must be in [0, 1]")
        if self.estimated_service_s <= 0.0:
            raise ValueError("estimated_service_s must be positive")
        if self.lead_remaining_s < 0.0 or self.task_weight <= 0.0:
            raise ValueError("lead must be non-negative and task_weight positive")

    @property
    def overlap_s(self) -> float:
        return min(self.estimated_service_s, self.lead_remaining_s)

    def net_utility_s(self, shadow_price: float) -> float:
        probability = self.exact_probability
        return (
            self.task_weight * probability * self.overlap_s
            - (1.0 - probability) * shadow_price * self.estimated_service_s
        )

    def utility_density(self, shadow_price: float) -> float:
        return self.net_utility_s(shadow_price) / self.estimated_service_s


@dataclass(frozen=True)
class AuthorityLoad:
    expected_authoritative_calls: float
    tool_capacity: int
    authoritative_running: int = 0
    authoritative_queued: int = 0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.expected_authoritative_calls)
            or self.expected_authoritative_calls < 0.0
        ):
            raise ValueError("expected_authoritative_calls must be non-negative")
        if self.tool_capacity <= 0:
            raise ValueError("tool_capacity must be positive")
        if self.authoritative_running < 0 or self.authoritative_queued < 0:
            raise ValueError("authoritative counts must be non-negative")

    @property
    def pressure(self) -> float:
        forecast = self.expected_authoritative_calls
        observed = self.authoritative_running + self.authoritative_queued
        return max(forecast, float(observed)) / self.tool_capacity


@dataclass(frozen=True)
class UtilityPolicyConfig:
    idle_pressure: float = 0.5
    medium_pressure: float = 1.0
    high_pressure: float = 2.0
    idle_shadow_price: float = 0.02
    medium_shadow_price: float = 0.05
    high_shadow_price: float = 0.10
    saturated_shadow_price: float = 0.20
    probability_discount: float = 1.0

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.idle_pressure
            <= self.medium_pressure
            <= self.high_pressure
        ):
            raise ValueError("pressure thresholds must be ordered")
        prices = (
            self.idle_shadow_price,
            self.medium_shadow_price,
            self.high_shadow_price,
            self.saturated_shadow_price,
        )
        if any(value < 0.0 or not math.isfinite(value) for value in prices):
            raise ValueError("shadow prices must be finite and non-negative")
        if not 0.0 < self.probability_discount <= 1.0:
            raise ValueError("probability_discount must be in (0, 1]")


@dataclass(frozen=True)
class CandidateUtility:
    candidate: UtilityCandidate
    discounted_probability: float
    shadow_price: float
    net_utility_s: float
    utility_density: float
    selected: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.candidate.pattern.decision_id,
            "session_id": self.candidate.pattern.session_id,
            "url": self.candidate.pattern.url,
            "position": self.candidate.pattern.position,
            "posterior_probability": self.candidate.exact_probability,
            "discounted_probability": self.discounted_probability,
            "shadow_price": self.shadow_price,
            "net_utility_s": self.net_utility_s,
            "utility_density": self.utility_density,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UtilitySelection:
    selected: tuple[CandidateUtility, ...]
    decisions: tuple[CandidateUtility, ...]
    load_pressure: float
    shadow_price: float
    start_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [row.to_dict() for row in self.selected],
            "decisions": [row.to_dict() for row in self.decisions],
            "load_pressure": self.load_pressure,
            "shadow_price": self.shadow_price,
            "start_budget": self.start_budget,
        }


@dataclass(frozen=True)
class SafeStartBudget:
    """Broker-certified starts that cannot consume baseline authority capacity.

    This is deliberately a certificate, not a load forecast.  A caller may
    issue a positive value only for isolated capacity, a separately reserved
    quota, or a resource with a proven zero-delay preemption contract.  Shared
    idle capacity for a non-preemptible call is not safe capacity.
    """

    certified_starts: int
    state_valid: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.certified_starts, bool)
            or not isinstance(self.certified_starts, int)
            or self.certified_starts < 0
        ):
            raise ValueError("certified_starts must be a non-negative integer")
        if not isinstance(self.state_valid, bool):
            raise ValueError("state_valid must be a boolean")

    @property
    def available_starts(self) -> int:
        return self.certified_starts if self.state_valid else 0


@dataclass(frozen=True)
class SafeGlobalBenefitConfig:
    """Small policy surface for the paper's strict no-regression method."""

    probability_discount: float = 1.0
    max_candidates_per_decision: int = 1
    coordination_cost_s: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.probability_discount <= 1.0:
            raise ValueError("probability_discount must be in (0, 1]")
        if (
            isinstance(self.max_candidates_per_decision, bool)
            or not isinstance(self.max_candidates_per_decision, int)
            or self.max_candidates_per_decision <= 0
        ):
            raise ValueError(
                "max_candidates_per_decision must be a positive integer"
            )
        if (
            isinstance(self.coordination_cost_s, bool)
            or not isinstance(self.coordination_cost_s, (int, float))
            or not math.isfinite(float(self.coordination_cost_s))
            or self.coordination_cost_s < 0.0
        ):
            raise ValueError(
                "coordination_cost_s must be finite and non-negative"
            )


class SafeGlobalBenefitPolicy:
    """Lexicographic no-regression admission followed by global benefit rank.

    Safety is not inferred from probability or average load.  The resource
    layer first supplies a :class:`SafeStartBudget`; this policy then chooses
    the candidates with the greatest expected critical-path saving minus a
    fixed, pre-calibrated per-start coordination cost within that hard feasible
    set. Descending expected net benefit is the exact optimum under the
    cardinality and per-decision constraints. Utility density is retained
    separately as the broker's causal dispatch priority.
    """

    def __init__(self, config: SafeGlobalBenefitConfig | None = None) -> None:
        self.config = config or SafeGlobalBenefitConfig()

    @staticmethod
    def _identity(candidate: UtilityCandidate) -> tuple[str, str, str]:
        pattern = candidate.pattern
        return pattern.session_id, pattern.decision_id, pattern.url

    @staticmethod
    def _decision_identity(candidate: UtilityCandidate) -> tuple[str, str]:
        pattern = candidate.pattern
        return pattern.session_id, pattern.decision_id

    @staticmethod
    def _stable_tie_break(candidate: UtilityCandidate) -> str:
        pattern = candidate.pattern
        value = f"{pattern.session_id}\0{pattern.decision_id}\0{pattern.url}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def select(
        self,
        candidates: Sequence[UtilityCandidate],
        *,
        safe_budget: SafeStartBudget,
        requested_start_budget: int | None = None,
    ) -> UtilitySelection:
        if requested_start_budget is not None and (
            isinstance(requested_start_budget, bool)
            or not isinstance(requested_start_budget, int)
            or requested_start_budget < 0
        ):
            raise ValueError(
                "requested_start_budget must be a non-negative integer"
            )
        safe_starts = safe_budget.available_starts
        budget = (
            safe_starts
            if requested_start_budget is None
            else min(safe_starts, requested_start_budget)
        )
        blocked_reason = (
            "invalid_resource_state"
            if not safe_budget.state_valid
            else "no_safe_capacity"
            if budget == 0
            else None
        )

        scored: list[tuple[float, float, str, int, CandidateUtility]] = []
        for input_index, candidate in enumerate(candidates):
            probability = (
                candidate.exact_probability
                * self.config.probability_discount
            )
            gross_benefit = (
                candidate.task_weight * probability * candidate.overlap_s
            )
            net_benefit = gross_benefit - self.config.coordination_cost_s
            density = net_benefit / candidate.estimated_service_s
            scored.append(
                (
                    density,
                    net_benefit,
                    self._stable_tie_break(candidate),
                    input_index,
                    CandidateUtility(
                        candidate=candidate,
                        discounted_probability=probability,
                        shadow_price=0.0,
                        net_utility_s=net_benefit,
                        utility_density=density,
                        selected=False,
                        reason=(
                            blocked_reason
                            or (
                                "nonpositive_net_benefit"
                                if net_benefit <= 0.0
                                else "safe_budget"
                            )
                        ),
                    ),
                )
            )
        scored.sort(key=lambda row: (-row[1], -row[0], row[2]))

        selected_indices: set[int] = set()
        decision_counts: dict[tuple[str, str], int] = defaultdict(int)
        decision_capped: set[int] = set()
        if blocked_reason is None:
            for _, benefit, _, input_index, row in scored:
                decision = self._decision_identity(row.candidate)
                if benefit <= 0.0:
                    continue
                if (
                    decision_counts[decision]
                    >= self.config.max_candidates_per_decision
                ):
                    decision_capped.add(input_index)
                    continue
                if len(selected_indices) >= budget:
                    continue
                selected_indices.add(input_index)
                decision_counts[decision] += 1

        decisions: list[CandidateUtility] = []
        selected: list[CandidateUtility] = []
        for _, _, _, input_index, row in scored:
            if input_index in selected_indices:
                row = CandidateUtility(
                    candidate=row.candidate,
                    discounted_probability=row.discounted_probability,
                    shadow_price=0.0,
                    net_utility_s=row.net_utility_s,
                    utility_density=row.utility_density,
                    selected=True,
                    reason="selected_safe_global_benefit",
                )
                selected.append(row)
            elif input_index in decision_capped:
                row = CandidateUtility(
                    candidate=row.candidate,
                    discounted_probability=row.discounted_probability,
                    shadow_price=0.0,
                    net_utility_s=row.net_utility_s,
                    utility_density=row.utility_density,
                    selected=False,
                    reason="per_decision_cap",
                )
            decisions.append(row)

        return UtilitySelection(
            selected=tuple(selected),
            decisions=tuple(decisions),
            load_pressure=0.0,
            shadow_price=0.0,
            start_budget=budget,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "safe_global_expected_benefit",
            "safety": "broker_certified_noninterfering_start_budget",
            "utility": (
                "task_weight*alpha*p_hat*min(lead,service)"
                "-coordination_cost"
            ),
            "selection": "descending_expected_net_benefit",
            "dispatch_priority": "expected_net_benefit/service",
            "global_not_round_robin": True,
            "config": asdict(self.config),
        }


class AuthorityFirstUtilityPolicy:
    """Global expected-utility selection with an authority backlog kill switch."""

    def __init__(self, config: UtilityPolicyConfig | None = None) -> None:
        self.config = config or UtilityPolicyConfig()

    def shadow_price(self, load: AuthorityLoad) -> float:
        pressure = load.pressure
        if pressure <= self.config.idle_pressure:
            return self.config.idle_shadow_price
        if pressure <= self.config.medium_pressure:
            return self.config.medium_shadow_price
        if pressure <= self.config.high_pressure:
            return self.config.high_shadow_price
        return self.config.saturated_shadow_price

    @staticmethod
    def _stable_tie_break(candidate: UtilityCandidate) -> str:
        value = (
            f"{candidate.pattern.session_id}\0{candidate.pattern.decision_id}"
            f"\0{candidate.pattern.url}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def select(
        self,
        candidates: Sequence[UtilityCandidate],
        *,
        load: AuthorityLoad,
        start_budget: int,
    ) -> UtilitySelection:
        if start_budget < 0:
            raise ValueError("start_budget must be non-negative")
        shadow_price = self.shadow_price(load)
        backlog = load.authoritative_queued > 0
        scored: list[tuple[float, float, str, int, CandidateUtility]] = []
        for input_index, candidate in enumerate(candidates):
            discounted_probability = (
                candidate.exact_probability * self.config.probability_discount
            )
            discounted = UtilityCandidate(
                pattern=candidate.pattern,
                exact_probability=discounted_probability,
                estimated_service_s=candidate.estimated_service_s,
                lead_remaining_s=candidate.lead_remaining_s,
                task_weight=candidate.task_weight,
            )
            net = discounted.net_utility_s(shadow_price)
            density = discounted.utility_density(shadow_price)
            scored.append(
                (
                    density,
                    net,
                    self._stable_tie_break(candidate),
                    input_index,
                    CandidateUtility(
                        candidate=candidate,
                        discounted_probability=discounted_probability,
                        shadow_price=shadow_price,
                        net_utility_s=net,
                        utility_density=density,
                        selected=False,
                        reason=(
                            "authoritative_backlog"
                            if backlog
                            else "nonpositive_utility"
                            if net <= 0.0
                            else "budget"
                        ),
                    ),
                )
            )

        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        selected_indices: set[int] = set()
        if not backlog and start_budget > 0:
            for _, net, _, input_index, _ in scored:
                if net <= 0.0 or len(selected_indices) >= start_budget:
                    continue
                selected_indices.add(input_index)

        decisions: list[CandidateUtility] = []
        selected: list[CandidateUtility] = []
        for _, _, _, input_index, row in scored:
            if input_index in selected_indices:
                row = CandidateUtility(
                    candidate=row.candidate,
                    discounted_probability=row.discounted_probability,
                    shadow_price=row.shadow_price,
                    net_utility_s=row.net_utility_s,
                    utility_density=row.utility_density,
                    selected=True,
                    reason="selected_global_utility",
                )
                selected.append(row)
            decisions.append(row)
        return UtilitySelection(
            selected=tuple(selected),
            decisions=tuple(decisions),
            load_pressure=load.pressure,
            shadow_price=shadow_price,
            start_budget=start_budget,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": "authority_first_expected_net_utility",
            "global_not_round_robin": True,
            "backlog_kill_switch": True,
            "utility": "p*min(lead,service)-(1-p)*shadow_price*service",
            "priority": "utility/service",
            "config": asdict(self.config),
        }
