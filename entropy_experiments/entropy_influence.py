"""Skeleton for entropy influence experiments.

This module will mirror the Fisher kernel runner but computes true entropy
changes (via SNIS) for both aggregate update vectors and per-sequence
contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from entropy_experiments.fisher_kernel import BatchRequest, WorkspaceSpec


@dataclass
class EntropyInfluencePlan:
    """Describe an entropy influence experiment run."""

    workspace: WorkspaceSpec
    evaluation_requests: List[BatchRequest] = field(default_factory=list)
    etas: List[float] = field(default_factory=list)
    per_sequence_etas: Optional[List[float]] = None
    microbatch_size: int = 4
    auto_scale: bool = False
    auto_scale_target: float = 1e-6


@dataclass
class AggregateEntropyResult:
    """Placeholder for aggregate entropy change diagnostics."""

    eta: float
    delta_h: Optional[float] = None
    per_sequence_delta: Optional[List[float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerSequenceEntropyResult:
    """Placeholder structure for per-U-sequence entropy contributions."""

    eta_used: float
    delta_matrix: Optional[List[List[float]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntropyInfluenceResults:
    """Container for the outputs of an entropy influence run."""

    plan: EntropyInfluencePlan
    workspace_metadata: Dict[str, Any] = field(default_factory=dict)
    aggregate_results: List[AggregateEntropyResult] = field(default_factory=list)
    per_sequence_result: Optional[PerSequenceEntropyResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntropyInfluenceRunner:
    """Coordinate entropy influence experiments (skeleton)."""

    def __init__(self, config: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger
        # SampleGenerator / optimizer handles will be loaded lazily in future work.

    def run(self, plan: EntropyInfluencePlan) -> EntropyInfluenceResults:
        """Execute the entropy influence workflow (to be implemented)."""

        raise NotImplementedError

