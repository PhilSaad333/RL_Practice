"""
Entropy Experiments Package

This package implements the offline entropy probe for analyzing entropy changes
in reinforcement learning training, based on the theoretical framework developed
in RL_studies.pdf and the implementation strategy in offline_entropy_probe_strategy.txt.

Main components:
- OfflineEntropyProbe: Main orchestrator class
- ProbeComponents: Core gradient computation and microbatching
- AdamPreconditioner: Extract and apply P from Adam optimizer state
- DeltaEntropyIS: Compute actual ΔH with true model update and importance sampling
- DistributedHelpers: Multi-GPU scalar communication helpers
"""

from .offline_entropy_probe import OfflineEntropyProbe
from .probe_components import ProbeComponents
from .adam_preconditioner import AdamPreconditioner
from .delta_entropy_is import DeltaEntropyIS
from .distributed_helpers import DistributedHelpers

__all__ = [
    'OfflineEntropyProbe',
    'ProbeComponents', 
    'AdamPreconditioner',
    'DeltaEntropyIS',
    'DistributedHelpers'
]