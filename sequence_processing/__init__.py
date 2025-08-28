"""
Sequence Processing Utilities

This module provides unified sequence generation and logprob computation
functionality across the RL_Practice project, consolidating patterns from
collect_rollouts.py, dr_grpo.py, and other scattered generation code.
"""

from .sequence_processor import SequenceProcessor, BatchedSequences, LogprobResults, GenerationConfig

__all__ = [
    'SequenceProcessor',
    'BatchedSequences', 
    'LogprobResults',
    'GenerationConfig'
]