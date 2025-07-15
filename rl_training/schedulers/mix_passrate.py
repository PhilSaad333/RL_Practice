# rl_training/schedulers/mix_passrate.py
"""
Mix-Pass-Rate Scheduler
-----------------------
Yields prompt IDs in a curriculum that emphasises *hard* questions (low pass
rate) more often than *easy* ones (high pass rate), while still keeping a mix
to avoid over-fitting.

Key ideas
~~~~~~~~~
* Maintain an *exponential moving average* (EMA) win-rate for every prompt.
* Categorise each prompt into three buckets by its EMA:
      hard   : win_rate < lo
      normal : lo ≤ win_rate ≤ hi
      easy   : win_rate > hi
* Sample bucket according to configurable weights, then round-robin inside
  that bucket so every prompt in it eventually gets seen.
* After each training step the runner calls `scheduler.update(prompt_id, r̄)`
  with the *mean reward* r̄ (0-1) obtained for that prompt group; this nudges
  the EMA and (if needed) moves the prompt to another bucket.

The scheduler *also* registers the gold answers with the reward function, so
nothing else needs to know about the dataset structure.
"""

from __future__ import annotations
import random
from collections import defaultdict, deque
from typing import Deque, Dict, Iterator, List

from rlp_datasets import DATASET_REGISTRY
from rl_training.rewards.tag_math_correct import set_prompt2gold


class MixPassRateScheduler(Iterator[int]):
    # ──────────────────────────────────────────────────────────────────
    # public config
    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        *,
        ema_alpha: float = 0.05,
        boundaries: tuple[float, float] = (0.1, 0.9),
        sample_weights: tuple[int, int, int] = (2, 5, 1),   # hard:normal:easy
    ):
        """
        Parameters
        ----------
        dataset_name    : key registered in rlp_datasets.DATASET_REGISTRY
        split           : split string understood by the dataset class
        ema_alpha       : smoothing factor for EMA (smaller → slower)
        boundaries      : (lo, hi) thresholds for hard / normal / easy
        sample_weights  : probability weights for (hard, normal, easy)
        """
        ds_cls = DATASET_REGISTRY[dataset_name]
        self.dataset = list(ds_cls(split))

        mapping = {
            hash(ex.question) & 0xFFFFFFFF: ex.answer
            for ex in self.dataset
            if hasattr(ex, "answer")
        }
        self.id2text = {
            hash(ex.question) & 0xFFFFFFFF: ex.question
            for ex in self.dataset
        }
        
        set_prompt2gold(mapping)

        self.ema_alpha = ema_alpha
        self.lo, self.hi = boundaries
        self.weights = sample_weights

        self.win_rate: Dict[int, float] = defaultdict(lambda: 0.5)
        self.buckets: Dict[str, Deque[int]] = {
            "hard": deque(),
            "normal": deque(),
            "easy": deque(),
        }
        self._rebuild_all_buckets()

    # ------------------------------------------------------------------
    # iterator protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> "MixPassRateScheduler":
        return self

    def __next__(self) -> int:
        """Return *prompt_id* for the next rollout."""
        bucket = random.choices(
            population=("hard", "normal", "easy"),
            weights=self.weights,
        )[0]

        if not self.buckets[bucket]:
            for alt in ("normal", "hard", "easy"):
                if self.buckets[alt]:
                    bucket = alt
                    break
            else:
                raise StopIteration("All buckets empty - dataset exhausted?")

        prompt_id = self.buckets[bucket].popleft()
        self.buckets[bucket].append(prompt_id)
        return prompt_id

    # ------------------------------------------------------------------
    # feedback after each prompt group is evaluated
    # ------------------------------------------------------------------
    def update(self, prompt_id: int, mean_reward: float) -> None:
        """
        mean_reward : scalar in [0, 1] : average over the G generations.
        """
        # 1. update EMA
        old = self.win_rate[prompt_id]
        new = (1 - self.ema_alpha) * old + self.ema_alpha * mean_reward
        self.win_rate[prompt_id] = new

        # 2. move prompt to new bucket if boundary crossed
        self._move_prompt_if_needed(prompt_id, new)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _bucket_name(self, rate: float) -> str:
        if rate < self.lo:
            return "hard"
        if rate > self.hi:
            return "easy"
        return "normal"

    def _rebuild_all_buckets(self) -> None:
        """Called once at init."""
        for ex in self.dataset:
            pid = hash(ex.text) & 0xFFFFFFFF
            bucket = self._bucket_name(self.win_rate[pid])
            self.buckets[bucket].append(pid)

    def _move_prompt_if_needed(self, prompt_id: int, new_rate: float) -> None:
        new_bucket = self._bucket_name(new_rate)
        # if already in correct bucket, nothing to do
        if prompt_id in self.buckets[new_bucket]:
            return
        # remove from any bucket it’s currently in
        for dq in self.buckets.values():
            try:
                dq.remove(prompt_id)
            except ValueError:
                pass
        # append to tail of its new bucket
        self.buckets[new_bucket].append(prompt_id)


# ----------------------------------------------------------------------
# factory for the collector (expects cfg["scheduler"]["name"]="mix_passrate")
# ----------------------------------------------------------------------
def get_prompt_sampler(cfg: dict) -> MixPassRateScheduler:
    """
    Collector passes `cfg["scheduler"]` here.
    Example YAML cfg section:

        scheduler:
            name: mix_passrate
            dataset_name: gsm8k
            split: train[:1%]
            ema_alpha: 0.05
            sample_weights: [4, 3, 1]
            boundaries: [0.2, 0.8]
    """
    params = cfg.copy()          # shallow copy
    params.pop("name", None)     # remove the key the collector added
    return MixPassRateScheduler(**params)
