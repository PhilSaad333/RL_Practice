#!/usr/bin/env python3
"""Colab-friendly smoke test for SampleGenerator.

This script exercises the key entry points provided by
`entropy_experiments.utils.sample_generator.SampleGenerator` so we can sanity
check new batch-sampling features on a GPU instance (e.g. Google Colab).

It reuses the standard entropy experiment config, but keeps batch sizes small so
we can validate behaviour quickly without heavy compute.
"""

from __future__ import annotations

import argparse
import pprint
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import yaml

from entropy_experiments.utils.sample_generator import GeneratedBatch, SampleGenerator


def _shorten(seq: Sequence[int], max_len: int = 8) -> str:
    if len(seq) <= max_len:
        return str(list(seq))
    head = list(seq[: max_len // 2])
    tail = list(seq[-max_len // 2 :])
    return f"{head} ... {tail} (len={len(seq)})"


def _record_summary(record_dict: Dict[str, Any]) -> Dict[str, Any]:
    summary = record_dict.copy()
    summary["prompt_tokens"] = _shorten(record_dict.get("prompt_tokens", []))
    summary["response_tokens"] = _shorten(record_dict.get("response_tokens", []))
    summary["logprob_per_token"] = _shorten(record_dict.get("logprob_per_token", []))
    summary["entropy_per_token"] = _shorten(record_dict.get("entropy_per_token", []))
    summary["logq_per_token"] = _shorten(record_dict.get("logq_per_token", []))
    return summary


def summarize_batch(batch: GeneratedBatch, *, label: str, max_records: int = 2) -> None:
    print("\n" + "=" * 80)
    print(f"{label} :: batch_type={batch.batch_type}  num_sequences={len(batch.sequences)}")
    if batch.full_sequence_tensor is not None:
        print(f"  full_sequence_tensor.shape = {tuple(batch.full_sequence_tensor.shape)}")
    if batch.attention_mask is not None:
        print(f"  attention_mask.shape      = {tuple(batch.attention_mask.shape)}")
    if batch.rewards is not None:
        print(f"  rewards.shape              = {tuple(batch.rewards.shape)}")
    if batch.advantages is not None:
        print(f"  advantages.shape           = {tuple(batch.advantages.shape)}")
    print("  sampling_metadata:")
    pprint.pprint(batch.sampling_metadata, indent=4)

    for idx, record in enumerate(batch.sequences[:max_records]):
        record_dict = asdict(record)
        print(f"  -- SequenceRecord[{idx}] :: sequence_id={record.sequence_id}")
        summary = _record_summary(record_dict)
        pprint.pprint(summary, indent=8)

    print("=" * 80 + "\n")


def collect_global_prompt_ids(batch: GeneratedBatch, limit: int = 3) -> List[int]:
    ids: List[int] = []
    for record in batch.sequences:
        gid = record.global_prompt_id
        if gid is not None and gid not in ids:
            ids.append(int(gid))
        if len(ids) >= limit:
            break
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SampleGenerator smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default="entropy_experiments/configs/config_template.yaml",
        help="Path to YAML config used for loading model/checkpoint",
    )
    parser.add_argument(
        "--update-prompts",
        type=int,
        default=2,
        help="Number of prompts in the update batch",
    )
    parser.add_argument(
        "--update-completions",
        type=int,
        default=2,
        help="Number of completions per prompt in the update batch",
    )
    parser.add_argument(
        "--evaluation-prompts",
        type=int,
        default=3,
        help="Number of prompts in the evaluation batch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Expected torch device (warn if unavailable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Global seed for deterministic sampling where relevant",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    if args.device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA expected but not available; continuing on CPU.")

    config = load_config(args.config)
    batch_cfg = config.get("batch_config", {}) or {}
    u_split = batch_cfg.get("U_split", batch_cfg.get("split", "train"))
    e_split = batch_cfg.get("E_split", batch_cfg.get("split", "train"))

    # Make sure generation defaults stay lightweight for smoke testing.
    config.setdefault("generation", {})
    config["generation"].setdefault("max_new_tokens", 96)
    config["generation"].setdefault("gen_batch_size", 16)
    config["generation"].setdefault("tf_batch_size", 8)

    print("[INFO] Instantiating SampleGenerator ...")
    generator = SampleGenerator(config, logger=None)
    generator._lazy_load_resources()  # The subsequent calls would trigger this anyway.
    print("[INFO] Resources loaded successfully.")

    print("[STEP] Generating update batch")
    update_batch = generator.generate_update_batch(
        batch_size_prompts=args.update_prompts,
        completions_per_prompt=args.update_completions,
        dataset_split=u_split,
        seed=args.seed,
    )
    summarize_batch(update_batch, label="Update batch")

    print("[STEP] Generating evaluation batch (with replacement)")
    eval_batch_repl = generator.generate_evaluation_batch(
        batch_size_prompts=args.evaluation_prompts,
        completions_per_prompt=1,
        with_replacement=True,
        dataset_split=e_split,
        seed=args.seed,
    )
    summarize_batch(eval_batch_repl, label="Evaluation batch (with replacement)")

    print("[STEP] Generating evaluation batch (without replacement)")
    eval_batch_norepl = generator.generate_evaluation_batch(
        batch_size_prompts=min(args.evaluation_prompts, args.update_prompts),
        completions_per_prompt=1,
        with_replacement=False,
        dataset_split=e_split,
        seed=args.seed,
    )
    summarize_batch(eval_batch_norepl, label="Evaluation batch (without replacement)")

    prompt_ids = collect_global_prompt_ids(update_batch)
    if prompt_ids:
        print(f"[STEP] Fetching prompts by global IDs: {prompt_ids}")
        prompt_id_batch = generator.generate_from_prompt_ids(
            prompt_ids,
            dataset_split=u_split,
            completions_per_prompt=1,
            seed=args.seed,
        )
        summarize_batch(prompt_id_batch, label="Prompt-ID batch")
    else:
        print("[WARN] Update batch did not expose global prompt ids; skipping prompt-id test.")

    custom_prompts = [
        "Alice has 7 apples, gives away 3, how many remain?",
        "A rectangle has sides 4 and 9. Compute its area.",
    ]
    print("[STEP] Sampling from custom prompts")
    custom_prompt_batch = generator.generate_from_custom_prompts(
        custom_prompts,
        completions_per_prompt=1,
        seed=args.seed,
        apply_template=True,
    )
    summarize_batch(custom_prompt_batch, label="Custom prompts batch")

    print("[STEP] Teacher-forcing a custom prompt/response pair")
    custom_sequence_batch = generator.build_custom_sequence(
        prompt="What is 2 + 2?",
        response="<think> Add the numbers. 2 + 2 = 4. </think><answer>4</answer>",
        metadata={"reward": 1.0},
        apply_template=True,
    )
    summarize_batch(custom_sequence_batch, label="Custom sequence batch")

    print("[DONE] SampleGenerator smoke test completed successfully.")


if __name__ == "__main__":  # pragma: no cover
    main()
