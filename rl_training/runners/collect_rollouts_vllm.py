# rl_training/runners/collect_rollouts_vllm.py
from __future__ import annotations
import re, json, math, time, tempfile, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase,
    LogitsProcessor, LogitsProcessorList
)

# VLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm")
    VLLM_AVAILABLE = False

from rl_training.utils.rollout_buffer import RolloutBuffer, RolloutBatch
from rl_training.rewards import get_reward_fns
from importlib import import_module

TAG_STOP = "</answer>"

# Import original dataclass and helper functions
from rl_training.runners.collect_rollouts import (
    GenSample, _accept_prompt_group, _count_think_tokens, 
    _append_jsonl, _next_prompt_batch, _unwrap
)


# --------------------------------------------------------------------------
# VLLM-based RolloutCollector
# --------------------------------------------------------------------------
class VLLMRolloutCollector:
    """
    VLLM-based rollout collector for faster inference.
    
    Key differences from standard collector:
    - Uses VLLM for batched generation (much faster)
    - Handles online model updates by periodically reloading VLLM engine
    - Supports LoRA adapters
    """
    def __init__(
        self,
        policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: Dict[str, Any],
        out_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        vllm_reload_every: int = 10,  # Reload VLLM engine every N steps
        vllm_tensor_parallel: int = 1,  # Number of GPUs for VLLM
        vllm_gpu_memory_utilization: float = 0.4,  # Leave room for training
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is required for VLLMRolloutCollector")
            
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg
        
        # VLLM configuration
        self.vllm_reload_every = vllm_reload_every
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Device handling
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        assert tokenizer.padding_side == "left"
        self.pad_id = tokenizer.pad_token_id
        self.G = cfg["num_generations"]
        self.B_opt = cfg["microbatch_size"]
        self.batch_size = cfg.get("rollout_batch_size", self.B_opt)

        # Setup output directory
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directory for model snapshots
        self.temp_model_dir = None
        
        # VLLM engine state
        self.vllm_engine: Optional[LLM] = None
        self.last_reload_step = -1
        
        # Original collector components
        self.reward_fns = get_reward_fns(cfg["reward_fns"])
        sched_cfg = cfg["scheduler"]
        sched_mod = import_module(f"rl_training.schedulers.{sched_cfg['name']}")
        self.prompt_sampler = sched_mod.get_prompt_sampler(sched_cfg)
        
        # Bookkeeping
        self.win_rate_ema: Dict[int, float] = {}
        self._step_idx = 0
        
        # Trace file
        trace_name = f"rollouts_rank{self.rank}.jsonl"
        self._trace_file = self.out_dir / trace_name

    def _get_base_model_name_or_path(self) -> str:
        """Extract base model name from the policy model."""
        unwrapped = _unwrap(self.policy)
        
        # Try different attributes where base model path might be stored
        for attr in ['name_or_path', '_name_or_path', 'config.name_or_path']:
            if hasattr(unwrapped, attr):
                path = getattr(unwrapped, attr)
                if path:
                    return path
        
        # Fallback: check config
        if hasattr(unwrapped, 'config'):
            config = unwrapped.config
            for attr in ['_name_or_path', 'name_or_path']:
                if hasattr(config, attr):
                    path = getattr(config, attr)
                    if path:
                        return path
        
        # Last resort: use a default model name
        print("Warning: Could not determine base model name, using default")
        return "Qwen/Qwen2.5-1.5B"

    def _save_current_model(self) -> Path:
        """Save current model state to temporary directory for VLLM loading."""
        if self.temp_model_dir is None:
            self.temp_model_dir = Path(tempfile.mkdtemp(prefix="vllm_model_"))
        
        # Clean up old model if exists
        if self.temp_model_dir.exists():
            shutil.rmtree(self.temp_model_dir)
        self.temp_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the current model state
        unwrapped = _unwrap(self.policy)
        
        # For LoRA models, we need to save the adapter weights
        if hasattr(unwrapped, 'save_pretrained'):
            unwrapped.save_pretrained(self.temp_model_dir)
        else:
            # Fallback: save state dict
            torch.save(unwrapped.state_dict(), self.temp_model_dir / "pytorch_model.bin")
        
        # Save tokenizer as well
        self.tokenizer.save_pretrained(self.temp_model_dir)
        
        return self.temp_model_dir

    def _reload_vllm_engine(self):
        """Reload VLLM engine with current model weights."""
        print(f"[VLLM] Rank {self.rank}: Reloading VLLM engine (step {self._step_idx})")
        
        # Clean up old engine
        if self.vllm_engine is not None:
            del self.vllm_engine
            torch.cuda.empty_cache()
        
        # Save current model
        model_path = self._save_current_model()
        
        # Determine if we're using LoRA
        base_model = self._get_base_model_name_or_path()
        
        # VLLM engine configuration
        vllm_kwargs = {
            "tensor_parallel_size": self.vllm_tensor_parallel,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": True,  # Helps with memory management
        }
        
        # Try to detect if this is a LoRA adapter
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            # This is a LoRA adapter
            print(f"[VLLM] Loading LoRA adapter from {model_path}")
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64  # Adjust based on your LoRA rank
            
            # Load with base model + LoRA
            self.vllm_engine = LLM(model=base_model, **vllm_kwargs)
        else:
            # Full model
            print(f"[VLLM] Loading full model from {model_path}")
            self.vllm_engine = LLM(model=str(model_path), **vllm_kwargs)
        
        self.last_reload_step = self._step_idx
        print(f"[VLLM] Engine reloaded successfully")

    def _should_reload_vllm(self) -> bool:
        """Check if VLLM engine should be reloaded."""
        return (
            self.vllm_engine is None or
            self._step_idx - self.last_reload_step >= self.vllm_reload_every
        )

    def _generate_with_vllm(self, prompts: List[str]) -> List[List[str]]:
        """Generate responses using VLLM."""
        if self._should_reload_vllm():
            self._reload_vllm_engine()
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            n=self.G,  # num_return_sequences
            temperature=float(self.cfg["temperature"]),
            max_tokens=int(self.cfg["max_new_tokens"]),
            stop=[TAG_STOP],  # Stop at </answer>
            logprobs=1,  # We need logprobs for entropy calculation
        )
        
        # Handle LoRA request if needed
        lora_request = None
        if hasattr(self.vllm_engine.llm_engine.model_config, 'enable_lora'):
            # Create LoRA request if using adapters
            adapter_path = self.temp_model_dir
            if (adapter_path / "adapter_config.json").exists():
                lora_request = LoRARequest("current_adapter", 1, str(adapter_path))
        
        # Generate
        if lora_request:
            outputs = self.vllm_engine.generate(prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = self.vllm_engine.generate(prompts, sampling_params)
        
        # Convert outputs to expected format
        results = []
        for output in outputs:
            completions = [completion.text for completion in output.outputs]
            # Add TAG_STOP if not present (VLLM strips stop tokens)
            completions = [comp + TAG_STOP if not comp.endswith(TAG_STOP) else comp 
                          for comp in completions]
            results.append(completions)
        
        return results

    def _extract_logprobs_from_vllm(self, prompts: List[str]) -> List[List[List[float]]]:
        """Extract token logprobs from VLLM outputs for entropy calculation."""
        # This is a simplified version - in practice you'd want to extract
        # the actual logprobs from VLLM's detailed outputs
        
        # For now, return dummy logprobs - you'd need to modify _generate_with_vllm
        # to return both text and logprobs
        dummy_logprobs = []
        for prompt in prompts:
            prompt_logprobs = []
            for g in range(self.G):
                # Dummy logprobs - replace with actual VLLM logprob extraction
                token_logprobs = [-1.0] * 50  # Placeholder
                prompt_logprobs.append(token_logprobs)
            dummy_logprobs.append(prompt_logprobs)
        
        return dummy_logprobs

    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        """Main collection method using VLLM for generation."""
        need = batch_prompts or self.batch_size
        buffer = RolloutBuffer(capacity=need, pad_id=self.pad_id)
        ans_pat = re.compile(r"</think>\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)

        bar = tqdm(total=need, desc="Collecting rollouts (VLLM)", leave=False, 
                  disable=(self.rank != 0))

        while len(buffer) < need:
            # 1) Sample prompt mini-batch
            take = min(self.batch_size, need - len(buffer))
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler, self.tokenizer, self.device, take
            )

            # 2) Generate with VLLM (much faster than standard generation)
            print(f"[DEBUG] Rank {self.rank}: Starting VLLM generation")
            t0 = time.time()
            
            # VLLM expects just the text, not tokenized input
            prompt_texts = [txt for txt in ptxts]
            generation_results = self._generate_with_vllm(prompt_texts)
            
            gen_time = time.time() - t0
            print(f"[DEBUG] Rank {self.rank}: Completed VLLM generation, took {gen_time:.2f}s")

            # 3) Process results (similar to original but adapted for VLLM output)
            before = len(buffer)
            
            for b in range(take):
                if len(buffer) >= need:
                    break
                    
                print(f"[DEBUG] Rank {self.rank}: Processing sample {b+1}/{take}, buffer has {len(buffer)}/{need}")
                
                pid = pids[b]
                q_text = ptxts[b]
                g_txts = generation_results[b]  # List of G generated texts
                
                # Tokenize the generated texts to get IDs and logprobs
                g_ids_list = []
                lp_rows = []
                ent_rows = []
                
                for g_text in g_txts:
                    # Tokenize generation (without the prompt)
                    # This is approximate - ideally we'd get this directly from VLLM
                    gen_tokens = self.tokenizer(
                        g_text, add_special_tokens=False, return_tensors="pt"
                    ).input_ids[0]
                    
                    g_ids_list.append(gen_tokens)
                    
                    # Dummy logprobs - in practice, extract from VLLM
                    dummy_lp = torch.full((len(gen_tokens),), -1.0, device=self.device)
                    lp_rows.append(dummy_lp)
                    ent_rows.append(-dummy_lp)  # entropy = -logprob

                # Pad to same length
                if g_ids_list:
                    g_ids_t = pad(g_ids_list, batch_first=True, padding_value=self.pad_id)
                    lp_t = pad(lp_rows, batch_first=True, padding_value=0.0)
                    ent_t = pad(ent_rows, batch_first=True, padding_value=0.0)
                else:
                    continue

                # 4) Rewards & acceptance (same as original)
                r_vec = torch.stack([fn(pid, g_txts) for fn in self.reward_fns]).sum(0)
                accept = _accept_prompt_group(
                    r_vec,
                    thresh=self.cfg["reward_var_thresh"],
                    allow_all_zero=not self.cfg["reject_all_zero"],
                    allow_all_max=not self.cfg["reject_all_max"],
                )
                
                succ = (r_vec > 0).float().mean().item()
                prev = self.win_rate_ema.get(pid, succ)
                self.win_rate_ema[pid] = 0.95 * prev + 0.05 * succ
                diff_tag = ("easy" if self.win_rate_ema[pid] > 0.8 else
                           "hard" if self.win_rate_ema[pid] < 0.2 else "normal")

                tag_ok = torch.tensor([bool(re.search(ans_pat, t)) for t in g_txts],
                                     dtype=torch.float32, device=self.device)
                t_len = torch.tensor([_count_think_tokens(t, self.tokenizer) for t in g_txts],
                                    dtype=torch.int32, device=self.device)

                # 5) Trace JSONL (same as original)
                samples = []
                for g in range(len(g_txts)):
                    seq_lp = float(lp_t[g].sum().item()) if g < len(lp_t) else 0.0
                    samples.append(GenSample(
                        prompt_id=pid,
                        prompt_text=q_text,
                        gen_text=g_txts[g],
                        think_len=int(t_len[g]) if g < len(t_len) else 0,
                        reward=float(r_vec[g]) if g < len(r_vec) else 0.0,
                        tag_correct=float(tag_ok[g]) if g < len(tag_ok) else 0.0,
                        include_in_batch=bool(accept),
                        difficulty_tag=diff_tag,
                        token_entropy=(-lp_t[g]).tolist() if g < len(lp_t) else [],
                        token_logprob=lp_t[g].tolist() if g < len(lp_t) else [],
                        generation_time_s=gen_time / len(g_txts),  # Approximate
                        step_idx=self._step_idx,
                        seq_logprob=seq_lp,
                    ))
                _append_jsonl(self._trace_file, samples)

                # 6) Add to buffer (same as original)
                if accept and len(buffer) < need:
                    buffer.add(
                        prompt_ids=prompt_ids[b].cpu(),
                        gen_ids=g_ids_t.cpu(),
                        rewards=r_vec.cpu(),
                        logprobs=lp_t.cpu(),
                        tag_correct=tag_ok.cpu(),
                        think_len=t_len.cpu(),
                    )

            bar.update(len(buffer) - before)

        bar.close()
        self._step_idx += 1
        return buffer

    def cleanup(self):
        """Clean up VLLM engine and temporary files."""
        if self.vllm_engine is not None:
            del self.vllm_engine
            torch.cuda.empty_cache()
        
        if self.temp_model_dir and self.temp_model_dir.exists():
            shutil.rmtree(self.temp_model_dir)


# Convenience function to create the appropriate collector
def create_rollout_collector(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    cfg: Dict[str, Any],
    out_dir: str | Path,
    use_vllm: bool = False,
    **vllm_kwargs
):
    """Factory function to create either standard or VLLM-based collector."""
    if use_vllm:
        if not VLLM_AVAILABLE:
            print("Warning: VLLM not available, falling back to standard collector")
            from rl_training.runners.collect_rollouts import RolloutCollector
            return RolloutCollector(policy, tokenizer, cfg, out_dir)
        else:
            return VLLMRolloutCollector(policy, tokenizer, cfg, out_dir, **vllm_kwargs)
    else:
        from rl_training.runners.collect_rollouts import RolloutCollector
        return RolloutCollector(policy, tokenizer, cfg, out_dir)