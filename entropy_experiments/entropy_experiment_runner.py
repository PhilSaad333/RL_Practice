"""
Main orchestrator class for the offline entropy experiments.

Key functionality:
- Sample batches of prompts and responses - E batch for "evaluation" or "entropy", U batch for "update"
- Compute the normalized update vector - the parameter updates from doing an RL update with the U batch, normalized by learning rate
- For various learning rates eta:  
    - Compute first-order entropy change prediction after updating with eta*update_vector, using E batch to estimate entropy gradient
    - Measure actual entropy change via importance sampling (entropy estimated on E batch before/after update with)

Usage:
    probe = EntropyMeasurements(config)
    results = probe.run_experiments()
"""

# === Model and optimizer loading ===
from entropy_experiments.utils.model_loader import load_peft_for_probe, load_adam_optimizer_from_path

import torch

# from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time
import math
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import json

# import torch.distributed as dist
from entropy_experiments.utils.sequence_processor import (
    SequenceProcessor, GenerationConfig, BatchedSequences,
)
from entropy_experiments.delta_entropy_approx import DeltaEntropyApprox
from entropy_experiments.delta_entropy_true import DeltaEntropyTrue

from entropy_experiments.update_vector import compute_update_vector
from entropy_experiments.utils.param_overrides import build_functional_params_named

from entropy_experiments.utils.precision_utils import apply_global_precision, str_to_dtype

from entropy_experiments.utils.detailed_logger import DetailedLogger


class EntropyMeasurements:
    """
    Main method is run_mixed_probe, at the end of the class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary matching configs/config_template.yaml
        """
        self.config = config
        
        # Apply global precision settings once at initialization
        pcfg = config.get('precision', {})
        apply_global_precision(
            allow_tf32=pcfg.get('allow_tf32', False),
            matmul_precision=pcfg.get('matmul_precision', 'high')
        )
        
        self.logger = self._setup_logging()
        
        # Single GPU mode for probe (from fix.txt)
        #self.SINGLE_GPU = os.getenv("ENTROPY_PROBE_SINGLE_GPU", "1") == "1"
        
        # Initialize distributed if needed (disabled in single GPU mode)
        #if self.SINGLE_GPU:
            # Skip DDP entirely for the probe
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        #else: #NOT IMPLEMENTED
        #    self.distributed = dist.is_initialized()
        #    self.rank = dist.get_rank() if self.distributed else 0
        #    self.world_size = dist.get_world_size() if self.distributed else 1
        
        # Initialize components
        self._sequence_processor = None
        self.delta_entropy_approx = None
        self.delta_entropy_true = None
        self.distributed_helpers = None
        
        # Model and data
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.dataset = None
        
        # Probe state
        self.checkpoint_loaded = False
        self.results = {}
        
        # Initialize detailed logger if enabled
        if self.config.get('detailed_logging', {}).get('enabled', False):
            self.detailed_logger = DetailedLogger(self.config, self.logger)
        else:
            self.detailed_logger = None
        
        self.logger.info(f"Initialized OfflineEntropyProbe on rank {self.rank}/{self.world_size}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with both console and file output."""
        logger = logging.getLogger(f"entropy_probe_rank_{self.rank if hasattr(self, 'rank') else 0}")
        logger.setLevel(getattr(logging, self.config['output']['log_level']))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.rank if hasattr(self, "rank") else 0}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler - save to logs directory alongside detailed logs
            if self.config.get('detailed_logging', {}).get('enabled', False):
                from datetime import datetime
                from pathlib import Path
                
                # Use same directory structure as DetailedLogger
                output_dir = Path(self.config.get('detailed_logging', {}).get('output_directory', 'entropy_experiments/logs'))
                day_dir = output_dir / datetime.now().strftime('%Y-%m-%d')
                day_dir.mkdir(parents=True, exist_ok=True)
                
                # Create log file with timestamp
                timestamp_str = datetime.now().strftime('%H-%M-%S')
                log_file = day_dir / f"entropy_probe_{timestamp_str}_console.log"
                
                file_handler = logging.FileHandler(log_file, mode='w')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                # Store log file path for reference
                self._console_log_file = str(log_file)
                logger.info(f"Console logging to file: {log_file}")
            
        return logger
        
    def load_checkpoint(self, adapter_path: str, optimizer_path: Optional[str] = None):
        # 1) Load PEFT model (fp32) and tokenizer
        self.model, self.tokenizer = load_peft_for_probe(
            base_id=self.config['checkpoint']['backbone'],
            adapter_path=adapter_path,
            device_map=self.config.get('device', 'cuda'),
            force_fp32_runtime=True,  # your model_loader already does this
        )

        # (Optional) sanity log of example LoRA weight dtypes
        try:
            for n, p in self.model.named_parameters():
                if ('lora_A' in n or 'lora_B' in n) and p.requires_grad:
                    self.logger.info(f"[dtype audit] {n} dtype: {p.dtype}")
                    break
        except Exception:
            pass

        # 2) (Re)create the SP with clean precision & generation config
        self._ensure_sequence_processor()

        # 3) Load optimizer if provided
        if optimizer_path:
            self.optimizer = load_adam_optimizer_from_path(
                self.model,
                optimizer_path=optimizer_path
            )

        self.checkpoint_loaded = True
        
            
          
        
    
    # --- SequenceProcessor setup and sampling helpers ---
    def _sp_precision_config(self) -> dict:
        """
        Return the exact precision dict to pass into SequenceProcessor(config=...).

        - runtime is forced to fp32 (no AMP)
        - entropy/log-softmax in {fp32, fp64} based on config
        - no per-call mutation of SP internals elsewhere
        """
        pcfg = (self.config.get('precision', {}) or {})
        runtime_dtype = str(pcfg.get('runtime_dtype', 'float32')).lower()
        entropy_dtype = str(pcfg.get('entropy_dtype', 'float32')).lower()

        # global knobs honored by SP
        allow_tf32 = bool(pcfg.get('allow_tf32', False))
        matmul_precision = pcfg.get('matmul_precision', 'high')

        # SP expects these trees; keep them lean and explicit
        sp_prec = {
            "allow_tf32": allow_tf32,
            "matmul_precision": matmul_precision,

            # functional_call override path
            "func_override": {
                "autocast": False,
                "dtype": runtime_dtype,   # parameters mapped to this dtype
                "cast_params": True,
            },

            # teacher-forcing no-grad (used for E/U batches; no AMP)
            "tf_nograd": {
                "autocast": False,
                "dtype": runtime_dtype,
                "cast_logits_fp32": True,  # log-softmax & diagnostics in fp32 minimum
            },

            # single toggle that SP interprets when computing entropies
            "entropy_fp64": (entropy_dtype == "float64"),
        }
        return sp_prec
   

    def _ensure_sequence_processor(self):
        """
        Instantiate SequenceProcessor with the model/tokenizer and a clean
        config payload that includes {generation, precision}. No mutation of
        SP internals outside this method.
        """
        assert self.model is not None and self.tokenizer is not None, \
            "Model/tokenizer must be loaded before constructing SequenceProcessor."

        # Generation config — keep it minimal & explicit
        gen_cfg = (self.config.get("generation", {}) or {}).copy()
        gen_cfg.setdefault("max_new_tokens", 256)
        gen_cfg.setdefault("temperature", 1.0)
        gen_cfg.setdefault("top_p", 1.0)  # force deterministic tail unless you really need otherwise

        sp_cfg = {
            "generation": gen_cfg,
            "precision": self._sp_precision_config(),
        }

        # Recreate SP each time model changes (simplest & safest)
        self._sequence_processor = SequenceProcessor(
            self.model,
            self.tokenizer,
            logger=self.logger,
            config=sp_cfg,
        )

    def _pack_E_from_sequences(self, seqs: BatchedSequences) -> dict:
        import torch as _torch
        B, G = seqs.sequences.shape[:2]
        assert G == 1, f"E-batch expected G=1, got G={G}"
        max_lengths = [max(lens) if len(lens) > 0 else 1 for lens in seqs.gen_lens]
        advantages = _torch.zeros((B, G), dtype=_torch.float32, device=seqs.sequences.device)
        return {
            'sequences': seqs.sequences,
            'attention_masks': seqs.attention_masks,
            'prompt_lens': seqs.prompt_lens,
            'advantages': advantages,
            'max_lengths': max_lengths,
            'gen_lens': seqs.gen_lens,  # Store the original gen_lens for reconstruction
            'num_prompts': B,
            'num_responses_per_prompt': 1,
        }

    def _pack_U_from_sequences(self, seqs: BatchedSequences, rewards: list[list[float]]) -> dict:
        import torch as _torch
        B, G = seqs.sequences.shape[:2]
        advantages = _torch.tensor(rewards, dtype=_torch.float32, device=seqs.sequences.device)
        advantages = advantages - advantages.mean(dim=1, keepdim=True)
        max_lengths = [max(lens) if len(lens) > 0 else 1 for lens in seqs.gen_lens]
        return {
            'sequences': seqs.sequences,
            'attention_masks': seqs.attention_masks,
            'prompt_lens': seqs.prompt_lens,
            'advantages': advantages,
            'max_lengths': max_lengths,
            'num_prompts': B,
            'num_responses_per_prompt': G,
        }

    def _sample_EU_via_sequence_processor(self, *, B_E: int, B_U: int, G_U: int) -> tuple[dict, dict]:
        self._ensure_sequence_processor()
        ds_name = self.config['batch_config']['dataset_name']
        
        # Support separate E_split and U_split, with backward compatibility
        batch_config = self.config['batch_config']
        E_split = batch_config.get('E_split', batch_config.get('split', 'test'))
        U_split = batch_config.get('U_split', batch_config.get('split', 'test'))
        
        # Log split configuration for clarity
        if 'E_split' in batch_config or 'U_split' in batch_config:
            self.logger.info(f"Using separate splits: E_split='{E_split}', U_split='{U_split}'")
        else:
            self.logger.info(f"Using single split: '{E_split}' for both E and U batches")

        # E: with replacement, G=1, compute RB for X if needed later
        # Log E-generation config reminder (top_p forced to 1.0)
        try:
            self.logger.info(
                f"[IS] E-generation: top_p forced to 1.0; temperature={self._sequence_processor.config.temperature}"
            )
        except Exception:
            pass
        E_sequences, _E_lp, _E_diag = self._sequence_processor.generate_with_replacement_sampling(
            total_sequences=B_E,
            dataset_name=ds_name,
            split=E_split,
            G=1,
            compute_rb=True,
        )
        E_batch = self._pack_E_from_sequences(E_sequences)

        # U: distinct prompts, G responses each (SequenceProcessor handles reward computation)
        U_sequences, U_lp, _U_diag = self._sequence_processor.generate_with_logprobs(
            prompts=None,
            G=G_U,
            dataset_name=ds_name,
            split=U_split,
            num_prompts=B_U,
            compute_rb=False,
            with_grad=False,
        )
        U_batch = self._pack_U_from_sequences(U_sequences, U_lp.rewards)
        return E_batch, U_batch

    def _get_splits(self) -> tuple[str, str]:
        """Get E and U splits, supporting both separate and single split configurations."""
        batch_config = self.config['batch_config']
        E_split = batch_config.get('E_split', batch_config.get('split', 'test'))
        U_split = batch_config.get('U_split', batch_config.get('split', 'test'))
        return E_split, U_split

    # --- Batch cache I/O helpers ---
    def _save_batch(self, batch: Dict[str, Any], path: str) -> None:
        import torch as _torch
        from pathlib import Path as _Path
        def _to_cpu(obj):
            if isinstance(obj, _torch.Tensor):
                return obj.detach().to('cpu')
            if isinstance(obj, list):
                return [_to_cpu(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            return obj
        cpu_batch = _to_cpu(batch)
        E_split, U_split = self._get_splits()
        meta = {
            'dataset_name': self.config['batch_config']['dataset_name'],
            'E_split': E_split,
            'U_split': U_split,
            'split': self.config['batch_config'].get('split', 'N/A'),  # For backward compatibility
            'B': int(cpu_batch.get('num_prompts', len(cpu_batch.get('prompt_lens', [])))) ,
            'G': int(cpu_batch.get('num_responses_per_prompt', 1)),
        }
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        _torch.save({'meta': meta, 'data': cpu_batch}, path)

    def _load_batch(self, path: str, device: torch.device) -> Dict[str, Any]:
        import torch as _torch
        blob = _torch.load(path, map_location='cpu')
        data = blob['data']
        def _to_device(obj):
            if isinstance(obj, _torch.Tensor):
                return obj.to(device)
            if isinstance(obj, list):
                return [_to_device(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_device(v) for k, v in obj.items()}
            return obj
        return _to_device(data)

    def _get_or_sample_E(self, B_E: int) -> Dict[str, Any]:
        reuse = self.config.get('probe_reuse', {}).get('reuse_E', False)
        cache_path = self.config.get('probe_reuse', {}).get('e_batch_cache_path', None)
        device = next(self.model.parameters()).device
        if reuse and cache_path and os.path.exists(cache_path):
            self.logger.info(f"Loading cached E-batch from {cache_path}")
            E_batch = self._load_batch(cache_path, device)
            # Sanity: E must have G=1
            assert int(E_batch.get('num_responses_per_prompt', 1)) == 1, "Cached E-batch must have G=1"
            return E_batch
        E_split, _ = self._get_splits()
        # Log E-generation config reminder (top_p forced to 1.0)
        try:
            self.logger.info(
                f"[IS] E-generation: top_p forced to 1.0; temperature={self._sequence_processor.config.temperature}"
            )
        except Exception:
            pass
        E_sequences, _E_lp, _E_diag = self._sequence_processor.generate_with_replacement_sampling(
            total_sequences=B_E,
            dataset_name=self.config['batch_config']['dataset_name'],
            split=E_split,
            G=1,
            compute_rb=True,
        )
        E_batch = self._pack_E_from_sequences(E_sequences)
        if reuse and cache_path:
            self.logger.info(f"Saving E-batch to {cache_path}")
            self._save_batch(E_batch, cache_path)
        return E_batch

    def _get_or_sample_U(self, B_U: int, G_U: int) -> Dict[str, Any]:
        reuse = self.config.get('probe_reuse', {}).get('reuse_U', False)
        cache_path = self.config.get('probe_reuse', {}).get('u_batch_cache_path', None)
        device = next(self.model.parameters()).device
        if reuse and cache_path and os.path.exists(cache_path):
            self.logger.info(f"Loading cached U-batch from {cache_path}")
            U_batch = self._load_batch(cache_path, device)
            return U_batch
        _, U_split = self._get_splits()
        U_sequences, U_lp, _U_diag = self._sequence_processor.generate_with_logprobs(
            prompts=None,
            G=G_U,
            dataset_name=self.config['batch_config']['dataset_name'],
            split=U_split,
            num_prompts=B_U,
            compute_rb=False,
            with_grad=False,
        )
        U_batch = self._pack_U_from_sequences(U_sequences, U_lp.rewards)
        if reuse and cache_path:
            self.logger.info(f"Saving U-batch to {cache_path}")
            self._save_batch(U_batch, cache_path)
        return U_batch
            
    def run_offline_analysis(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Run complete offline entropy analysis.
        
        Args:
            checkpoint_path: Path to checkpoint to analyze
            
        Returns:
            Dictionary with analysis results matching the interface specified
            in offline_entropy_probe_strategy.txt section "Appendix: Minimal interfaces"
        """
        if not self.checkpoint_loaded:
            # Get optimizer path from config if available
            optimizer_path = self.config['checkpoint'].get('optimizer_path')
            self.load_checkpoint(checkpoint_path, optimizer_path)
                    
        try:
            # Delegate to the cleaner run_mixed_probe implementation
            self.logger.info("Delegating to run_mixed_probe for unified analysis")
            return self.run_mixed_probe()
            
        except Exception as e:
            self.logger.error(f"Error during offline analysis: {e}")
            raise
                
    
        
        
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        if not self.config['output']['results_path']:
            # Auto-generate path
            timestamp = int(time.time())
            results_path = f"entropy_probe_results_{timestamp}.json"
        else:
            results_path = self.config['output']['results_path']
            
        if self.rank == 0:  # Only rank 0 saves
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {results_path}")
        
            
    @classmethod
    def from_config_file(cls, config_path: str) -> 'OfflineEntropyProbe':
        """Create probe instance from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
        
    def get_config_template(self) -> str:
        """Return the path to the config template for reference."""
        return str(Path(__file__).parent / "configs" / "probe_config_template.yaml")


    #----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    #-------------------- Main methods for doing all the measurements -------------------------------
    #----------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------


    def run_experiments(self) -> Dict[str, Any]:
        """
        Unified, single-GPU mixed probe:
        1) Sample U and E batches via SequenceProcessor (no DDP).
        2) Compute the update vector v on U (normalized by LR).
        3) For an eta sweep, evaluate:
            - Ground-truth ΔH_true on E via parameter overrides (IS path).
            - First-order ΔH1 ≈ X̄ · Δη using the same Δη = η v.

        Assumptions / conventions:
        - SequenceProcessor enforces runtime fp32, with optional fp64 for entropy/log-softmax
            controlled by `config["precision"]["entropy_dtype"] in {float32,float64}`.
        - Top-p is forced to 1.0 in SP to keep the policy simple/consistent during probing.
        - `compute_update_vector` returns a named mapping `v_named: Dict[str, Tensor]`.
        - We interpret the physical update as Δη = η v_named (η are "learning rate" scalars).

        Returns:
        dict with:
            - sweep: list of per-eta results (deltaH_true, deltaH1, bars_dot, norms, timing)
            - E/U batch sizes and precision summary
            - aggregate timing
        """

        if not self.checkpoint_loaded:
            # Get optimizer path from config if available
            checkpoint_path = self.config['checkpoint'].get('checkpoint_path')
            optimizer_path = self.config['checkpoint'].get('optimizer_path')
            self.load_checkpoint(checkpoint_path, optimizer_path)


        t0 = time.time()
        self.logger.info("=== Mixed Probe: start ===")

        # ---------------------------------------------------------------------
        # 0) Ensure a clean SP instance with your trimmed precision config
        # ---------------------------------------------------------------------
        self._ensure_sequence_processor()

        # Pull a compact precision summary for logs
        pcfg = (self.config.get("precision", {}) or {})
        runtime_dtype = str(pcfg.get("runtime_dtype", "float32")).lower()
        entropy_dtype = str(pcfg.get("entropy_dtype", "float32")).lower()
        self.logger.info(f"[precision] runtime={runtime_dtype}, entropy={entropy_dtype}")

        # ---------------------------------------------------------------------
        # 1) Sample E and U batches (or load from cache) — single GPU, no DDP
        # ---------------------------------------------------------------------
        B_E = int(self.config["batch_config"]["B_E"])
        B_U = int(self.config["batch_config"]["B_U"])
        G_U = int(self.config["batch_config"]["G"])

        self.logger.info(f"[sampling] Target sizes: E: B={B_E}, G=1;  U: B={B_U}, G={G_U}")

        t_samp = time.time()
        E_batch = self._get_or_sample_E(B_E)
        U_batch = self._get_or_sample_U(B_U, G_U)
        t_samp = time.time() - t_samp

        B_E_real = int(E_batch.get("num_prompts", B_E))
        B_U_real = int(U_batch.get("num_prompts", B_U))
        self.logger.info(f"[sampling] Done in {t_samp:.2f}s.  E(B={B_E_real}, G=1)  U(B={B_U_real}, G={G_U})")

        # ---------------------------------------------------------------------
        # 2) Compute update vector v on U (normalized by LR)
        # ---------------------------------------------------------------------
        self.logger.info("[update-vector] Computing v (normalized by LR) on U …")
        t_v = time.time()
        v_named, v_stats = compute_update_vector(
            model=self.model,
            optimizer=self.optimizer,
            U_batch=U_batch,
            config=self.config,
            logger=self.logger,
        )
        t_v = time.time() - t_v
        v_norm = float(v_stats.get("vec_norm", 0.0))
        self.logger.info(f"[update-vector] ||v||₂ ≈ {v_stats.get('vec_norm', 0.0):.3e}  ({t_v:.2f}s)")


        # ---------------------------------------------------------------------
        # 3) Prepare components for the two ΔH estimates
        # ---------------------------------------------------------------------
        # 3A) ΔH_true via importance sampling + functional param overrides
        if self.delta_entropy_true is None:
            self.delta_entropy_true = DeltaEntropyTrue(
                model=self.model,
                sequence_processor=self._sequence_processor,
                config=self.config,
                logger=self.logger,
            )

        # Base IS config; we’ll override lr/eta per sweep item
        true_cfg_base = {
            "microbatch_size": self.config.get("true_delta_h", {}).get("microbatch_size", 1),
            "is_mode": self.config.get("true_delta_h", {}).get("is_mode", "snis"),
            "clip_c": self.config.get("true_delta_h", {}).get("clip_c", 10.0),
            "report_per_token": self.config.get("true_delta_h", {}).get("report_per_token", False),
            "snapshot_device": self.config.get("true_delta_h", {}).get("snapshot_device", "cpu"),
            # we’ll set "lr_override" per-eta below when the fallback path is used
        }

        # 3B) ΔH1 ≈ X̄ · Δη (first-order approx on E)
        #if self.delta_entropy_approx is None:
        #    self.delta_entropy_approx = DeltaEntropyApprox(
        #        model=self.model, config=self.config, logger=self.logger
        #    )
        

        trainable_names = {n for n, _ in self.model.named_parameters() if _.requires_grad}
        extra = [k for k in v_named if k not in trainable_names]
        if extra:
            self.logger.warning(f"[update-vector] v_named contains non-trainable keys (ignored by LoRA path): {extra[:5]}")



        # ---------------------------------------------------------------------
        # 4) Eta sweep
        # ---------------------------------------------------------------------

        comp_cfg = self.config.get("estimator", {})
        if comp_cfg.get("eta_sweep", False):
            eta_list = [float(eta) for eta in comp_cfg.get("eta_list", [2e-5])]
        else:
            eta_list = [float(comp_cfg.get("single_eta", 2e-5))]

        self.logger.info(f"[sweep] η values: {eta_list}")

        sweep_results = []

        # Compute normalized h_approx once
        t_approx = time.time()

        # temporary debug dummy result

        h_approx_normalized = 0.0
        #h_approx_normalized = self.delta_entropy_approx.compute_delta_h_approx(
        #        E_batch=E_batch,
        #        v_named=v_named,
        #    )
        t_approx = time.time() - t_approx
        self.logger.info(f"[ΔHapprox] Computed h_approx_normalized in {t_approx:.2f}s")


        # Then sweep over eta to compue true deltaH, using h_approx = eta*h_approx_normalized

        t_true_total = 0.0

        for eta in eta_list:
            self.logger.info(f"— sweep η={eta:g} —")

            # 4A) Ground-truth ΔH_true on E
            t_true = time.time()
            # If we’re on the fallback path, we must pass lr_override in cfg
            true_cfg = dict(true_cfg_base)

            deltaH_true = self.delta_entropy_true.compute_delta_h_true(
                E_batch, 
                v_named, 
                float(eta), 
                true_cfg)
            
            t_true = time.time() - t_true
            t_true_total += t_true

            # 4B) Approximate ΔH ≈ X̄ · Δ\theta on E
            deltaH_approx = float(eta) * h_approx_normalized

            self.logger.info(
                f"[η={eta:g}] ΔH_true={deltaH_true:.6e}   ΔH1={deltaH_approx:.6e}   "
                f"||v||₂={v_norm:.3e}"
                f"True {t_true:.2f}s"
            )

            sweep_results.append({
                "eta": float(eta),
                "deltaH_true": deltaH_true,
                "deltaH_approx": deltaH_approx,
            })

        # ---------------------------------------------------------------------
        # 5) Package results
        # ---------------------------------------------------------------------
        total_time = time.time() - t0
        out = {
            "sweep": sweep_results,
            "B_E": B_E_real,
            "B_U": B_U_real,
            "G_U": G_U,
            "precision": {
                "runtime_dtype": runtime_dtype,
                "entropy_dtype": entropy_dtype,
            },
            "timing": {
                "total": total_time,
                "sampling": t_samp,
                "update_vector": t_v,
                "true_total": t_true_total,
                "h_approx": t_approx,
            },
        }

        self.logger.info(
            f"=== Mixed Probe: done in {total_time:.2f}s "
            f"(sampling {t_samp:.2f}s, v {t_v:.2f}s, true Σ {t_true_total:.2f}s, h_approx Σ {t_approx:.2f}s) ==="
        )
        return out
