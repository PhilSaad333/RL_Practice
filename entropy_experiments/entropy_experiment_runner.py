"""
Main orchestrator class for the entropy experiments.

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

from dataclasses import dataclass, field, asdict

# === Model and optimizer loading ===
from entropy_experiments.utils.model_loader import load_peft_for_probe, load_adam_optimizer_from_path

import torch

# from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time
import math
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence
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

from entropy_experiments.utils import control_variates as cv





@dataclass
class ExperimentPlan:
    compute_true: bool = True
    compute_linear: bool = True
    compute_linquad: bool = False
    run_control_variates: bool = False
    capture_per_sequence: bool = False
    eta_list: Optional[Sequence[float]] = None
    clip_overrides: Optional[Sequence[float]] = None
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
        
    @dataclass
    class _Batches:
        E: Dict[str, Any]
        U: Dict[str, Any]
        sampling_sec: float
        info: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _UpdateInfo:
        v_named: Dict[str, torch.Tensor]
        stats: Dict[str, Any]
        seconds: float

    def _resolve_plan(
        self,
        plan: Optional[ExperimentPlan],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentPlan:
        """Merge plan defaults with any overrides supplied by the caller."""
        resolved = ExperimentPlan() if plan is None else plan
        if overrides:
            for key, value in overrides.items():
                if hasattr(resolved, key):
                    setattr(resolved, key, value)
        return resolved

    def _prepare_batches(self) -> "_Batches":
        """Sample (or load) the E and U batches once."""
        B_E = int(self.config["batch_config"]["B_E"])
        B_U = int(self.config["batch_config"]["B_U"])
        G_U = int(self.config["batch_config"]["G"])

        self.logger.info(f"[sampling] Target sizes: E: B={B_E}, G=1;  U: B={B_U}, G={G_U}")

        t_samp = time.time()
        E_batch = self._get_or_sample_E(B_E)
        U_batch = self._get_or_sample_U(B_U, G_U)
        sampling_sec = time.time() - t_samp

        B_E_real = int(E_batch.get("num_prompts", B_E))
        B_U_real = int(U_batch.get("num_prompts", B_U))
        self.logger.info(
            f"[sampling] Done in {sampling_sec:.2f}s.  E(B={B_E_real}, G=1)  U(B={B_U_real}, G={G_U})"
        )

        info = {
            "B_E": B_E,
            "B_U": B_U,
            "G_U": G_U,
            "B_E_real": B_E_real,
            "B_U_real": B_U_real,
        }
        return self._Batches(E=E_batch, U=U_batch, sampling_sec=sampling_sec, info=info)

    def _compute_update_info(self, U_batch: Dict[str, Any]) -> "_UpdateInfo":
        """Compute the normalized update vector plus timing/stats."""
        self.logger.info("[update-vector] Computing v (normalized by LR) on U …")
        t_v = time.time()
        v_named, v_stats = compute_update_vector(
            model=self.model,
            optimizer=self.optimizer,
            U_batch=U_batch,
            config=self.config,
            logger=self.logger,
        )
        seconds = time.time() - t_v
        self.logger.info(
            f"[update-vector] ||v|| = {v_stats.get('vec_norm', 0.0):.3e}  ({seconds:.2f}s)"
        )
        return self._UpdateInfo(v_named=v_named, stats=v_stats, seconds=seconds)

    def _maybe_compute_delta_h_approx(
        self,
        E_batch: Dict[str, Any],
        update: "_UpdateInfo",
        plan: ExperimentPlan,
    ) -> Optional[Dict[str, Any]]:
        """Return approx δH diagnostics if requested."""
        if not plan.compute_linear and not plan.compute_linquad:
            return None

        if self.delta_entropy_approx is None:
            self.delta_entropy_approx = DeltaEntropyApprox(
                model=self.model,
                sequence_processor=self._sequence_processor,
                config=self.config,
                logger=self.logger,
            )

        approx_cfg = (self.config.get("approx_delta_h", {}) or {})
        method = str(approx_cfg.get("method", "grad_dot")).lower()
        curv_cfg = (approx_cfg.get("curvature", {}) or {})
        use_quad = bool(curv_cfg.get("enabled", False)) or plan.compute_linquad

        t_approx = time.time()
        if method in {"jvp", "forward", "jvp_rb"}:
            self.logger.info("[h_approx] Using JVP method")
            approx_result = self.delta_entropy_approx.compute_delta_h_approx(
                E_batch=E_batch,
                v_named=update.v_named,
                include_quadratic=use_quad,
                return_per_sequence=plan.capture_per_sequence,
            )
        else:
            self.logger.info("[h_approx] Using grad·dot method")
            approx_result = self.delta_entropy_approx.compute_delta_h_approx(
                E_batch=E_batch,
                v_named=update.v_named,
                include_quadratic=use_quad,
                return_per_sequence=plan.capture_per_sequence,
            )
        duration = time.time() - t_approx
        self.logger.info(f"[δHapprox] Computed in {duration:.2f}s")

        approx_result["duration"] = duration
        approx_result["method"] = method
        approx_result["used_quad"] = use_quad
        approx_result["update_seconds"] = update.seconds
        approx_result["update_stats"] = update.stats
        return approx_result

    def _maybe_compute_delta_h_true(
        self,
        E_batch: Dict[str, Any],
        update: "_UpdateInfo",
        plan: ExperimentPlan,
        approx: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not plan.compute_true:
            return None

        if self.delta_entropy_true is None:
            self.delta_entropy_true = DeltaEntropyTrue(
                model=self.model,
                sequence_processor=self._sequence_processor,
                config=self.config,
                logger=self.logger,
            )

        true_cfg = (self.config.get("true_delta_h", {}) or {})
        comp_cfg = (self.config.get("estimator", {}) or {})

        if plan.eta_list is not None:
            eta_list = [float(x) for x in plan.eta_list]
        elif comp_cfg.get("eta_sweep", False):
            eta_list = [float(x) for x in comp_cfg.get("eta_list", [])]
        else:
            eta_list = [float(comp_cfg.get("single_eta", 2e-5))]

        clip_overrides = plan.clip_overrides or true_cfg.get("clip_overrides") or []
        symmetric_eta_cfg = true_cfg.get("symmetric_eta") if plan.clip_overrides is None else None

        entries = []
        total_true_time = 0.0
        for eta in eta_list:
            t_true = time.time()
            result = self.delta_entropy_true.compute_delta_h_true(
                E_batch=E_batch,
                v_named=update.v_named,
                eta=float(eta),
                cfg=true_cfg,
                return_details=True,
                symmetric_eta=symmetric_eta_cfg,
                clip_overrides=clip_overrides,
            )
            total_true_time += time.time() - t_true

            if isinstance(result, dict):
                delta_val = float(result.get("delta_h_true", 0.0))
                diagnostics = result
            else:
                delta_val = float(result)
                diagnostics = None

            if self.logger:
                ess_val = diagnostics.get("ess") if diagnostics else "n/a"
                self.logger.info(f"[δH_true] η={eta:g}  ΔH={delta_val:.6e}  ESS={ess_val}")

            entries.append(
                {
                    "eta": float(eta),
                    "delta_h_true": delta_val,
                    "diagnostics": diagnostics,
                }
            )

        return {
            "entries": entries,
            "duration": total_true_time,
        }
    
    def _maybe_run_control_variates(
        self,
        E_batch: Dict[str, Any],
        update: "_UpdateInfo",
        plan: ExperimentPlan,
        approx: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not plan.run_control_variates:
            return None
        cv_cfg = (self.config.get("control_variates", {}) or {})
        features = plan.clip_overrides or cv_cfg.get("features")
        return cv.run_control_variate_analysis(
            delta_approx=self.delta_entropy_approx,
            E_batch=E_batch,
            v_named=update.v_named,
            normalization=str((self.config.get("approx_delta_h", {}) or {}).get("normalize", "per_token")).lower(),
            out_dir=None,
            features=features
        )

    def _assemble_outputs(
        self,
        plan: ExperimentPlan,
        batches: "_Batches",
        update: "_UpdateInfo",
        approx: Optional[Dict[str, Any]],
        true: Optional[Dict[str, Any]],
        cv: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "B_E": batches.info.get("B_E_real", batches.info.get("B_E")),
            "B_U": batches.info.get("B_U_real", batches.info.get("B_U")),
            "G_U": batches.info.get("G_U"),
            "sampling_sec": batches.sampling_sec,
            "update": update.stats,
            "approx": approx,
            "true": true,
            "control_variates": cv,
        }



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



    def _resolve_plan(
        self,
        plan: Optional['ExperimentPlan'],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> 'ExperimentPlan':
        """Return a concrete ExperimentPlan based on explicit overrides."""
        if plan is None:
            plan = ExperimentPlan()
        if overrides:
            for key, value in overrides.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
        return plan

    def run_experiments(
        self,
        plan: Optional[ExperimentPlan] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Run ΔH experiments according to `plan`, returning measurements and diagnostics."""
        resolved_plan = self._resolve_plan(plan, overrides)

        if not self.checkpoint_loaded:
            checkpoint_path = (self.config["checkpoint"] or {}).get("checkpoint_path")
            optimizer_path = (self.config["checkpoint"] or {}).get("optimizer_path")
            self.load_checkpoint(checkpoint_path, optimizer_path)

        t0 = time.time()
        self.logger.info("=== Mixed Probe: start ===")

        self._ensure_sequence_processor()
        pcfg = (self.config.get("precision", {}) or {})
        runtime_dtype = str(pcfg.get("runtime_dtype", "float32")).lower()
        entropy_dtype = str(pcfg.get("entropy_dtype", "float32")).lower()
        self.logger.info(f"[precision] runtime={runtime_dtype}, entropy={entropy_dtype}")

        batches = self._prepare_batches()
        update_info = self._compute_update_info(batches.U)

        approx_result = self._maybe_compute_delta_h_approx(
            batches.E,
            update_info,
            resolved_plan,
        )
        true_result = self._maybe_compute_delta_h_true(
            batches.E,
            update_info,
            resolved_plan,
            approx_result,
        )
        cv_result = self._maybe_run_control_variates(
            batches.E,
            update_info,
            resolved_plan,
            approx_result,
        )

        result = self._assemble_outputs(
            resolved_plan,
            batches,
            update_info,
            approx_result,
            true_result,
            cv_result,
        )

        precision = result.setdefault("precision", {})
        precision["runtime_dtype"] = runtime_dtype
        precision["entropy_dtype"] = entropy_dtype

        timing = result.setdefault("timing", {})
        total_time = time.time() - t0
        timing["total"] = total_time

        self.logger.info(
            "=== Mixed Probe: done in "
            f"{total_time:.2f}s "
            f"(sampling {timing.get('sampling', 0.0):.2f}s, "
            f"update {timing.get('update_vector', 0.0):.2f}s, "
            f"true {timing.get('true', 0.0) or 0.0:.2f}s, "
            f"approx {timing.get('approx', 0.0) or 0.0:.2f}s) ==="
        )

        return result




