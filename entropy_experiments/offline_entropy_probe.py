"""
Offline Entropy Probe

Main orchestrator class for the offline entropy probe analysis.
Implements the strategy outlined in offline_entropy_probe_strategy.txt
based on the theory from RL_studies.pdf.

Key functionality:
- Load LoRA/QLoRA adapter and Adam optimizer state
- Sample batches of prompts and responses - E batch for "evaluation" or "entropy", U batch for "update"  
- Compute first-order entropy change prediction δH₁ after updating with U batch, using E batch to estimate entropy gradient
- Compute estimators of the variance of δH₁ (optional, config-driven) to get a sense of uncertainty
- Measure actual entropy change ΔH via importance sampling (entropy estimated on E batch before/after update with U batch)

Usage:
    probe = OfflineEntropyProbe(config)
    results = probe.run_offline_analysis(checkpoint_path)
"""

# === Model and optimizer loading ===
from entropy_experiments.model_loader import load_peft_for_probe, load_adam_optimizer_from_path

import torch
import torch.distributed as dist
from sequence_processing.sequence_processor import (
    SequenceProcessor, GenerationConfig, BatchedSequences,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time
import math
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import json

from entropy_experiments.probe_components import ProbeComponents
from entropy_experiments.adam_preconditioner import AdamPreconditioner  
from entropy_experiments.delta_entropy_is import DeltaEntropyIS
import entropy_experiments.distributed_helpers as distributed_helpers
from entropy_experiments.distributed_helpers import DistributedHelpers
from entropy_experiments.detailed_logger import DetailedLogger


class OfflineEntropyProbe:
    """
    Offline entropy probe for analyzing entropy changes in RL training.
    
    This implements the complete pipeline for measuring δH₁ (predicted entropy change)
    and ΔH (actual entropy change) given a training checkpoint.

    Main method is run_mixed_probe.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the offline entropy probe.
        
        Args:
            config: Configuration dictionary matching probe_config_template.yaml
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Single GPU mode for probe (from fix.txt)
        self.SINGLE_GPU = os.getenv("ENTROPY_PROBE_SINGLE_GPU", "1") == "1"
        
        # Initialize distributed if needed (disabled in single GPU mode)
        if self.SINGLE_GPU:
            # Skip DDP entirely for the probe
            self.distributed = False
            self.rank = 0
            self.world_size = 1
        else:
            self.distributed = dist.is_initialized()
            self.rank = dist.get_rank() if self.distributed else 0
            self.world_size = dist.get_world_size() if self.distributed else 1
        
        # Initialize components
        self.probe_components = None
        self.adam_preconditioner = None  
        self.delta_entropy_is = None
        self.u_statistics = None
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
        
    def load_checkpoint(self, checkpoint_path: str, optimizer_path: Optional[str] = None) -> None:
        """
        Load model checkpoint and optimizer state.
        
        Args:
            checkpoint_path: Path to model checkpoint (can be directory or file)
            optimizer_path: Optional path to optimizer state (if separate)
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Update config with provided paths
        if checkpoint_path:
            self.config['checkpoint']['checkpoint_path'] = checkpoint_path
        if optimizer_path:
            self.config['checkpoint']['optimizer_path'] = optimizer_path
        
        # Initialize model using LoRA/QLoRA adapter via model_loader
        adapter_path = self.config['checkpoint']['checkpoint_path']
        # Prefer the existing helper which wraps load_peft_for_probe and sets adapter
        self.model = self._load_lora_model(adapter_path)
        
        # Resolve optimizer path (required for Adam preconditioning state)
        if not optimizer_path:
            parent_dir = os.path.dirname(checkpoint_path)
            possible_optimizer_paths = [
                os.path.join(parent_dir, "optimizer.pt"),
                os.path.join(adapter_path, "optimizer.pt"),
                os.path.join(checkpoint_path, "optimizer.pt"),
            ]
            for opt_path in possible_optimizer_paths:
                if os.path.exists(opt_path):
                    optimizer_path = opt_path
                    self.logger.info(f"Found optimizer state at {opt_path}")
                    break
        if not optimizer_path or not os.path.exists(optimizer_path):
            raise FileNotFoundError(
                f"Optimizer state not found. Provide checkpoint.optimizer_path or place optimizer.pt near {adapter_path}"
            )
        # Load AdamW with state (ID remap handled inside)
        self.optimizer = load_adam_optimizer_from_path(self.model, optimizer_path)
        
        # Move to GPU and setup DDP if distributed
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.rank],
                find_unused_parameters=self.config['distributed']['find_unused_parameters']
            )
            
            # DIAGNOSTICS: Check if DDP wrapping affected parameter trainability
            trainable_after_ddp = 0
            lora_after_ddp = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_after_ddp += 1
                    if 'lora_A' in name or 'lora_B' in name:
                        lora_after_ddp += 1
            self.logger.info(f"Trainable parameters after DDP wrapping: {trainable_after_ddp} total, {lora_after_ddp} LoRA params")
            
        # Set to training mode (required for autograd)
        self.model.train()
        
        # Initialize components now that we have model and optimizer
        self._initialize_components()
        
        self.checkpoint_loaded = True
        self.logger.info("Checkpoint loaded successfully")
        
            
    def _load_lora_model(self, lora_path: str) -> torch.nn.Module:
        """Load LoRA/QLoRA model based on config toggle."""
        backbone = self.config['checkpoint'].get('model_config_path', 'Qwen/Qwen2.5-1.5B')
        use_qlora = bool(self.config['checkpoint'].get('use_qlora', False))
        dtype = self.config['checkpoint'].get('dtype', 'bf16')
        device_map = self.config['checkpoint'].get('device_map', 'cuda')

        model = load_peft_for_probe(
            base_id=backbone,
            adapter_path=lora_path,
            use_qlora=use_qlora,
            dtype=dtype,
            device_map=device_map,
            use_checkpointing=False,
        )
        model.to("cuda")
        model.eval()
        if hasattr(model, "set_adapter"):
            model.set_adapter("default")
        
        # DIAGNOSTICS: Verify LoRA adapters are active and trainable
        self.logger.info(f"Active adapters: {getattr(model, 'active_adapter', None)}")
        
        # DIAGNOSTICS: Count trainable parameters (consistent with canonical registry)
        # Get PEFT module (consistent with ProbeComponents approach)
        peft_model = model.module if hasattr(model, "module") else model
        trainable_named = [(n, p) for (n, p) in peft_model.named_parameters() if p.requires_grad]
        lora_named = [(n, p) for (n, p) in trainable_named 
                      if ("lora_a" in n.lower()) or ("lora_b" in n.lower()) or n.endswith("lm_head.weight")]
        
        self.logger.info(f"Trainable parameters after LoRA loading: {len(trainable_named)} total, {len(lora_named)} LoRA params")
        
        # Show first few LoRA parameter names for verification
        lora_names = [name for name, _ in lora_named[:5]]
        if lora_names:
            self.logger.info(f"Sample LoRA parameters: {lora_names}")
        else:
            self.logger.error("ERROR: No LoRA parameters found in trainable params!")
        
        self.logger.info(f"Loaded LoRA model: {backbone} + {lora_path}")
        return model
        
    
        
    def _initialize_components(self) -> None:
        """Initialize all probe components after model/optimizer are loaded."""
        self.probe_components = ProbeComponents(
            model=self.model,
            config=self.config,
            logger=self.logger
        )
        
        self.adam_preconditioner = AdamPreconditioner(
            optimizer=self.optimizer,
            config=self.config,
            logger=self.logger
        )
        
        # Only initialize DeltaEntropyIS if importance sampling is enabled
        if self.config['true_delta_h']['enabled']:
            self.delta_entropy_is = DeltaEntropyIS(
                model=self.model,
                config=self.config,
                logger=self.logger,
                sequence_processor=getattr(self, '_sequence_processor', None),
            )
        else:
            self.delta_entropy_is = None
            self.logger.info("Importance sampling disabled (true_delta_h.enabled: false)")
    
        
        if self.distributed:
            self.distributed_helpers = DistributedHelpers(
                world_size=self.world_size,
                rank=self.rank,
                config=self.config,
                logger=self.logger
            )
    
    # --- SequenceProcessor setup and sampling helpers ---
    def _ensure_sequence_processor(self):
        if hasattr(self, "_sequence_processor") and self._sequence_processor is not None:
            return
        
        # BUG FIX: Ensure model is loaded before creating SequenceProcessor
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model must be loaded before SequenceProcessor can be created. "
                "Call load_checkpoint() or run_mixed_probe() first."
            )
            
        from transformers import AutoTokenizer
        backbone = self.config['checkpoint'].get('model_config_path', 'Qwen/Qwen2.5-1.5B')
        tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token

        gen_cfg = self.config.get('generation', {})
        
        est_mode = (self.config.get('estimator', {}) or {}).get('x_estimator_mode', 'naive')
        rb_rg_cfg = gen_cfg.get('rb_requires_grad', False)
        rb_rg_final = True if est_mode == 'rb_residual' else rb_rg_cfg
        
        if est_mode == 'rb_residual' and not rb_rg_cfg:
            self.logger.warning(
                f"OVERRIDING rb_requires_grad: {rb_rg_cfg} → True (required for x_estimator_mode=rb_residual)"
            )
        
        self.logger.info(f"SequenceProcessor config: x_estimator_mode={est_mode}, rb_requires_grad={rb_rg_final}")
        
        sp_cfg = GenerationConfig(
            temperature=gen_cfg.get('temperature', 1.0),
            top_p=gen_cfg.get('top_p', 1.0),
            max_new_tokens=gen_cfg.get('max_new_tokens', 256),
            do_sample=True,
            num_return_sequences=self.config['batch_config']['G'],
            gen_batch_size=gen_cfg.get('gen_batch_size', 8),
            tf_batch_size=gen_cfg.get('tf_batch_size', 64),
            rb_requires_grad=rb_rg_final,
        )
        self._sequence_processor = SequenceProcessor(self.model, tok, sp_cfg)
        # Ensure DeltaEntropyIS uses the same SequenceProcessor (if already constructed)
        if hasattr(self, 'delta_entropy_is') and self.delta_entropy_is is not None:
            try:
                self.delta_entropy_is.sequence_processor = self._sequence_processor
                self.logger.info("DeltaEntropyIS bound to SequenceProcessor for RB evaluation")
            except Exception:
                pass

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
                
    
    def run_mixed_probe(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """
        STAGE 1: Run mixed E/U batch entropy probe analysis.
        
        This implements the new approach with separate evaluation (E) and update (U) batches:
        - E batch: Used for computing X gradients (∇H_w)
        - U batch: Used for computing Y gradients (P∇J, preconditioned)
        - Estimator: δH₁ = lr * (X̄ · Ȳ)
        
        Returns:
            Dict with deltaH1, bars_dot, B_E, B_U, timing, and diagnostics
        """
        # Guard against probe being called under inference_mode or no_grad
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Probe entrypoint called under torch.inference_mode(); remove that context.")
        
        start_time = time.time()
        
        # Initialize detailed logging if enabled
        if self.detailed_logger:
            log_file = self.detailed_logger.log_run_start(checkpoint_path or 
                                                        self.config['checkpoint']['checkpoint_path'], 
                                                        self.config)
        
        try:
            self.logger.info("Starting Stage 1 Mixed E/U Batch Probe Analysis")
            
            # Load checkpoint if provided, otherwise use config paths
            if checkpoint_path is None:
                checkpoint_path = self.config['checkpoint']['checkpoint_path']
            
            # Load model and optimizer (following run_offline_analysis pattern)
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load LoRA/QLoRA adapter using model_loader helper
            self.model = self._load_lora_model(checkpoint_path)
            
            # Load optimizer state - use fallback logic like load_checkpoint()
            optimizer_path = self.config['checkpoint'].get('optimizer_path')
            
            if not optimizer_path:
                # Auto-discover optimizer path relative to checkpoint
                parent_dir = os.path.dirname(checkpoint_path)
                possible_optimizer_paths = [
                    os.path.join(checkpoint_path, "optimizer.pt"),  # First: checkpoint-dir/optimizer.pt  
                    os.path.join(parent_dir, "optimizer.pt"),  # Standard: ../optimizer.pt
                    os.path.join(checkpoint_path, "..", "optimizer.pt"),  # Alternative
                ]
                
                for opt_path in possible_optimizer_paths:
                    if os.path.exists(opt_path):
                        optimizer_path = opt_path
                        self.logger.info(f"Auto-discovered optimizer at {optimizer_path}")
                        break
            
            if not optimizer_path:
                raise FileNotFoundError(
                    f"Could not find optimizer.pt relative to checkpoint {checkpoint_path}. "
                    f"Searched locations: {possible_optimizer_paths}. "
                    f"Please ensure optimizer.pt exists in one of these locations."
                )
            
            if not os.path.exists(optimizer_path):
                raise FileNotFoundError(
                    f"Optimizer path does not exist: {optimizer_path}. "
                    f"Check that optimizer.pt was saved alongside the model checkpoint."
                )
            
            self.logger.info(f"Loading optimizer state from {optimizer_path}")
            self.optimizer = load_adam_optimizer_from_path(self.model, optimizer_path)
            
            # Initialize components
            self._initialize_components()
            
            # Get config parameters - handle unified B_E_values format
            batch_config = self.config['batch_config']
            
            # Handle B_E: could be direct value or from B_E_values list
            if 'B_E' in batch_config:
                B_E = batch_config['B_E']
            elif 'B_E_values' in batch_config:
                B_E_values = batch_config['B_E_values']
                B_E = B_E_values[0] if isinstance(B_E_values, list) else B_E_values
            else:
                raise KeyError("Config must contain either 'B_E' or 'B_E_values' in batch_config")
                
            # Handle B_U: direct value
            if 'B_U' in batch_config:
                B_U = batch_config['B_U']  
            else:
                raise KeyError("Config must contain 'B_U' in batch_config")
            mb_size_prompts = self.config.get('computation_options', {}).get('mb_size_prompts', 2)
            weighting_mode = self.config.get('computation_options', {}).get('weighting_mode', 'dr_grpo')
            
            self.logger.info(f"Mixed probe config: B_E={B_E}, B_U={B_U}, mb_size={mb_size_prompts}, weighting={weighting_mode}")
            
            # Stage 3: Deterministic E/U index selection for multi-GPU consistency
            is_dist, rank, world_size = distributed_helpers.get_dist_info()
            self.logger.info(f"Distributed info: dist={is_dist}, rank={rank}/{world_size}")
            
            # Load dataset to get size (use E_split for distributed sampling consistency)
            from rlp_datasets import DATASET_REGISTRY
            dataset = DATASET_REGISTRY[self.config['batch_config']['dataset_name']]
            E_split, U_split = self._get_splits()
            ds_examples = dataset(E_split)
            dataset_size = len(ds_examples)
            
            # Log which splits are being used
            self.logger.info(f"Using dataset splits: E_split='{E_split}', U_split='{U_split}' (dataset size calculation based on E_split)")
            
            # Get master seed for deterministic sampling
            master_seed = self.config.get('computation_options', {}).get('master_seed', 42)
            
            if is_dist:
                # Deterministic sampling: rank 0 generates global indices, broadcasts to all ranks
                E_indices_global = None
                U_indices_global = None
                
                if rank == 0:
                    # Rank 0: Generate global E/U indices deterministically
                    random.seed(master_seed)
                    all_indices = list(range(dataset_size))
                    
                    # Ensure we have enough samples for both batches
                    total_needed = B_E + B_U
                    if dataset_size < total_needed:
                        self.logger.warning(f"Dataset size {dataset_size} < needed samples {total_needed}")
                        # Extend indices by repeating
                        all_indices = all_indices * ((total_needed // dataset_size) + 1)
                    
                    # Sample E and U indices without overlap
                    sampled_indices = random.sample(all_indices, min(total_needed, len(all_indices)))
                    E_indices_global = sampled_indices[:B_E]
                    U_indices_global = sampled_indices[B_E:B_E + B_U]
                    
                    self.logger.info(f"Rank 0 generated {len(E_indices_global)} E indices, {len(U_indices_global)} U indices")
                
                # Broadcast indices from rank 0 to all ranks
                E_indices_global = distributed_helpers.broadcast_int_list(root_rank=0, indices=E_indices_global)
                U_indices_global = distributed_helpers.broadcast_int_list(root_rank=0, indices=U_indices_global)
                
                # Create local shards per rank
                E_indices_local = E_indices_global[rank::world_size]
                U_indices_local = U_indices_global[rank::world_size]
                
                B_E_local = len(E_indices_local)
                B_U_local = len(U_indices_local)
                
                self.logger.info(f"Rank {rank}: E_local={B_E_local}, U_local={B_U_local}")
                
            else:
                # Single GPU: use all indices locally
                E_indices_local = None  # Will trigger random sampling in sample_batch
                U_indices_local = None
                B_E_local = B_E
                B_U_local = B_U
                
                self.logger.info(f"Single GPU: using random sampling")
            
            # Phase 0: Sampling E and U batches (with optional cache reuse)
            self.logger.info("Phase 0: Sampling E and U batches")
            if self.detailed_logger:
                self.detailed_logger.log_phase_start("phase0_sampling")
            phase0_start = time.time()
            G_U = self.config['batch_config']['G']
            # Ensure SP is ready before sampling
            self._ensure_sequence_processor()
            E_batch = self._get_or_sample_E(B_E)
            U_batch = self._get_or_sample_U(B_U, G_U)
            phase0_time = time.time() - phase0_start
            self.logger.info(f"Phase 0 complete: {phase0_time:.2f}s")
            self.logger.info(f"E-batch: G=1 (replacement), {E_batch['sequences'].shape[0]} prompts, {E_batch['sequences'].shape[1]} responses/prompt")
            self.logger.info(f"U-batch: G={G_U} (distinct), {U_batch['sequences'].shape[0]} prompts, {U_batch['sequences'].shape[1]} responses/prompt")

            # New deltaH1 prediction: compute Delta-theta via real RL step (snapshot/restore), then deltaH1 = <Xbar, Delta-theta>
            self.logger.info("Phase 1: Computing Delta-theta via real RL step on U (snapshot/restore)")
            delta_theta_buf, delta_theta_norm, B_U_used = self._compute_param_update_buffer(U_batch, mb_size_prompts)
            self.logger.info(f"Delta-theta ready: ||Delta-theta||={delta_theta_norm:.3e} across {len(delta_theta_buf)} trainables")

            # Compute Xbar on E, then deltaH1 = <Xbar, Delta-theta> (no lr scaling)
            self.logger.info("Phase 2-3: Computing Xbar on E and deltaH1 = <Xbar, Delta-theta>")
            compute = self.probe_components.compute_delta_h1_from_batches(
                E_batch=E_batch,
                U_batch=U_batch,
                mb_size_prompts=mb_size_prompts,
                weighting_mode=weighting_mode,
                adam_preconditioner=self.adam_preconditioner,
                optimizer=self.optimizer,
                param_update_buf=delta_theta_buf,
            )
            delta_h1 = compute['deltaH1']
            bars_dot = compute['bars_dot']
            learning_rate = compute['learning_rate']
            phase1_time = compute['timing']['phase1_time']
            phase2_time = compute['timing']['phase2_time']
            phase3_time = compute['timing']['phase3_time']
            B_E_global = B_E
            B_U_global = B_U_used
            self.logger.info(f"[RESULTS][delta-theta] bars_dot={bars_dot:.10f}, deltaH1={delta_h1:.10f}")

            # Skip legacy Ybar path
            compute_delta_h1 = False
            
            # Log batch data for detailed logging
            if self.detailed_logger:
                self.detailed_logger.log_phase_end("phase0_sampling")
                self.detailed_logger.log_batch_data("E_batch", E_batch)
                self.detailed_logger.log_batch_data("U_batch", U_batch)
            
            # Check if δH₁ computation should be performed
            probe_config = self.config.get('computation_options', {})
            compute_delta_h1 = probe_config.get('compute_delta_h1', True)
            
            # Initialize variables for potential use in later stages
            delta_h1 = 0.0
            bars_dot = 0.0
            learning_rate = 0.0
            phase1_time = phase2_time = phase3_time = 0.0
            B_E_global = B_E_local if not is_dist else None
            B_U_global = B_U_local if not is_dist else None
            
            if not compute_delta_h1:
                self.logger.info("δH₁ computation disabled (compute_delta_h1=False)")
                # Set global batch sizes for downstream use
                if is_dist:
                    B_E_global = distributed_helpers.count_global(B_E_local)
                    B_U_global = distributed_helpers.count_global(B_U_local)
                else:
                    B_E_global = B_E_local
                    B_U_global = B_U_local
            
            
            # ================================================================
            # STAGE 2: Two-Batch Ground-Truth Entropy Change - Optional
            # ================================================================
            compute_importance_sampling = probe_config.get('compute_importance_sampling', False)
            importance_enabled = self.config.get('true_delta_h', {}).get('enabled', False) or compute_importance_sampling
            ground_truth_results = {}
            
            if importance_enabled:
                self.logger.info("Phase 5: Computing two-batch ground-truth entropy change")
                if self.detailed_logger:
                    self.detailed_logger.log_phase_start("phase5_importance")
                phase5_start = time.time()
                
                # Initialize DeltaEntropyIS if not already done
                if not hasattr(self, 'delta_entropy_is') or self.delta_entropy_is is None:
                    self.delta_entropy_is = DeltaEntropyIS(
                        model=self.model, config=self.config, logger=self.logger,
                        sequence_processor=getattr(self, '_sequence_processor', None)
                    )
                
                # Enable detailed importance sampling logging if detailed logger is active
                if self.detailed_logger and self.detailed_logger.level in ['detailed', 'debug']:
                    self.delta_entropy_is._store_importance_details = True
                
                # Extract importance sampling configuration
                cfg_importance = {
                    'training_loss': self.config.get('true_delta_h', {}).get('training_loss', 'nll'),
                    'importance_microbatch_size': self.config.get('true_delta_h', {}).get('microbatch_size', 1),
                    'is_mode': self.config.get('true_delta_h', {}).get('is_mode', 'snis'),
                    'clip_c': self.config.get('true_delta_h', {}).get('clip_c', 10.0),
                    'report_per_token': self.config.get('true_delta_h', {}).get('report_per_token', False),
                    'snapshot_device': self.config.get('true_delta_h', {}).get('snapshot_device', 'cpu')
                }
                
                # Compute ground-truth entropy change
                ground_truth_results = self.delta_entropy_is.entropy_change_two_batch(
                    self.model, E_batch, U_batch, self.optimizer, cfg_importance
                )
                
                phase5_time = time.time() - phase5_start
                self.logger.info(f"🔍 [GROUND-TRUTH] deltaH_true={ground_truth_results['deltaH_true']:.10f}")
                self.logger.info(f"Phase 5 complete: {phase5_time:.2f}s")
                
                # Log ground truth results for detailed logging
                if self.detailed_logger:
                    self.detailed_logger.log_phase_end("phase5_importance")
                    self.detailed_logger.log_ground_truth_results(ground_truth_results)
                    
                    # Log detailed importance sampling data if available
                    if hasattr(self.delta_entropy_is, '_importance_details'):
                        details = self.delta_entropy_is._importance_details
                        self.detailed_logger.log_importance_sampling_details(
                            details['S_orig'], details['S_upd'],
                            details['RB_orig'], details['RB_upd']
                        )
            
            # ================================================================
            # Compile Final Results  
            # ================================================================
            total_time = time.time() - start_time
            results = {
                # Core Stage 1 results
                "bars_dot": bars_dot,
                "deltaH1": delta_h1,
                "learning_rate": learning_rate,
                
                # Batch sizes
                "B_E": B_E_global,
                "B_U": B_U_global,
                
                # Configuration
                "mb_size_prompts": mb_size_prompts,
                "weighting_mode": weighting_mode,
                
                # Timing breakdown
                "timing": {
                    "total_time": total_time,
                    "phase0_sampling": phase0_time,
                    "phase1_sum_X": phase1_time,
                    "phase2_sum_Y": phase2_time,
                    "phase3_delta_h1": phase3_time,
                }
            }

            # Attach LR sweep if present
            if 'sweep_results' in locals() and sweep_results is not None:
                results['sweep'] = sweep_results
            
            
            # Add Stage 2 ground-truth results if computed
            if importance_enabled:
                results.update({
                    **ground_truth_results,
                    "importance_enabled": True
                })
                results["timing"]["phase5_importance"] = phase5_time
            
            stage = "Stage 1+2" if importance_enabled else "Stage 1"
            self.logger.info(f"{stage} Mixed probe analysis completed in {total_time:.2f}s")
            
            # Finalize detailed logging
            if self.detailed_logger:
                log_file = self.detailed_logger.finalize_log(results)
                if log_file:
                    self.logger.info(f"Detailed log saved: {log_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during mixed probe analysis: {e}")
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
        
    def _compute_param_update_buffer(self, U_batch: Dict[str, Any], mb_size_prompts: int) -> tuple[dict[int, torch.Tensor], float, int]:
        """Compute actual parameter update (Delta-theta) via a real RL step on U, with snapshot/restore.

        Returns: (delta_theta_buf, l2_norm, B_U_used)
        """
        assert self.delta_entropy_is is not None, "DeltaEntropyIS component required"
        # Canonical trainables: match ProbeComponents' registry (requires_grad on PEFT-wrapped model)
        peft_model = self.model.module if hasattr(self.model, "module") else self.model
        trainable_params = [p for _, p in peft_model.named_parameters() if p.requires_grad]

        # Snapshot parameters and optimizer
        cpu_snaps, opt_state_snapshot = self.delta_entropy_is._snapshot_model_optimizer(self.model, self.optimizer, snapshot_device='cpu')
        before: dict[int, torch.Tensor] = {id(p): p.detach().to('cpu', torch.float32).clone() for p in trainable_params}

        # Perform the real RL step
        rl_grad_accum = int(self.config.get('computation_options', {}).get('rl_grad_accum', 1))
        importance_mb_size = int(self.config.get('true_delta_h', {}).get('microbatch_size', 1))
        self.delta_entropy_is._rl_update_streaming(U_batch, self.optimizer, rl_grad_accum, importance_mb_size)

        # Compute Delta-theta on CPU
        delta_theta: dict[int, torch.Tensor] = {}
        l2 = torch.zeros((), dtype=torch.float64)
        with torch.no_grad():
            for p in trainable_params:
                pid = id(p)
                after = p.detach().to('cpu', torch.float32)
                dth = (after - before[pid]).contiguous()
                delta_theta[pid] = dth
                l2 += (dth.double().pow(2).sum())
        l2 = float(l2.sqrt().item())

        # Restore model and optimizer
        self.delta_entropy_is._restore_model_optimizer(self.model, self.optimizer, cpu_snaps, opt_state_snapshot)
        B_U_used = int(U_batch.get('num_prompts', len(U_batch.get('max_lengths', []))))
        return delta_theta, l2, B_U_used

            
    @classmethod
    def from_config_file(cls, config_path: str) -> 'OfflineEntropyProbe':
        """Create probe instance from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
        
    def get_config_template(self) -> str:
        """Return the path to the config template for reference."""
        return str(Path(__file__).parent / "configs" / "probe_config_template.yaml")
