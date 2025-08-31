"""
Offline Entropy Probe

Main orchestrator class for the offline entropy probe analysis.
Implements the strategy outlined in offline_entropy_probe_strategy.txt
based on the theory from RL_studies.pdf.

Key functionality:
- Load model checkpoint and optimizer state
- Sample batches of prompts and responses - E batch for "evaluation" or "entropy", U batch for "update"  
- Compute first-order entropy change prediction δH₁ after updating with U batch, using E batch to estimate entropy gradient
- Compute estimators of the variance of δH₁ (optional, config-driven) to get a sense of uncertainty
- Measure actual entropy change ΔH via importance sampling (entropy estimated on E batch before/after update with U batch)

Usage:
    probe = OfflineEntropyProbe(config)
    results = probe.run_offline_analysis(checkpoint_path)
"""

# === Begin: LoRA-simple / QLoRA loader ===
from typing import Optional
import torch

def load_peft_for_probe(
    base_id: str,
    adapter_path: str,
    *,
    mode: str = "lora_simple",   # "lora_simple" or "qlora"
    dtype: str = "bf16",         # "bf16" or "fp16"
    device_map: str = "cuda",
    use_checkpointing: bool = False
):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    if mode == "lora_simple":
        # Plain fp16/bf16 base — NO quantization, NO k-bit preparation
        base = AutoModelForCausalLM.from_pretrained(
            base_id, 
            device_map=device_map, 
            torch_dtype=torch_dtype,
            attn_implementation="eager"  # Disable flash attention for VJP compatibility
        )
        # Ensure no checkpointing and cache is allowed (deterministic probe)
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = True

        peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        # For safety with some HF versions; helps when you later enable ckpting elsewhere
        if hasattr(peft, "enable_input_require_grads"):
            peft.enable_input_require_grads()

        return peft

    elif mode == "qlora":
        # Keep original QLoRA path if you ever want to compare—but you won't use it now
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_id, 
            device_map=device_map, 
            torch_dtype=torch_dtype, 
            quantization_config=bnb_cfg,
            attn_implementation="eager"  # Disable flash attention for VJP compatibility
        )
        base = prepare_model_for_kbit_training(base)
        if use_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
            if hasattr(base.config, "use_cache"):
                base.config.use_cache = False
        peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        if hasattr(peft, "enable_input_require_grads"):
            peft.enable_input_require_grads()
        return peft

    else:
        raise ValueError("mode must be 'lora_simple' or 'qlora'")
# === End: LoRA-simple / QLoRA loader ===

import torch
import torch.distributed as dist
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

from .probe_components import ProbeComponents
from .adam_preconditioner import AdamPreconditioner  
from .importance_sampling import ImportanceSampler
from . import distributed_helpers
from .distributed_helpers import DistributedHelpers


class OfflineEntropyProbe:
    """
    Offline entropy probe for analyzing entropy changes in RL training.
    
    This implements the complete pipeline for measuring δH₁ (predicted entropy change)
    and ΔH (actual entropy change) given a training checkpoint.
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
        self.importance_sampler = None
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
        
        self.logger.info(f"Initialized OfflineEntropyProbe on rank {self.rank}/{self.world_size}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"entropy_probe_rank_{self.rank if hasattr(self, 'rank') else 0}")
        logger.setLevel(getattr(logging, self.config['output']['log_level']))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.rank if hasattr(self, "rank") else 0}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
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
        
        # Initialize model using the updated config
        # Pass empty dict since we're using config paths instead
        self.model = self._initialize_model_from_checkpoint({})
        
        # Load optimizer state
        if optimizer_path:
            self.logger.info(f"Loading optimizer state from {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location='cpu')
        else:
            # Try to load optimizer from a companion file or embedded in checkpoint
            # For LoRA adapters, optimizer is typically in parent directory
            parent_dir = os.path.dirname(checkpoint_path)
            possible_optimizer_paths = [
                os.path.join(parent_dir, "optimizer.pt"),  # Most common: ../optimizer.pt
                checkpoint_path + "/optimizer.pt",         # Inside model dir (unlikely)
                checkpoint_path.replace('.pt', '_optimizer.pt'),
                checkpoint_path.replace('.pth', '_optimizer.pth'),
            ]
            
            optimizer_state = None
            for opt_path in possible_optimizer_paths:
                if os.path.exists(opt_path):
                    self.logger.info(f"Found optimizer state at {opt_path}")
                    optimizer_state = torch.load(opt_path, map_location='cpu')
                    break
            
            # Try loading as a full checkpoint with optimizer embedded
            if optimizer_state is None:
                try:
                    if os.path.isfile(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'optimizer' in checkpoint:
                            optimizer_state = checkpoint['optimizer']
                            self.logger.info("Found optimizer state embedded in checkpoint")
                except Exception as e:
                    self.logger.debug(f"Could not load checkpoint as dict: {e}")
            
            if optimizer_state is None:
                self.logger.warning("No optimizer state found - creating fresh optimizer")
                # Create a minimal state dict to initialize fresh optimizer
                optimizer_state = {'param_groups': [{'lr': 1e-6, 'weight_decay': 0.01}]}
            
        self.optimizer = self._initialize_optimizer_from_state(optimizer_state)
        
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
        
    def _initialize_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        """
        Initialize model from checkpoint.
        
        This follows the same pattern as rl_runner.py for loading LoRA models.
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        import os
        
        self.logger.info("Initializing model from checkpoint...")
        
        # Extract checkpoint path - handle both full checkpoints and directory paths
        checkpoint_path = self.config['checkpoint']['checkpoint_path']
        if not checkpoint_path:
            raise ValueError("checkpoint_path not specified in config")
            
        # Determine if this is a direct model checkpoint or LoRA adapter
        if os.path.isdir(checkpoint_path) and any(f.endswith('.safetensors') for f in os.listdir(checkpoint_path) if 'adapter' in f):
            # This is a LoRA adapter directory
            self.logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
            return self._load_lora_model(checkpoint_path)
        else:
            # This is a full model checkpoint
            self.logger.info(f"Loading full model checkpoint from: {checkpoint_path}")
            return self._load_full_model_checkpoint(checkpoint_path)
            
    def _load_lora_model(self, lora_path: str) -> torch.nn.Module:
        """Load LoRA model using LoRA-Simple approach (no quantization)."""
        # Get backbone from config
        backbone = self.config['checkpoint'].get('model_config_path', 'Qwen/Qwen2.5-1.5B')
        
        # Use LoRA-Simple loader (no quantization, no checkpointing)
        model = load_peft_for_probe(
            base_id=backbone,
            adapter_path=lora_path,
            mode="lora_simple",
            dtype="bf16",
            device_map="cuda",
            use_checkpointing=False
        )
        
        # Single-GPU probe: no DDP; keep deterministic eval
        model.to("cuda")
        model.eval()
        # IMPORTANT: ensure adapter is active BEFORE any forward that produces S/logits
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
        
    def _load_full_model_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """Load full model checkpoint."""
        from transformers import AutoModelForCausalLM
        
        # Load full model from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Enable training mode and gradients  
        model.train()
        # Disable gradient checkpointing for entropy probe - needed for clean autograd graphs
        # model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        self.logger.info(f"Loaded full model from: {checkpoint_path}")
        return model
        
    def _initialize_optimizer_from_state(self, optimizer_state: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Initialize optimizer from saved state.
        
        This follows the same pattern as rl_runner.py -> DRGRPO for creating AdamW optimizers.
        """
        import torch.optim as optim
        
        self.logger.info("Initializing optimizer from saved state...")
        
        # Extract optimizer configuration from saved state or use defaults from RL training
        # These defaults match DRGRPO.__init__ in dr_grpo.py
        lr = optimizer_state.get('param_groups', [{}])[0].get('lr', 1e-6)  # Extract from state
        weight_decay = optimizer_state.get('param_groups', [{}])[0].get('weight_decay', 0.01)
        
        self.logger.info(f"Creating AdamW optimizer with lr={lr}, weight_decay={weight_decay}")
        
        # Create optimizer with same configuration as DRGRPO (matches dr_grpo.py)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Load the saved state with custom parameter ID mapping
        try:
            # Handle different state formats and extract the actual optimizer state dict
            if isinstance(optimizer_state, dict):
                if 'state_dict' in optimizer_state:
                    state_dict = optimizer_state['state_dict']
                elif 'param_groups' in optimizer_state:
                    state_dict = optimizer_state
                else:
                    state_dict = optimizer_state
            else:
                raise ValueError(f"Unexpected optimizer_state type: {type(optimizer_state)}")
            
            # Custom parameter ID remapping to handle model reconstruction
            remapped_state_dict = self._remap_optimizer_state_ids(state_dict, optimizer)
            optimizer.load_state_dict(remapped_state_dict)
            
            self.logger.info("Successfully loaded optimizer state with ID remapping")
            
            # Validate that optimizer has state (exp_avg_sq for Adam preconditioning)
            param_count = len(list(self.model.parameters()))
            state_count = len(optimizer.state)
            self.logger.info(f"Optimizer state: {state_count} parameters have state out of {param_count} total")
            
            if state_count == 0:
                self.logger.warning("No optimizer state found - this may cause issues with Adam preconditioning")
            
            return optimizer
            
        except Exception as e:
            self.logger.error(f"Failed to load optimizer state: {e}")
            self.logger.info("Creating fresh optimizer without state")
            
            # Return fresh optimizer if loading fails
            fresh_optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            return fresh_optimizer
    
    def _remap_optimizer_state_ids(self, saved_state_dict: dict, current_optimizer: torch.optim.Optimizer) -> dict:
        """
        Remap parameter IDs in saved optimizer state to match current model parameters.
        
        This handles the common issue where model reconstruction creates new parameter IDs
        that don't match the saved optimizer state keys.
        
        Args:
            saved_state_dict: Original optimizer state dict with old parameter IDs
            current_optimizer: Current optimizer with new parameter IDs
            
        Returns:
            Remapped state dict with current parameter IDs as keys
        """
        # Extract current parameter list (in order)
        current_params = []
        for group in current_optimizer.param_groups:
            current_params.extend(group['params'])
            
        # Extract saved parameter IDs and states (in order)
        saved_state = saved_state_dict.get('state', {})
        saved_param_groups = saved_state_dict.get('param_groups', [])
        
        # Get saved parameter IDs in the same order they were saved
        saved_param_ids = []
        for group in saved_param_groups:
            if 'params' in group:
                saved_param_ids.extend(group['params'])
        
        self.logger.info(f"Remapping optimizer IDs: {len(saved_param_ids)} saved → {len(current_params)} current")
        
        if len(saved_param_ids) != len(current_params):
            self.logger.warning(f"Parameter count mismatch: saved={len(saved_param_ids)}, current={len(current_params)}")
            self.logger.warning("This might indicate model architecture changes - some optimizer state may be lost")
        
        # Create ID mapping: old_id → new_id based on position
        id_mapping = {}
        min_count = min(len(saved_param_ids), len(current_params))
        for i in range(min_count):
            old_id = saved_param_ids[i]
            new_id = id(current_params[i])
            id_mapping[old_id] = new_id
            
        # Remap the state dict
        remapped_state = {}
        successful_remaps = 0
        for old_id, param_state in saved_state.items():
            if old_id in id_mapping:
                new_id = id_mapping[old_id]
                remapped_state[new_id] = param_state
                successful_remaps += 1
            else:
                self.logger.debug(f"Skipping unmappable parameter ID: {old_id}")
                
        # Update param_groups with new IDs
        remapped_param_groups = []
        for i, group in enumerate(saved_param_groups):
            new_group = group.copy()
            if 'params' in group:
                # Map old param IDs to new param IDs
                old_param_ids = group['params']
                new_param_ids = []
                for old_id in old_param_ids:
                    if old_id in id_mapping:
                        new_param_ids.append(id_mapping[old_id])
                new_group['params'] = new_param_ids
            remapped_param_groups.append(new_group)
        
        self.logger.info(f"Successfully remapped {successful_remaps}/{len(saved_state)} parameter states")
        
        return {
            'state': remapped_state,
            'param_groups': remapped_param_groups
        }
        
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
        
        # Only initialize ImportanceSampler if importance sampling is enabled
        if self.config['importance']['enabled']:
            self.importance_sampler = ImportanceSampler(
                model=self.model,
                config=self.config,
                logger=self.logger
            )
        else:
            self.importance_sampler = None
            self.logger.info("ImportanceSampler disabled (importance.enabled: false)")
    
        
        if self.distributed:
            self.distributed_helpers = DistributedHelpers(
                world_size=self.world_size,
                rank=self.rank,
                config=self.config,
                logger=self.logger
            )
            
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
            
        start_time = time.time()
        
        try:
            # Step 1: Sample batch of prompts and responses
            self.logger.info("Sampling batch of prompts and responses")
            batch_data = self._sample_batch()
            
            # Step 2: Compute U-statistic for δH₁
            self.logger.info("Computing first-order entropy change prediction δH₁") 
            delta_h1_results = self._compute_delta_h1(batch_data)
            
            # Step 3: Compute actual entropy change ΔH via importance sampling
            self.logger.info("Computing actual entropy change ΔH via importance sampling")
            actual_entropy_results = self._compute_actual_entropy_change(batch_data)
            
            # Step 4: Compile results
            results = self._compile_results(delta_h1_results, actual_entropy_results)
            results['timing']['total_time'] = time.time() - start_time
            
            # Step 5: Save results if requested
            if self.config['output']['save_results']:
                self._save_results(results)
                
            self.results = results
            self.logger.info(f"Offline analysis completed in {results['timing']['total_time']:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during offline analysis: {e}")
            raise
            
    def _sample_batch(self) -> Dict[str, Any]:
        """Sample batch of prompts and responses."""
        # Use first B_E value from the unified config format
        B_E_values = self.config['batch_config']['B_E_values']
        B_E = B_E_values[0] if isinstance(B_E_values, list) else B_E_values
        
        return self.probe_components.sample_batch(
            B=B_E,
            G=self.config['batch_config']['G']
        )
        
    def _compute_delta_h1(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute first-order entropy change prediction."""
        # This coordinates the main computation pipeline
        return self.probe_components.compute_delta_h1(
            batch_data=batch_data,
            adam_preconditioner=self.adam_preconditioner,
            u_statistics=self.u_statistics,
            distributed_helpers=self.distributed_helpers,
            optimizer=self.optimizer  # 🔍 Pass optimizer for correct learning rate extraction
        )
        
    def _compute_actual_entropy_change(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute actual entropy change via importance sampling."""
        if self.importance_sampler is None:
            self.logger.warning("Importance sampling disabled - cannot compute actual entropy change")
            return {"error": "ImportanceSampler not initialized (importance sampling disabled)"}
        return self.importance_sampler.compute_entropy_change(batch_data, self.optimizer)
    
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
        
        try:
            self.logger.info("Starting Stage 1 Mixed E/U Batch Probe Analysis")
            
            # Load checkpoint if provided, otherwise use config paths
            if checkpoint_path is None:
                checkpoint_path = self.config['checkpoint']['checkpoint_path']
            
            # Load model and optimizer (following run_offline_analysis pattern)
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint
            if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model = self._initialize_model_from_checkpoint(checkpoint)
            else:
                # LoRA checkpoint directory
                self.model = self._load_lora_model(checkpoint_path)
            
            # Load optimizer state - use fallback logic like load_checkpoint()
            optimizer_path = self.config['checkpoint'].get('optimizer_path')
            
            if not optimizer_path:
                # Auto-discover optimizer path relative to checkpoint
                parent_dir = os.path.dirname(checkpoint_path)
                possible_optimizer_paths = [
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
                    f"Expected to find optimizer.pt in parent directory. "
                    f"Check checkpoint structure: model/ and optimizer.pt should be siblings."
                )
            
            if not os.path.exists(optimizer_path):
                raise FileNotFoundError(
                    f"Optimizer path does not exist: {optimizer_path}. "
                    f"Check that optimizer.pt was saved alongside the model checkpoint."
                )
            
            self.logger.info(f"Loading optimizer state from {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location='cpu')
            self.optimizer = self._initialize_optimizer_from_state(optimizer_state)
            
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
            mb_size_prompts = self.config.get('probe_rework', {}).get('mb_size_prompts', 2)
            weighting_mode = self.config.get('probe_rework', {}).get('weighting_mode', 'dr_grpo')
            
            self.logger.info(f"Mixed probe config: B_E={B_E}, B_U={B_U}, mb_size={mb_size_prompts}, weighting={weighting_mode}")
            
            # Stage 3: Deterministic E/U index selection for multi-GPU consistency
            is_dist, rank, world_size = distributed_helpers.get_dist_info()
            self.logger.info(f"Distributed info: dist={is_dist}, rank={rank}/{world_size}")
            
            # Load dataset to get size
            from rlp_datasets import DATASET_REGISTRY
            dataset = DATASET_REGISTRY[self.config['batch_config']['dataset_name']]
            ds_examples = dataset(self.config['batch_config']['split'])
            dataset_size = len(ds_examples)
            
            # Get master seed for deterministic sampling
            master_seed = self.config.get('probe_rework', {}).get('master_seed', 42)
            
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
            
            # Phase 0: Sample two batches using deterministic indices (if available)
            self.logger.info("Phase 0: Sampling E and U batches")
            phase0_start = time.time()
            
            # E-batch: Use replacement sampling with G=1
            # For distributed: each rank samples independently (statistically equivalent)
            G_U = self.config['batch_config']['G']  # U-batch uses configured G
            
            if is_dist:
                # Distributed: E-batch samples independently per rank
                E_total_sequences = B_E_local * 1  # G=1 for E-batch
                E_batch = self.probe_components.sample_E_batch_with_replacement(
                    E_total_sequences=E_total_sequences, 
                    G=1
                )
            else:
                # Single GPU: E-batch samples with replacement
                E_total_sequences = B_E * 1  # G=1 for E-batch
                E_batch = self.probe_components.sample_E_batch_with_replacement(
                    E_total_sequences=E_total_sequences, 
                    G=1
                )
            
            # U-batch: Keep existing distinct sampling with configured G
            U_batch = self.probe_components.sample_batch(
                B=B_U_local if is_dist else B_U, 
                G=G_U,
                indices=U_indices_local
            )
            
            phase0_time = time.time() - phase0_start
            self.logger.info(f"Phase 0 complete: {phase0_time:.2f}s")
            self.logger.info(f"E-batch: G=1 (replacement), {E_batch['sequences'].shape[0]} prompts, {E_batch['sequences'].shape[1]} responses/prompt")
            self.logger.info(f"U-batch: G={G_U} (distinct), {U_batch['sequences'].shape[0]} prompts, {U_batch['sequences'].shape[1]} responses/prompt")
            
            # Check if δH₁ computation should be performed
            probe_config = self.config.get('probe_rework', {})
            compute_delta_h1 = probe_config.get('compute_delta_h1', True)
            
            # Initialize variables for potential use in later stages
            delta_h1 = 0.0
            bars_dot = 0.0
            learning_rate = 0.0
            phase1_time = phase2_time = phase3_time = 0.0
            B_E_global = B_E_local if not is_dist else None
            B_U_global = B_U_local if not is_dist else None
            
            if compute_delta_h1:
                # Phase 1: Accumulate ΣX (raw gradients from E batch)
                self.logger.info("Phase 1: Accumulating ΣX from E batch")
                phase1_start = time.time()
                
                sum_X_buf, B_E_local = self.probe_components.accumulate_sum_X(
                    E_batch, mb_size_prompts, weighting_mode
                )
            
                phase1_time = time.time() - phase1_start
                self.logger.info(f"Phase 1 complete: {phase1_time:.2f}s, B_E_local={B_E_local}")
                
                # Phase 2: Accumulate ΣY (preconditioned gradients from U batch)
                self.logger.info("Phase 2: Accumulating ΣY from U batch")
                phase2_start = time.time()
                
                sum_Y_buf, B_U_local = self.probe_components.accumulate_sum_Y(
                    U_batch, mb_size_prompts, self.adam_preconditioner
                )
                
                phase2_time = time.time() - phase2_start
                self.logger.info(f"Phase 2 complete: {phase2_time:.2f}s, B_U_local={B_U_local}")
                
                # Phase 3: All-reduce and compute means and δH₁  
                self.logger.info("Phase 3: All-reduce and computing means and δH₁")
                phase3_start = time.time()
                
                # Stage 3: All-reduce counts and parameter buffers
                if is_dist:
                    self.logger.info("Multi-GPU: All-reducing counts and parameter buffers")
                    
                    # All-reduce counts
                    B_E_global = distributed_helpers.count_global(B_E_local)
                    B_U_global = distributed_helpers.count_global(B_U_local)
                    
                    # All-reduce parameter buffers (in-place)
                    distributed_helpers.all_reduce_param_buffer_(sum_X_buf)
                    distributed_helpers.all_reduce_param_buffer_(sum_Y_buf)
                    
                    self.logger.info(f"All-reduced: B_E_global={B_E_global}, B_U_global={B_U_global}")
                else:
                    # Single GPU: use local counts
                    B_E_global = B_E_local
                    B_U_global = B_U_local
                    self.logger.info(f"Single GPU: B_E={B_E_global}, B_U={B_U_global}")
                
                # Compute means: μ_X = ΣX / B_E_global, μ_Y = ΣY / B_U_global
                mu_X = {param_id: buf_tensor / max(B_E_global, 1) 
                       for param_id, buf_tensor in sum_X_buf.items()}
                mu_Y = {param_id: buf_tensor / max(B_U_global, 1) 
                       for param_id, buf_tensor in sum_Y_buf.items()}
                
                # Instrumentation to confirm numerical stability (from fix.txt)
                nz_X = sum(int(t.abs().sum().item() > 0) for t in mu_X.values())
                nz_Y = sum(int(t.abs().sum().item() > 0) for t in mu_Y.values())
                self.logger.info(f"[CHECK] nonzero μ_X entries: {nz_X}, nonzero μ_Y entries: {nz_Y}")
                norm_X = sum(float((t.double().pow(2)).sum().item()) for t in mu_X.values())**0.5
                norm_Y = sum(float((t.double().pow(2)).sum().item()) for t in mu_Y.values())**0.5
                self.logger.info(f"[CHECK] ||μ_X||₂={norm_X:.6e}, ||μ_Y||₂={norm_Y:.6e}")
                
                # Compute dot product: bars_dot = μ_X · μ_Y (now uses fp64 accumulation)
                bars_dot = self.probe_components.dot_param_buffers(mu_X, mu_Y)
                
                # Get learning rate and compute δH₁
                learning_rate = self.probe_components._get_learning_rate(self.optimizer)
                delta_h1 = learning_rate * bars_dot
                
                phase3_time = time.time() - phase3_start
                
                # Log results with high precision
                self.logger.info(f"🔍 [RESULTS] bars_dot={bars_dot:.10f}")
                self.logger.info(f"🔍 [RESULTS] learning_rate={learning_rate:.2e}")
                self.logger.info(f"🔍 [RESULTS] deltaH1={delta_h1:.10f}")
                self.logger.info(f"Phase 3 complete: {phase3_time:.2f}s")
                
                # Update globals for distributed case
                if is_dist:
                    # B_E_global and B_U_global already set above
                    pass
                else:
                    B_E_global = B_E_local
                    B_U_global = B_U_local
            else:
                self.logger.info("δH₁ computation disabled (compute_delta_h1=False)")
                # Set global batch sizes for downstream use
                if is_dist:
                    B_E_global = distributed_helpers.count_global(B_E_local)
                    B_U_global = distributed_helpers.count_global(B_U_local)
                else:
                    B_E_global = B_E_local
                    B_U_global = B_U_local
            
            # ================================================================
            # STAGE 2: Variance Components - Optional (configurable)
            # ================================================================
            compute_conditional_variance = probe_config.get('compute_conditional_variance', False)
            
            SE_deltaH1 = 0.0
            SE_conditional = 0.0
            phase4_time, phase5_time = 0.0, 0.0
            
            
            # ================================================================
            # STAGE 2: Conditional Variance Var_E(δH₁ | U) - Optional
            # ================================================================
            if compute_conditional_variance:
                self.logger.info("Phase 4b: Computing conditional variance Var_E(δH₁ | U)")
                phase4b_start = time.time()
                
                # Compute conditional variance using scalar projections
                sum_s_local, sum_s2_local, B_E_local = self.probe_components.compute_conditional_variance_over_E(
                    E_batch, mu_Y, mb_size_prompts, weighting_mode
                )
                
                # All-reduce scalar statistics
                if is_dist:
                    S1 = distributed_helpers.all_reduce_scalar_sum(
                        torch.tensor(sum_s_local, dtype=torch.float64, device='cpu')
                    )
                    S2 = distributed_helpers.all_reduce_scalar_sum(
                        torch.tensor(sum_s2_local, dtype=torch.float64, device='cpu')
                    )
                    B_E_cond = distributed_helpers.all_reduce_scalar_sum(
                        torch.tensor(B_E_local, dtype=torch.float64, device='cpu')  
                    )
                    self.logger.info(f"All-reduced conditional stats: S1={S1:.10f}, S2={S2:.10f}, B_E={B_E_cond:.0f}")
                else:
                    S1, S2, B_E_cond = sum_s_local, sum_s2_local, B_E_local
                
                # Compute conditional variance and standard error
                if B_E_cond >= 2:
                    s_bar = S1 / B_E_cond
                    sample_var_s = (S2 - B_E_cond * s_bar * s_bar) / (B_E_cond - 1)
                    var_cond_E = (learning_rate * learning_rate) * sample_var_s / B_E_cond
                    SE_conditional = learning_rate * math.sqrt(max(sample_var_s / B_E_cond, 0.0))
                else:
                    sample_var_s, var_cond_E, SE_conditional = 0.0, 0.0, 0.0
                    self.logger.warning("B_E < 2, cannot compute conditional variance")
                
                phase4b_time = time.time() - phase4b_start
                self.logger.info(f"🔍 [CONDITIONAL VARIANCE] Var_E(δH₁|U)={var_cond_E:.10f}")
                self.logger.info(f"🔍 [CONDITIONAL VARIANCE] SE_E(δH₁|U)={SE_conditional:.10f}")
                self.logger.info(f"Phase 4b complete: {phase4b_time:.2f}s")
            
            # ================================================================
            # STAGE 2c: Two-Batch Ground-Truth Entropy Change - Optional
            # ================================================================
            compute_importance_sampling = probe_config.get('compute_importance_sampling', False)
            importance_enabled = self.config.get('importance', {}).get('enabled', False) or compute_importance_sampling
            ground_truth_results = {}
            
            if importance_enabled:
                self.logger.info("Phase 5: Computing two-batch ground-truth entropy change")
                phase5_start = time.time()
                
                # Initialize importance sampler if not already done
                if not hasattr(self, 'importance_sampler') or self.importance_sampler is None:
                    from .importance_sampling import ImportanceSampler
                    self.importance_sampler = ImportanceSampler(self.model, self.config, self.logger)
                
                # Extract importance sampling configuration
                cfg_importance = {
                    'training_loss': self.config.get('importance', {}).get('training_loss', 'nll'),
                    'importance_microbatch_size': self.config.get('importance', {}).get('importance_microbatch_size', 1),
                    'is_mode': self.config.get('importance', {}).get('is_mode', 'snis'),
                    'clip_c': self.config.get('importance', {}).get('clip_c', 10.0),
                    'report_per_token': self.config.get('importance', {}).get('report_per_token', False),
                    'snapshot_device': self.config.get('importance', {}).get('snapshot_device', 'cpu')
                }
                
                # Compute ground-truth entropy change
                ground_truth_results = self.importance_sampler.entropy_change_two_batch(
                    self.model, E_batch, U_batch, self.optimizer, cfg_importance
                )
                
                phase5_time = time.time() - phase5_start
                self.logger.info(f"🔍 [GROUND-TRUTH] deltaH_true={ground_truth_results['deltaH_true']:.10f}")
                self.logger.info(f"Phase 5 complete: {phase5_time:.2f}s")
            
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
            
            # Add Stage 2 variance results if computed
            if compute_vx_vy_variance:
                results.update({
                    "V_X": V_X,
                    "V_Y": V_Y, 
                    "SE_deltaH1": SE_deltaH1,
                    "compute_vx_vy_variance": True
                })
                results["timing"]["phase4_variance"] = phase4_time
            
            # Add conditional variance results if computed
            if compute_conditional_variance:
                results.update({
                    "SE_conditional": SE_conditional,
                    "compute_conditional_variance": True
                })
                if "phase4b_time" in locals():
                    results["timing"]["phase4b_conditional"] = phase4b_time
            
            # Add Stage 2 ground-truth results if computed
            if importance_enabled:
                results.update({
                    **ground_truth_results,
                    "importance_enabled": True
                })
                results["timing"]["phase5_importance"] = phase5_time
            
            stage = "Stage 1+2" if (compute_vx_vy_variance or compute_conditional_variance or importance_enabled) else "Stage 1"
            self.logger.info(f"{stage} Mixed probe analysis completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during mixed probe analysis: {e}")
            raise
        
    def _compile_results(self, delta_h1_results: Dict[str, Any], 
                        actual_entropy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results dictionary."""
        return {
            # Core results
            "U_cross": delta_h1_results["U_cross"],
            "deltaH1": delta_h1_results["deltaH1"], 
            
            # Standard error estimates
            "se_plugin": delta_h1_results["se_plugin"],
            "zeta1_plugin": delta_h1_results["zeta1_plugin"],
            "se_jack": delta_h1_results["se_jack"],
            "zeta1_jack": delta_h1_results["zeta1_jack"],
            
            # Actual entropy change
            "deltaH_snis": actual_entropy_results["deltaH_snis"],
            "ESS": actual_entropy_results["ESS"],
            "psis_k": actual_entropy_results.get("psis_k", None),
            
            # Timing and diagnostics
            "timing": {
                **delta_h1_results.get("timing", {}),
                **actual_entropy_results.get("timing", {})
            },
            "config": self.config,
            
            # Additional diagnostics
            "diagnostics": {
                **delta_h1_results.get("diagnostics", {}),
                **actual_entropy_results.get("diagnostics", {})
            }
        }
        
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
            
    def test_minimal_alpha_trick(self, batch_ids: torch.Tensor, batch_mask: torch.Tensor) -> None:
        """
        Minimal α-trick probe test from fix.txt to localize gradient flow issues.
        
        This implements the diagnostic test from fix.txt section C to verify:
        1. Model forward produces gradients
        2. LoRA parameters receive gradients from a simple loss
        
        Args:
            batch_ids: Input token IDs [batch_size, seq_len]
            batch_mask: Attention mask [batch_size, seq_len]
        """
        self.logger.info("Running minimal α-trick probe test...")
        
        # Ensure model is in training mode
        self.model.train()
        
        # 1) Tiny TF forward on the PEFT model
        outs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = outs.logits  # must have grad_fn
        
        # Assert logits are tracking gradients
        if not (logits.requires_grad and logits.grad_fn is not None):
            raise RuntimeError("Minimal probe: logits not tracking grad - model may be frozen or in eval mode")
        
        # 2) Build S and a vectorized loss L = <alpha, L_vec(S)>
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = logprobs[:, :-1].gather(2, batch_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        S = token_lp.sum(dim=1).view(-1, 1)  # [k, 1] or adapt to your [k,G]
        S.retain_grad()
        
        alpha = torch.ones(S.shape[0], device=S.device, requires_grad=True)
        L = (-(alpha.view(-1, 1) * S).mean())  # minus sign; simple cross-entropy variant
        
        # 3) Try a *known* LoRA param first (more robust than "all params")
        lora_params = [p for n, p in self.model.named_parameters() if ("lora_A" in n or "lora_B" in n)]
        
        if len(lora_params) == 0:
            raise RuntimeError("Minimal probe: No lora_A/B parameters found (adapters missing/merged?)")
        
        self.logger.info(f"Found {len(lora_params)} LoRA parameters for gradient test")
        
        g_lora = torch.autograd.grad(L, lora_params, allow_unused=True, retain_graph=True)
        non_none_lora = sum(int(g is not None) for g in g_lora)
        
        self.logger.info(f"Minimal probe results: {non_none_lora}/{len(lora_params)} LoRA params received gradients")
        
        if non_none_lora == 0:
            raise RuntimeError("Minimal probe FAILED: Forward isn't using the same adapter tensors you're differentiating w.r.t. (wrong model object or adapters disabled)")
        else:
            self.logger.info("Minimal probe PASSED: LoRA gradient flow is working")
            
        # Test all parameters as well
        all_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(all_params) == 0:
            raise RuntimeError("Minimal probe: No trainable parameters found - all parameters frozen!")
            
        g_all = torch.autograd.grad(L, all_params, allow_unused=True, retain_graph=True)
        non_none_all = sum(int(g is not None) for g in g_all)
        
        self.logger.info(f"Full gradient test: {non_none_all}/{len(all_params)} total params received gradients")
        
        if non_none_all == 0:
            raise RuntimeError("Minimal probe FAILED: No parameters received gradients from L")
            
    @classmethod
    def from_config_file(cls, config_path: str) -> 'OfflineEntropyProbe':
        """Create probe instance from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
        
    def get_config_template(self) -> str:
        """Return the path to the config template for reference."""
        return str(Path(__file__).parent / "configs" / "probe_config_template.yaml")