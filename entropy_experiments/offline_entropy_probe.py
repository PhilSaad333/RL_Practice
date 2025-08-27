"""
Offline Entropy Probe

Main orchestrator class for the offline entropy probe analysis.
Implements the strategy outlined in offline_entropy_probe_strategy.txt
based on the theory from RL_studies.pdf.

Key functionality:
- Load model checkpoint and optimizer state
- Sample batch of prompts and responses  
- Compute first-order entropy change prediction δH₁ via U-statistic
- Measure actual entropy change ΔH via importance sampling
- Provide variance estimates and statistical diagnostics

Usage:
    probe = OfflineEntropyProbe(config)
    results = probe.run_offline_analysis(checkpoint_path)
"""

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
from .u_statistics import UStatisticsCalculator
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
        
        # Initialize distributed if needed
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
        """Load LoRA model following rl_runner.py pattern."""
        # Import here to match rl_runner.py pattern
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training
        
        # Get backbone from config or infer from LoRA adapter
        backbone = self.config['checkpoint'].get('model_config_path', 'Qwen/Qwen2.5-1.5B')
        
        # Setup quantization config (exact copy from rl_runner.py)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model (exact copy from rl_runner.py pattern)
        base = AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb
        )
        base = prepare_model_for_kbit_training(base)  # PEFT/QLoRA prep
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        
        # Load LoRA adapter (exact copy from rl_runner.py pattern)
        model = PeftModel.from_pretrained(base, lora_path, is_trainable=True)
        model.enable_input_require_grads()
        
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
        model.gradient_checkpointing_enable()
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
        
        # Load the saved state
        try:
            # Handle different state formats
            if isinstance(optimizer_state, dict):
                if 'state_dict' in optimizer_state:
                    # State wrapped in a dictionary
                    optimizer.load_state_dict(optimizer_state['state_dict'])
                elif 'param_groups' in optimizer_state:
                    # Direct optimizer state dict
                    optimizer.load_state_dict(optimizer_state)
                else:
                    # Assume the whole thing is the state dict
                    optimizer.load_state_dict(optimizer_state)
            else:
                raise ValueError(f"Unexpected optimizer_state type: {type(optimizer_state)}")
                
            self.logger.info("Successfully loaded optimizer state")
            
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
        
        self.importance_sampler = ImportanceSampler(
            model=self.model,
            config=self.config,
            logger=self.logger
        )
        
        self.u_statistics = UStatisticsCalculator(
            config=self.config,
            logger=self.logger
        )
        
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
        # This will be implemented by ProbeComponents
        return self.probe_components.sample_batch(
            B=self.config['batch_config']['B'],
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
            
            # Load optimizer state
            optimizer_path = self.config['checkpoint']['optimizer_path']
            if os.path.exists(optimizer_path):
                self.logger.info(f"Loading optimizer state from {optimizer_path}")
                optimizer_state = torch.load(optimizer_path, map_location='cpu')
                self.optimizer = self._initialize_optimizer_from_state(optimizer_state)
            else:
                self.logger.warning(f"Optimizer path not found: {optimizer_path}, creating fresh optimizer")
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-6)
            
            # Initialize components
            self._initialize_components()
            
            # Get config parameters
            B_E = self.config['batch_config'].get('B_E', self.config['batch_config']['B'])
            B_U = self.config['batch_config'].get('B_U', self.config['batch_config']['B'])
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
            
            E_batch = self.probe_components.sample_batch(
                B=B_E_local if is_dist else B_E, 
                G=self.config['batch_config']['G'],
                indices=E_indices_local
            )
            U_batch = self.probe_components.sample_batch(
                B=B_U_local if is_dist else B_U, 
                G=self.config['batch_config']['G'],
                indices=U_indices_local
            )
            
            phase0_time = time.time() - phase0_start
            self.logger.info(f"Phase 0 complete: {phase0_time:.2f}s")
            
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
            
            # Compute dot product: bars_dot = μ_X · μ_Y
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
            
            # ================================================================
            # STAGE 2: Variance Components - Optional (configurable)
            # ================================================================
            probe_config = self.config.get('probe_rework', {})
            compute_vx_vy_variance = probe_config.get('compute_vx_vy_variance', False)
            compute_conditional_variance = probe_config.get('compute_conditional_variance', False)
            
            V_X, V_Y, SE_deltaH1 = 0.0, 0.0, 0.0
            SE_conditional = 0.0
            phase4_time, phase5_time = 0.0, 0.0
            
            if compute_vx_vy_variance:
                self.logger.info("Phase 4: Computing variance components V_X and V_Y")
                phase4_start = time.time()
                
                # Compute V_X via per-unit scalar projections on E batch
                V_X_local = self.probe_components.compute_VX(
                    E_batch, mu_X, mu_Y, mb_size_prompts, weighting_mode
                )
                
                # Compute V_Y via per-unit scalar projections on U batch
                V_Y_local = self.probe_components.compute_VY(
                    U_batch, mu_X, mu_Y, mb_size_prompts, self.adam_preconditioner
                )
                
                # All-reduce variance components via tensor all-reduce
                if is_dist:
                    V_X = distributed_helpers.all_reduce_scalar_sum(
                        torch.tensor(V_X_local, dtype=torch.float64, device='cpu')
                    )
                    V_Y = distributed_helpers.all_reduce_scalar_sum(
                        torch.tensor(V_Y_local, dtype=torch.float64, device='cpu')
                    )
                    self.logger.info(f"All-reduced variances: V_X_sum={V_X:.10f}, V_Y_sum={V_Y:.10f}")
                else:
                    V_X = V_X_local
                    V_Y = V_Y_local
                
                # Apply final normalization: V_X /= (B_E * max(B_E-1, 1))
                V_X = V_X / (B_E_global * max(B_E_global - 1, 1))
                V_Y = V_Y / (B_U_global * max(B_U_global - 1, 1))
                
                # Compute standard error (no more fractional variance)
                SE_deltaH1 = learning_rate * math.sqrt(max(V_X, 0.0) + max(V_Y, 0.0))
                
                phase4_time = time.time() - phase4_start
                self.logger.info(f"🔍 [V_X+V_Y VARIANCE] V_X={V_X:.10f}, V_Y={V_Y:.10f}")
                self.logger.info(f"🔍 [V_X+V_Y VARIANCE] SE_deltaH1={SE_deltaH1:.10f}")
                self.logger.info(f"Phase 4 complete: {phase4_time:.2f}s")
            
            # ================================================================
            # STAGE 2b: Conditional Variance Var_E(δH₁ | U) - Optional
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
            
    @classmethod
    def from_config_file(cls, config_path: str) -> 'OfflineEntropyProbe':
        """Create probe instance from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
        
    def get_config_template(self) -> str:
        """Return the path to the config template for reference."""
        return str(Path(__file__).parent / "configs" / "probe_config_template.yaml")