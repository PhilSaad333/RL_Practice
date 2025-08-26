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
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import json

from .probe_components import ProbeComponents
from .adam_preconditioner import AdamPreconditioner  
from .importance_sampling import ImportanceSampler
from .u_statistics import UStatisticsCalculator
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