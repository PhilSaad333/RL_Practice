#!/usr/bin/env python3
"""
🔬 Gradient Isolation Test Script

This script isolates the gradient flow issues we're experiencing in the entropy probe
by testing all the components separately before running the full probe.

Based on fix.txt analysis, this tests:
1. LoRA parameter loading and trainability  
2. Optimizer state loading (√v values)
3. Parameter identity consistency (DDP vs PEFT)
4. Adam preconditioner behavior with/without state
5. Simple gradient computation using different approaches
6. Forward pass LoRA activation verification

Target: Load RL checkpoint step_40 and isolate the zero bars_dot issue.
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# Setup imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Core imports  
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model_state_dict
import yaml

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GradientIsolationTest:
    """Isolated gradient flow testing for RL checkpoint debugging."""
    
    def __init__(self, checkpoint_path: str):
        """Initialize with RL checkpoint path."""
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Results storage
        self.results = {
            "checkpoint_loading": {},
            "parameter_analysis": {},
            "optimizer_state_analysis": {},
            "forward_pass_tests": {},
            "gradient_tests": {},
            "preconditioner_tests": {}
        }
        
    def run_all_tests(self):
        """Run complete battery of gradient isolation tests."""
        logger.info("🔬 Starting comprehensive gradient isolation tests...")
        
        try:
            # Phase 1: Load and analyze checkpoint
            self.test_checkpoint_loading()
            self.test_parameter_analysis()  
            self.test_optimizer_state_loading()
            
            # Phase 2: Forward pass and LoRA verification
            self.test_forward_pass_lora_activation()
            
            # Phase 3: Gradient computation isolation
            self.test_parameter_identity_consistency()
            self.test_basic_gradient_computation()
            self.test_preconditioner_behavior()
            
            # Phase 4: Simulate entropy probe conditions
            self.test_entropy_probe_simulation()
            
            # Generate final report
            self.generate_diagnostic_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_checkpoint_loading(self):
        """Test 1: Verify RL checkpoint loading."""
        logger.info("🔧 Test 1: Checkpoint Loading")
        
        # Check checkpoint structure
        model_path = self.checkpoint_path / "model"
        optimizer_path = self.checkpoint_path / "optimizer.pt"
        scheduler_path = self.checkpoint_path / "scheduler.pt" 
        training_info_path = self.checkpoint_path / "training_info.json"
        
        loading_status = {
            "model_dir_exists": model_path.exists(),
            "optimizer_file_exists": optimizer_path.exists(),
            "scheduler_file_exists": scheduler_path.exists(),
            "training_info_exists": training_info_path.exists()
        }
        
        logger.info(f"Checkpoint structure: {loading_status}")
        
        # Load training info if available
        if training_info_path.exists():
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                logger.info(f"Training info: {training_info}")
                
        # Load base model and tokenizer
        logger.info("Loading base Qwen2.5-1.5B model...")
        base_model_name = "Qwen/Qwen2.5-1.5B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load PEFT model
        logger.info("Loading PEFT model from checkpoint...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
            is_trainable=True
        )
        
        # Enable input gradients for QLoRA
        self.model.enable_input_require_grads()
        
        loading_status["model_loaded"] = True
        loading_status["tokenizer_loaded"] = True
        
        self.results["checkpoint_loading"] = loading_status
        logger.info("✅ Checkpoint loading completed")
    
    def test_parameter_analysis(self):
        """Test 2: Analyze which parameters are trainable."""
        logger.info("🔧 Test 2: Parameter Analysis")
        
        # Get all parameters
        all_params = list(self.model.named_parameters())
        trainable_params = [(n, p) for n, p in all_params if p.requires_grad]
        frozen_params = [(n, p) for n, p in all_params if not p.requires_grad]
        
        # Analyze LoRA parameters specifically
        lora_params = [(n, p) for n, p in trainable_params if "lora_A" in n or "lora_B" in n]
        lm_head_params = [(n, p) for n, p in trainable_params if "lm_head" in n]
        other_trainable = [(n, p) for n, p in trainable_params 
                          if not ("lora_A" in n or "lora_B" in n or "lm_head" in n)]
        
        analysis = {
            "total_parameters": len(all_params),
            "trainable_parameters": len(trainable_params),
            "frozen_parameters": len(frozen_params),
            "lora_parameters": len(lora_params),
            "lm_head_parameters": len(lm_head_params),
            "other_trainable_parameters": len(other_trainable),
            "trainable_param_names": [n for n, _ in trainable_params[:10]],  # First 10
            "lora_param_names": [n for n, _ in lora_params[:5]],  # First 5 LoRA
        }
        
        logger.info(f"Parameter analysis: {analysis}")
        
        # Check parameter device and dtype
        if lora_params:
            sample_lora = lora_params[0][1]
            analysis["lora_device"] = str(sample_lora.device)
            analysis["lora_dtype"] = str(sample_lora.dtype)
            analysis["lora_requires_grad"] = sample_lora.requires_grad
            
        self.results["parameter_analysis"] = analysis
        logger.info("✅ Parameter analysis completed")
    
    def test_optimizer_state_loading(self):
        """Test 3: Load optimizer state and check √v values."""
        logger.info("🔧 Test 3: Optimizer State Loading")
        
        optimizer_path = self.checkpoint_path / "optimizer.pt"
        
        if not optimizer_path.exists():
            logger.warning("❌ No optimizer.pt found")
            self.results["optimizer_state_analysis"] = {"optimizer_found": False}
            return
            
        # Load optimizer state
        logger.info("Loading optimizer state...")
        optimizer_state = torch.load(optimizer_path, map_location=self.device)
        
        # Create dummy optimizer to load state into
        from torch.optim import AdamW
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        dummy_optimizer = AdamW(trainable_params, lr=1e-4)
        
        try:
            dummy_optimizer.load_state_dict(optimizer_state)
            logger.info("✅ Optimizer state loaded successfully")
            
            # Analyze Adam state for LoRA parameters
            lora_params = [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
            
            adam_analysis = {
                "total_lora_params": len(lora_params),
                "params_with_exp_avg": 0,
                "params_with_exp_avg_sq": 0,
                "nonzero_exp_avg_sq": 0,
                "exp_avg_sq_norms": []
            }
            
            for param in lora_params:
                state = dummy_optimizer.state.get(param, {})
                if "exp_avg" in state:
                    adam_analysis["params_with_exp_avg"] += 1
                if "exp_avg_sq" in state:
                    adam_analysis["params_with_exp_avg_sq"] += 1
                    exp_avg_sq_norm = float(state["exp_avg_sq"].norm().item())
                    adam_analysis["exp_avg_sq_norms"].append(exp_avg_sq_norm)
                    if exp_avg_sq_norm > 1e-8:
                        adam_analysis["nonzero_exp_avg_sq"] += 1
            
            # Store optimizer for later tests
            self.optimizer = dummy_optimizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimizer state: {e}")
            adam_analysis = {"loading_failed": str(e)}
            
        state_analysis = {
            "optimizer_found": True,
            "adam_analysis": adam_analysis
        }
        
        logger.info(f"Optimizer state analysis: {state_analysis}")
        self.results["optimizer_state_analysis"] = state_analysis
        logger.info("✅ Optimizer state analysis completed")
    
    def test_forward_pass_lora_activation(self):
        """Test 4: Verify LoRA activation in forward pass."""
        logger.info("🔧 Test 4: Forward Pass LoRA Activation")
        
        # Create small test batch
        test_prompts = [
            "What is 2 + 2?",
            "The capital of France is",
            "In mathematics, a prime number is"
        ]
        
        # Tokenize
        inputs = self.tokenizer(
            test_prompts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(self.device)
        
        logger.info(f"Test batch shape: {inputs.input_ids.shape}")
        
        # Test LoRA on/off comparison (as suggested in fix.txt)
        with torch.no_grad():
            # Forward with LoRA ON
            self.model.set_adapter("default")  # Ensure adapter is active
            logits_on = self.model(**inputs).logits
            
            # Forward with LoRA OFF
            with self.model.disable_adapter():
                logits_off = self.model(**inputs).logits
        
        # Compute difference
        logits_delta = (logits_on - logits_off).abs()
        max_delta = float(logits_delta.max().item())
        mean_delta = float(logits_delta.mean().item())
        
        activation_results = {
            "batch_size": inputs.input_ids.shape[0],
            "sequence_length": inputs.input_ids.shape[1],
            "vocab_size": logits_on.shape[-1],
            "max_lora_delta": max_delta,
            "mean_lora_delta": mean_delta,
            "lora_active": max_delta > 1e-7  # Threshold from original code
        }
        
        logger.info(f"LoRA activation results: {activation_results}")
        
        # Store inputs for gradient tests
        self.test_inputs = inputs
        self.test_logits_on = logits_on
        
        self.results["forward_pass_tests"] = activation_results
        logger.info("✅ Forward pass LoRA activation test completed")
    
    def test_parameter_identity_consistency(self):
        """Test 5: Check parameter identity consistency (DDP vs PEFT)."""
        logger.info("🔧 Test 5: Parameter Identity Consistency")
        
        # Get parameters from different paths (simulating the entropy probe issue)
        
        # Path 1: Direct model parameters (what buffers might iterate)
        direct_params = list(self.model.parameters())
        direct_trainable = [p for p in direct_params if p.requires_grad]
        
        # Path 2: PEFT model parameters (what forward uses)
        peft_model = self.model.module if hasattr(self.model, "module") else self.model
        peft_params = list(peft_model.parameters())
        peft_trainable = [p for p in peft_params if p.requires_grad]
        
        # Path 3: LoRA-specific parameters  
        lora_params = [p for n, p in self.model.named_parameters() 
                      if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        
        # Check identity consistency
        direct_ids = set(id(p) for p in direct_trainable)
        peft_ids = set(id(p) for p in peft_trainable)
        lora_ids = set(id(p) for p in lora_params)
        
        consistency_results = {
            "direct_trainable_count": len(direct_trainable),
            "peft_trainable_count": len(peft_trainable),  
            "lora_param_count": len(lora_params),
            "direct_peft_identical": direct_ids == peft_ids,
            "lora_subset_of_direct": lora_ids.issubset(direct_ids),
            "lora_subset_of_peft": lora_ids.issubset(peft_ids),
            "id_overlap_direct_peft": len(direct_ids & peft_ids),
            "has_ddp_wrapper": hasattr(self.model, "module")
        }
        
        logger.info(f"Parameter identity consistency: {consistency_results}")
        
        # Store consistent parameter list for later tests
        self._consistent_trainable_params = lora_params  # Use LoRA as ground truth
        
        self.results["gradient_tests"]["parameter_consistency"] = consistency_results
        logger.info("✅ Parameter identity consistency test completed")
    
    def test_basic_gradient_computation(self):
        """Test 6: Basic gradient computation with different approaches.""" 
        logger.info("🔧 Test 6: Basic Gradient Computation")
        
        # Ensure model is in train mode
        self.model.train()
        
        # Clear any existing gradients
        self.model.zero_grad()
        
        # Method 1: Standard backward pass
        logger.info("Testing Method 1: Standard backward pass")
        
        # Simple loss: sum of logits for first token
        logits = self.model(**self.test_inputs).logits
        simple_loss = logits[:, 0, :].sum()  # Sum logits for first token
        
        simple_loss.backward()
        
        # Check gradients on LoRA parameters
        method1_results = self._analyze_gradients_after_backward("method1_standard")
        
        # Clear gradients
        self.model.zero_grad()
        
        # Method 2: autograd.grad approach (similar to entropy probe)
        logger.info("Testing Method 2: autograd.grad approach")
        
        # Use autograd.grad instead of backward
        lora_params = self._consistent_trainable_params
        
        try:
            grads = torch.autograd.grad(
                simple_loss,
                lora_params,
                retain_graph=True,
                create_graph=False
            )
            
            method2_results = {
                "autograd_grad_success": True,
                "returned_gradients": len([g for g in grads if g is not None]),
                "none_gradients": len([g for g in grads if g is None]),
                "total_params": len(lora_params),
                "gradient_norms": [float(g.norm().item()) if g is not None else 0.0 
                                  for g in grads[:5]]  # First 5
            }
            
            # Manually assign gradients (simulate entropy probe buffer filling)
            for param, grad in zip(lora_params, grads):
                if grad is not None:
                    param.grad = grad.detach()
                    
            method2_gradients = self._analyze_gradients_after_backward("method2_autograd")
            method2_results.update(method2_gradients)
            
        except Exception as e:
            logger.error(f"autograd.grad failed: {e}")
            method2_results = {"autograd_grad_failed": str(e)}
        
        gradient_results = {
            "method1_standard_backward": method1_results,
            "method2_autograd_grad": method2_results
        }
        
        logger.info(f"Basic gradient computation results: {gradient_results}")
        self.results["gradient_tests"]["basic_computation"] = gradient_results
        logger.info("✅ Basic gradient computation test completed")
    
    def test_preconditioner_behavior(self):
        """Test 7: Adam preconditioner behavior with/without optimizer state."""
        logger.info("🔧 Test 7: Preconditioner Behavior")
        
        if self.optimizer is None:
            logger.warning("❌ No optimizer available for preconditioner test")
            self.results["preconditioner_tests"] = {"no_optimizer": True}
            return
            
        # Get LoRA parameters with gradients
        lora_params = self._consistent_trainable_params
        
        # Ensure we have gradients (from previous test)
        params_with_grads = [p for p in lora_params if p.grad is not None]
        logger.info(f"Testing preconditioner on {len(params_with_grads)} params with gradients")
        
        if not params_with_grads:
            logger.warning("❌ No gradients available for preconditioner test")
            self.results["preconditioner_tests"] = {"no_gradients": True}
            return
            
        # Import Adam preconditioner (similar to entropy probe)
        try:
            # Simplified Adam preconditioner logic
            def apply_adam_preconditioner(grad, param, optimizer):
                """Apply Adam-style preconditioning: grad / sqrt(exp_avg_sq + eps)"""
                state = optimizer.state.get(param, {})
                
                if "exp_avg_sq" not in state:
                    logger.warning(f"No exp_avg_sq for param {id(param)}")
                    return None
                    
                exp_avg_sq = state["exp_avg_sq"]
                eps = 1e-8
                
                # Adam preconditioning
                preconditioned = grad / (torch.sqrt(exp_avg_sq) + eps)
                return preconditioned
            
            preconditioner_results = {
                "params_tested": len(params_with_grads),
                "successful_preconditioning": 0,
                "failed_preconditioning": 0,
                "original_grad_norms": [],
                "preconditioned_grad_norms": []
            }
            
            for param in params_with_grads[:3]:  # Test first 3 params
                original_grad = param.grad.clone()
                original_norm = float(original_grad.norm().item())
                
                preconditioned_grad = apply_adam_preconditioner(original_grad, param, self.optimizer)
                
                if preconditioned_grad is not None:
                    preconditioned_norm = float(preconditioned_grad.norm().item())
                    preconditioner_results["successful_preconditioning"] += 1
                    preconditioner_results["preconditioned_grad_norms"].append(preconditioned_norm)
                else:
                    preconditioner_results["failed_preconditioning"] += 1
                    preconditioner_results["preconditioned_grad_norms"].append(0.0)
                    
                preconditioner_results["original_grad_norms"].append(original_norm)
            
        except Exception as e:
            logger.error(f"Preconditioner test failed: {e}")
            preconditioner_results = {"preconditioner_failed": str(e)}
            
        logger.info(f"Preconditioner behavior results: {preconditioner_results}")
        self.results["preconditioner_tests"] = preconditioner_results
        logger.info("✅ Preconditioner behavior test completed")
    
    def test_entropy_probe_simulation(self):
        """Test 8: Simulate entropy probe conditions."""
        logger.info("🔧 Test 8: Entropy Probe Simulation")
        
        # Simulate the exact conditions in the entropy probe that lead to zero bars_dot
        
        # Create parameter buffers (like sum_X_buf, sum_Y_buf)
        def create_param_buffer(params):
            """Create buffer like zeros_like_params in probe."""
            buffer = {}
            for param in params:
                buffer[id(param)] = torch.zeros_like(param.data)
            return buffer
        
        def add_to_buffer(buffer, params):
            """Add gradients to buffer like add_into_param_buffer."""
            for param in params:
                if param.grad is not None:
                    buffer[id(param)] += param.grad.data
                    
        def compute_buffer_dot(buf1, buf2):
            """Compute dot product between buffers."""
            total_dot = 0.0
            for param_id in buf1:
                if param_id in buf2:
                    total_dot += float((buf1[param_id] * buf2[param_id]).sum().item())
            return total_dot
            
        # Create buffers for X and Y paths
        lora_params = self._consistent_trainable_params
        
        sum_X_buf = create_param_buffer(lora_params)
        sum_Y_buf = create_param_buffer(lora_params)
        
        # Simulate X-path: compute loss and gradients
        self.model.zero_grad()
        
        # X-path loss (simplified)
        logits_X = self.model(**self.test_inputs).logits
        
        # Simulate leave-one-out construction (simplified)
        # In real probe: L_X = (S_w - S_w_LOO.detach()) * prompt_mean
        loss_X = logits_X[:, 0, :100].sum()  # Simplified loss
        
        loss_X.backward(retain_graph=True)
        add_to_buffer(sum_X_buf, lora_params)
        
        # Compute buffer norm for X
        def buffer_norm(buf):
            total = 0.0
            for t in buf.values():
                total += float((t * t).sum().item())
            return total ** 0.5
            
        x_norm = buffer_norm(sum_X_buf)
        
        # Simulate Y-path: different loss and gradients  
        self.model.zero_grad()
        
        # Y-path loss (different from X)
        logits_Y = self.model(**self.test_inputs).logits 
        loss_Y = logits_Y[:, 1, :100].sum()  # Different slice
        
        loss_Y.backward(retain_graph=True)
        
        # Apply preconditioner if available (like Phase-2 in probe)
        if self.optimizer is not None:
            for param in lora_params:
                if param.grad is not None:
                    state = self.optimizer.state.get(param, {})
                    if "exp_avg_sq" in state:
                        eps = 1e-8
                        preconditioned = param.grad / (torch.sqrt(state["exp_avg_sq"]) + eps)
                        param.grad.copy_(preconditioned)
        
        add_to_buffer(sum_Y_buf, lora_params)
        y_norm = buffer_norm(sum_Y_buf)
        
        # Compute bars_dot (the critical metric that's zero in entropy probe)
        bars_dot = compute_buffer_dot(sum_X_buf, sum_Y_buf)
        
        simulation_results = {
            "x_buffer_norm": x_norm,
            "y_buffer_norm": y_norm,
            "bars_dot": bars_dot,
            "bars_dot_zero": abs(bars_dot) < 1e-10,
            "buffer_sizes": {
                "sum_X_buf": len(sum_X_buf),
                "sum_Y_buf": len(sum_Y_buf)
            }
        }
        
        logger.info(f"Entropy probe simulation results: {simulation_results}")
        self.results["gradient_tests"]["entropy_simulation"] = simulation_results
        logger.info("✅ Entropy probe simulation completed")
    
    def _analyze_gradients_after_backward(self, test_name: str) -> Dict:
        """Helper to analyze gradient state after backward pass."""
        lora_params = self._consistent_trainable_params
        
        analysis = {
            "total_lora_params": len(lora_params),
            "params_with_grad": 0,
            "params_with_nonzero_grad": 0,
            "gradient_norms": [],
            "total_gradient_norm": 0.0
        }
        
        total_norm_sq = 0.0
        for param in lora_params:
            if param.grad is not None:
                analysis["params_with_grad"] += 1
                grad_norm = float(param.grad.norm().item())
                analysis["gradient_norms"].append(grad_norm)
                total_norm_sq += grad_norm ** 2
                
                if grad_norm > 1e-8:
                    analysis["params_with_nonzero_grad"] += 1
            else:
                analysis["gradient_norms"].append(0.0)
                
        analysis["total_gradient_norm"] = float(total_norm_sq ** 0.5)
        
        logger.info(f"{test_name} gradient analysis: {analysis}")
        return analysis
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        logger.info("📋 Generating Diagnostic Report")
        
        report = {
            "summary": {
                "checkpoint_loaded": self.results["checkpoint_loading"].get("model_loaded", False),
                "lora_params_found": self.results["parameter_analysis"].get("lora_parameters", 0),
                "optimizer_state_loaded": self.results["optimizer_state_analysis"].get("optimizer_found", False),
                "lora_active_in_forward": self.results["forward_pass_tests"].get("lora_active", False),
                "gradients_computed": self.results["gradient_tests"].get("basic_computation", {}).get("method1_standard", {}).get("params_with_nonzero_grad", 0) > 0,
                "bars_dot_zero": self.results["gradient_tests"].get("entropy_simulation", {}).get("bars_dot_zero", True)
            },
            "detailed_results": self.results
        }
        
        # Save report
        report_path = Path("entropy_experiments/gradient_isolation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print key findings
        logger.info("🔍 KEY FINDINGS:")
        logger.info(f"  ✓ Checkpoint loaded: {report['summary']['checkpoint_loaded']}")
        logger.info(f"  ✓ LoRA parameters: {report['summary']['lora_params_found']}")
        logger.info(f"  ✓ Optimizer state: {report['summary']['optimizer_state_loaded']}")
        logger.info(f"  ✓ LoRA active in forward: {report['summary']['lora_active_in_forward']}")
        logger.info(f"  ✓ Gradients computed: {report['summary']['gradients_computed']}")
        logger.info(f"  ❌ bars_dot zero: {report['summary']['bars_dot_zero']}")
        
        # Recommend next steps based on results
        self._recommend_next_steps(report)
        
        logger.info(f"📄 Full report saved to: {report_path}")
        logger.info("✅ Diagnostic report generation completed")
    
    def _recommend_next_steps(self, report):
        """Recommend next steps based on test results.""" 
        logger.info("🎯 RECOMMENDED NEXT STEPS:")
        
        summary = report["summary"]
        
        if not summary["lora_active_in_forward"]:
            logger.info("  1. ❌ CRITICAL: LoRA not active in forward pass - fix PEFT setup")
        elif not summary["gradients_computed"]:
            logger.info("  1. ❌ CRITICAL: No gradients computed - check autograd setup")
        elif summary["bars_dot_zero"]:
            # Check specific causes from fix.txt
            optimizer_loaded = summary["optimizer_state_loaded"]
            
            if not optimizer_loaded:
                logger.info("  1. 🔧 Try disabling preconditioner (no optimizer state)")
            else:
                # Check preconditioner results
                precond_results = self.results["preconditioner_tests"]
                if precond_results.get("failed_preconditioning", 0) > 0:
                    logger.info("  1. 🔧 Fix Adam state mapping for LoRA parameters")
                else:
                    logger.info("  1. 🔧 Check parameter identity consistency (DDP vs PEFT)")
                    
            logger.info("  2. 🔧 Add buffer norm logging after accumulation")  
            logger.info("  3. 🔧 Verify LOO coefficient construction doesn't cancel")
        else:
            logger.info("  1. ✅ Run full entropy probe - gradient isolation successful!")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Gradient Isolation Test for RL Checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to RL checkpoint (e.g., localfs/training_state/step_40)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = GradientIsolationTest(args.checkpoint)
    tester.run_all_tests()


if __name__ == "__main__":
    main()