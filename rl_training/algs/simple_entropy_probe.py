# simple_entropy_probe.py - Lightweight entropy change estimation for regular training
import torch
import torch.distributed as dist
from typing import Optional, Dict, List, Any
import numpy as np
from dataclasses import dataclass


class SimpleEntropyProbe:
    """
    Lightweight entropy change estimation during regular RL training.
    
    Computes δH ≈ Σ_α (∂_α H) × (δθ_α) × P_α without storing Fisher kernel.
    This provides fast entropy monitoring with minimal computational overhead.
    
    Key features:
    - O(n_params) memory usage (not O(n_sequences²))
    - ~5% computational overhead
    - Real-time entropy change prediction
    - Configurable Adam preconditioning
    """
    
    def __init__(
        self,
        enabled: bool = True,
        debug: bool = False,
        preconditioning_mode: str = "previous_step",  # "none", "previous_step"
        log_every: int = 1,
    ):
        """
        Args:
            enabled: Whether the probe is active
            debug: Whether to print debug information
            preconditioning_mode: How to handle Adam conditioning factors
            log_every: Log metrics every N steps
        """
        self.enabled = enabled
        self.debug = debug
        self.preconditioning_mode = preconditioning_mode
        self.log_every = log_every
        
        # Metrics storage
        self.step_counter = 0
        self.last_delta_h = None
        self.last_entropy_gradient_norm = None
        self.last_param_update_norm = None
        
        # Running statistics
        self.delta_h_history = []
        self.entropy_history = []
        
    def compute_entropy_gradients_only(
        self,
        entropy_rollouts: Any,
        trainable_params: List[torch.nn.Parameter],
        policy_model: torch.nn.Module,
        cfg: dict,
        step_number: int,
        microbatch_size: int = 4
    ) -> torch.Tensor:
        """
        Compute only entropy gradients ∇H from independent entropy rollouts.
        Uses gradient accumulation with same microbatch size as training.
        Used in dual-buffer approach to avoid Fisher kernel correlation bias.
        
        Args:
            microbatch_size: Size of microbatches for gradient accumulation
        
        Returns:
            torch.Tensor: Flattened entropy gradients ∇H
        """
        device = next(policy_model.parameters()).device
        B, G, T_g = entropy_rollouts.gen_ids.shape
        
        # Initialize accumulated gradients
        accumulated_grads = None
        total_samples = 0
        
        # Clear gradients
        for param in trainable_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Process in microbatches like training
        for batch_start in range(0, B, microbatch_size):
            batch_end = min(batch_start + microbatch_size, B)
            mb_B = batch_end - batch_start
            
            # Extract microbatch
            mb_rollouts_data = type(entropy_rollouts)(
                prompt_ids=entropy_rollouts.prompt_ids[batch_start:batch_end],
                gen_ids=entropy_rollouts.gen_ids[batch_start:batch_end],
                reward=entropy_rollouts.reward[batch_start:batch_end],
                logprobs=entropy_rollouts.logprobs[batch_start:batch_end],
                tag_correct=entropy_rollouts.tag_correct[batch_start:batch_end],
                think_len=entropy_rollouts.think_len[batch_start:batch_end]
            )
            
            # Build sequences for microbatch
            seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences_from_rollouts(
                mb_rollouts_data, cfg, device
            )
            
            # Forward pass through policy model (in training mode for gradients)
            policy_model.train()
            outputs = policy_model(input_ids=seq_flat, attention_mask=attn_mask)
            logits = outputs.logits[:, :-1, :]  # Remove last position
            
            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_log_probs = torch.gather(log_probs, dim=-1, index=targets_tok.unsqueeze(-1)).squeeze(-1)
            
            # Apply generation mask to sum only over generated tokens
            gen_mask_flat = gen_mask.reshape(mb_B * G, T_g)
            masked_log_probs = target_log_probs * gen_mask_flat
            sequence_log_probs = masked_log_probs.sum(dim=-1)  # (mb_B*G,) - sum per sequence
            
            # Reshape back to (mb_B, G) for entropy gradient computation
            sequence_log_probs = sequence_log_probs.reshape(mb_B, G)
            
            # Compute entropy gradients for this microbatch
            pad_id = cfg.get("pad_token_id", 0)
            mb_entropy_grads = self._compute_entropy_gradients(
                mb_rollouts_data, sequence_log_probs, trainable_params, pad_id
            )
            
            # Accumulate gradients (weighted by microbatch size)
            weight = mb_B / B  # Weight by proportion of total batch
            if accumulated_grads is None:
                accumulated_grads = mb_entropy_grads * weight
            else:
                accumulated_grads += mb_entropy_grads * weight
            
            total_samples += mb_B
            
            if self.debug:
                mb_grad_norm = torch.norm(mb_entropy_grads).item()
                print(f"[ENTROPY] Step {step_number}, microbatch {batch_start//microbatch_size + 1}: grad norm = {mb_grad_norm:.6f}")
        
        if self.debug:
            total_grad_norm = torch.norm(accumulated_grads).item()
            print(f"[ENTROPY] Step {step_number}: total accumulated gradient norm = {total_grad_norm:.6f}")
        
        return accumulated_grads

    def compute_entropy_dual_buffer(
        self,
        entropy_rollouts: Any,          # Independent rollouts for entropy gradient estimation
        training_rollouts: Any,         # Independent rollouts for training (to compute δθ)
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_number: int,
        policy_model: torch.nn.Module,
        cfg: dict
    ) -> Dict[str, float]:
        """
        Dual-buffer entropy probe: Use independent rollout buffers for unbiased δH estimation.
        
        This approach eliminates bias from using the same samples for both ∇H estimation
        and δθ computation by using two independent rollout buffers.
        
        Args:
            entropy_rollouts: Independent rollouts for computing ∇H = E[(S-S̄) ∇S]
            training_rollouts: Independent rollouts for training/computing δθ
            trainable_params: Trainable parameters
            optimizer: Adam optimizer with previous step's states
            learning_rate: Current learning rate
            step_number: Current step number (skip if step 1)
            policy_model: Policy model for forward passes
            cfg: Configuration dict for model settings
            
        Returns:
            Dictionary with entropy change prediction and diagnostic metrics
        """
        if not self.enabled:
            return {}
            
        # Skip first step (no previous Adam states)
        if step_number <= 1:
            if self.debug:
                print(f"[SimpleEntropyProbe] Skipping first step (no previous Adam states)")
            return {
                "simple_entropy_delta_h_predicted": 0.0,
                "simple_entropy_grad_norm": 0.0,
                "simple_entropy_param_norm": 0.0,
                "simple_entropy_step_count": self.step_counter,
            }
            
        B, G, T_g = token_log_probs.shape
        device = token_log_probs.device
        
        if self.debug:
            print(f"[SimpleEntropyProbe] Sequential entropy computation (step {step_number}):")
        
        try:
            # STEP 1: Compute ∇H using ENTROPY BUFFER (independent samples)
            optimizer.zero_grad()
            
            # Forward pass on entropy buffer
            entropy_seq_flat, entropy_attn_mask, entropy_targets_tok, entropy_gen_mask = self._build_sequences_from_rollouts(
                entropy_rollouts, cfg, policy_model.device
            )
            B_ent, G_ent, T_g_ent = entropy_rollouts.gen_ids.shape
            
            # Forward pass for entropy computation
            with torch.cuda.amp.autocast(enabled=cfg.get("bf16", True), dtype=torch.bfloat16):
                entropy_logits = policy_model(entropy_seq_flat, attention_mask=entropy_attn_mask).logits
            entropy_logits = entropy_logits / cfg.get("temperature", 1.0)
            entropy_logp_all = torch.nn.functional.log_softmax(entropy_logits.float(), dim=-1)
            entropy_new_logp = entropy_logp_all[:, :-1].gather(-1, entropy_targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g_ent:]
            entropy_new_logp = torch.nan_to_num(entropy_new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
            entropy_new_logp = entropy_new_logp.view(B_ent, G_ent, T_g_ent)
            
            # Compute sequence log probabilities for entropy buffer
            entropy_seq_log_probs = (entropy_new_logp * entropy_gen_mask).sum(dim=-1)  # (B, G) with gradients
            
            # Compute mean log probability for baseline (distributed)
            local_mean = entropy_seq_log_probs.mean()
            if dist.is_initialized():
                global_mean = local_mean.clone()
                dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
                mean_log_prob = global_mean
            else:
                mean_log_prob = local_mean
            
            # Centered log probabilities: S(t) - S̄_global (detached to remove unwanted gradients)
            centered_log_probs = (entropy_seq_log_probs - mean_log_prob).detach()  # (B, G)
            
            # Create entropy loss: sum((S-S̄) * S) where only S has gradients
            entropy_loss = torch.sum(centered_log_probs * entropy_seq_log_probs)
            
            # Compute entropy gradients using .backward()
            entropy_loss.backward()
            
            # Extract entropy gradients ∇H
            entropy_grad_chunks = []
            if dist.is_initialized():
                world_size = dist.get_world_size()
                global_batch_size_ent = B_ent * G_ent * world_size
            else:
                global_batch_size_ent = B_ent * G_ent
            
            entropy_non_zero_grads = 0
            for param in trainable_params:
                if param.grad is not None:
                    # Entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N_global) Σ (S-S̄) ∂_α S
                    entropy_grad = -param.grad.detach().flatten() / global_batch_size_ent
                    entropy_grad_chunks.append(entropy_grad)
                    if torch.any(param.grad != 0):
                        entropy_non_zero_grads += 1
                else:
                    entropy_grad_chunks.append(torch.zeros(param.numel(), device=policy_model.device))
            
            entropy_grads = torch.cat(entropy_grad_chunks)
            
            # STEP 2: Compute δθ using TRAINING BUFFER (independent samples)  
            optimizer.zero_grad()
            
            # Forward pass on training buffer
            train_seq_flat, train_attn_mask, train_targets_tok, train_gen_mask = self._build_sequences_from_rollouts(
                training_rollouts, cfg, policy_model.device
            )
            B_train, G_train, T_g_train = training_rollouts.gen_ids.shape
            
            # Forward pass for training computation
            with torch.cuda.amp.autocast(enabled=cfg.get("bf16", True), dtype=torch.bfloat16):
                train_logits = policy_model(train_seq_flat, attention_mask=train_attn_mask).logits
            train_logits = train_logits / cfg.get("temperature", 1.0)
            train_logp_all = torch.nn.functional.log_softmax(train_logits.float(), dim=-1)
            train_new_logp = train_logp_all[:, :-1].gather(-1, train_targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g_train:]
            train_new_logp = torch.nan_to_num(train_new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
            train_new_logp = train_new_logp.view(B_train, G_train, T_g_train)
            
            # Compute a dummy training loss (we just need gradients for δθ estimation)
            train_seq_log_probs = (train_new_logp * train_gen_mask).sum(dim=-1)  # (B, G)
            train_loss = -train_seq_log_probs.mean()  # Simple loss for gradient computation
            
            # Compute training gradients
            train_loss.backward()
            
            # Extract parameter updates δθ
            param_update_chunks = []
            train_non_zero_grads = 0
            for param in trainable_params:
                if param.grad is not None:
                    # Parameter update from current gradients (use current learning rate)
                    param_update = -learning_rate * param.grad.detach().flatten()
                    param_update_chunks.append(param_update)
                    if torch.any(param.grad != 0):
                        train_non_zero_grads += 1
                else:
                    param_update_chunks.append(torch.zeros(param.numel(), device=policy_model.device))
            
            param_updates = torch.cat(param_update_chunks)
            
            # STEP 3: Apply Adam preconditioning using PREVIOUS step's states
            conditioned_updates = self._apply_adam_preconditioning_sequential(
                param_updates, optimizer, trainable_params
            )
            
            # 9. Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(entropy_grads * conditioned_updates).item()
            
            # 10. All-reduce delta_h across ranks (same as training metrics)
            if dist.is_initialized():
                delta_h_tensor = torch.tensor(delta_h, device=device)
                dist.all_reduce(delta_h_tensor, op=dist.ReduceOp.AVG)
                delta_h = delta_h_tensor.item()
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(entropy_grads).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            self.step_counter += 1
            
            if self.debug:
                print(f"  Entropy buffer size: ({B_ent}, {G_ent}, {T_g_ent})")
                print(f"  Training buffer size: ({B_train}, {G_train}, {T_g_train})")
                print(f"  Entropy loss: {entropy_loss.item():.6f}")
                print(f"  Training loss: {train_loss.item():.6f}")
                print(f"  Entropy non-zero gradients: {entropy_non_zero_grads}/{len(trainable_params)}")
                print(f"  Training non-zero gradients: {train_non_zero_grads}/{len(trainable_params)}")
                print(f"  Entropy grad norm: {torch.norm(entropy_grads).item():.6f}")
                print(f"  Param update norm: {torch.norm(conditioned_updates).item():.6f}")
                print(f"  Predicted δH (dual-buffer): {delta_h:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in sequential computation: {e}")
            return {}
        
        finally:
            # Always clear gradients after entropy probe to not interfere with training
            optimizer.zero_grad()
    
    def _apply_adam_preconditioning_sequential(
        self,
        param_updates: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        trainable_params: List[torch.nn.Parameter]
    ) -> torch.Tensor:
        """Apply Adam preconditioning factors P_α to parameter updates using previous step's states."""
        
        if self.preconditioning_mode == "none":
            return param_updates
        
        if self.preconditioning_mode == "previous_step":
            # Use Adam conditioning from previous step
            conditioning_chunks = []
            param_idx = 0
            
            for param in trainable_params:
                param_size = param.numel()
                
                if param in optimizer.state and 'exp_avg_sq' in optimizer.state[param]:
                    # Adam conditioning: 1 / (sqrt(v) + eps)
                    state = optimizer.state[param]
                    v = state['exp_avg_sq']
                    eps = optimizer.param_groups[0].get('eps', 1e-8)
                    conditioning = 1.0 / (torch.sqrt(v) + eps)
                    conditioning_chunks.append(conditioning.flatten())
                else:
                    # No conditioning available (shouldn't happen after step 1)
                    conditioning_chunks.append(torch.ones(param_size, device=param.device))
                
                param_idx += param_size
            
            conditioning_factors = torch.cat(conditioning_chunks)
            return param_updates * conditioning_factors
        
        return param_updates
    
    def _build_sequences_from_rollouts(self, rollouts: Any, cfg: dict, device: torch.device):
        """Build sequences from rollouts (same logic as dr_grpo._build_sequences)."""
        import torch
        
        # Extract pad_id from cfg or use default
        pad_id = cfg.get("pad_token_id", 0)
        
        B, G, T_g = rollouts.gen_ids.shape
        prompt_rep = rollouts.prompt_ids.unsqueeze(1).expand(-1, G, -1)
        seq_ids = torch.cat((prompt_rep, rollouts.gen_ids), dim=-1)  # (B, G, T_total)

        seq_flat = seq_ids.reshape(B * G, -1)
        attn_mask = (seq_flat != pad_id).long()
        targets_tok = seq_flat[:, 1:]  # teacher forcing targets
        gen_mask = (rollouts.gen_ids != pad_id).float()  # (B, G, T_g)
        
        return seq_flat, attn_mask, targets_tok, gen_mask
    
    def compute_entropy_sequential(
        self,
        token_log_probs: torch.Tensor,  # (B, G, T_g) token-level log probabilities with gradients
        gen_mask: torch.Tensor,         # (B, G, T_g) generation mask
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_number: int
    ) -> Dict[str, float]:
        """
        Original sequential entropy probe (fallback method).
        """
        if not self.enabled:
            return {}
            
        # Skip first step (no previous Adam states)
        if step_number <= 1:
            if self.debug:
                print(f"[SimpleEntropyProbe] Skipping first step (no previous Adam states)")
            return {
                "simple_entropy_delta_h_predicted": 0.0,
                "simple_entropy_grad_norm": 0.0,
                "simple_entropy_param_norm": 0.0,
                "simple_entropy_step_count": self.step_counter,
            }
            
        B, G, T_g = token_log_probs.shape
        device = token_log_probs.device
        
        if self.debug:
            print(f"[SimpleEntropyProbe] Sequential entropy computation (step {step_number}):")
        
        try:
            # Clear any existing gradients
            optimizer.zero_grad()
            
            # Compute sequence log probabilities from token-level (same as optimizer path)
            seq_log_probs = (token_log_probs * gen_mask).sum(dim=-1)  # (B, G) with gradients
            
            # Compute mean log probability for baseline (distributed)
            local_mean = seq_log_probs.mean()
            if dist.is_initialized():
                # Average across all GPUs for global baseline
                global_mean = local_mean.clone()
                dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
                mean_log_prob = global_mean
            else:
                mean_log_prob = local_mean
            
            # Centered log probabilities: S(t) - S̄_global (detached to remove unwanted gradients)
            centered_log_probs = (seq_log_probs - mean_log_prob).detach()  # (B, G)
            
            # Create entropy loss: sum((S-S̄) * S) where only S has gradients
            entropy_loss = torch.sum(centered_log_probs * seq_log_probs)
            
            # Compute entropy gradients using .backward() (we know this works!)
            entropy_loss.backward()
            
            # Extract and process entropy gradients
            entropy_grad_chunks = []
            param_update_chunks = []
            
            # Get global batch size for proper normalization
            if dist.is_initialized():
                world_size = dist.get_world_size()
                global_batch_size = B * G * world_size
            else:
                global_batch_size = B * G
            
            non_zero_grads = 0
            none_grads = 0
            
            for param in trainable_params:
                if param.grad is not None:
                    # Entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N_global) Σ (S-S̄) ∂_α S
                    entropy_grad = -param.grad.detach().flatten() / global_batch_size
                    entropy_grad_chunks.append(entropy_grad)
                    
                    # Parameter update from previous step (use current learning rate)
                    param_update = -learning_rate * param.grad.detach().flatten()
                    param_update_chunks.append(param_update)
                    
                    if torch.any(param.grad != 0):
                        non_zero_grads += 1
                else:
                    none_grads += 1
                    entropy_grad_chunks.append(torch.zeros(param.numel(), device=device))
                    param_update_chunks.append(torch.zeros(param.numel(), device=device))
            
            entropy_grads = torch.cat(entropy_grad_chunks)
            param_updates = torch.cat(param_update_chunks)
            
            # Apply Adam preconditioning using PREVIOUS step's states
            conditioned_updates = self._apply_adam_preconditioning_sequential(
                param_updates, optimizer, trainable_params
            )
            
            # Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(entropy_grads * conditioned_updates).item()
            
            # All-reduce delta_h across ranks (same as training metrics)
            if dist.is_initialized():
                delta_h_tensor = torch.tensor(delta_h, device=device)
                dist.all_reduce(delta_h_tensor, op=dist.ReduceOp.AVG)
                delta_h = delta_h_tensor.item()
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(entropy_grads).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            self.step_counter += 1
            
            if self.debug:
                print(f"  Entropy loss: {entropy_loss.item():.6f}")
                print(f"  Non-zero gradients: {non_zero_grads}/{len(trainable_params)}")
                print(f"  None gradients: {none_grads}/{len(trainable_params)}")
                print(f"  Entropy grad norm: {torch.norm(entropy_grads).item():.6f}")
                print(f"  Param update norm: {torch.norm(conditioned_updates).item():.6f}")
                print(f"  Predicted δH: {delta_h:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in sequential computation: {e}")
            return {}
        
        finally:
            # Always clear gradients after entropy probe to not interfere with training
            optimizer.zero_grad()
    
    def complete_delta_h_calculation(
        self,
        accumulated_entropy_grads: torch.Tensor,  # Accumulated entropy gradients across microbatches
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float
    ) -> Dict[str, float]:
        """
        Complete the delta H calculation using accumulated entropy gradients and current Adam states.
        
        Args:
            accumulated_entropy_grads: Entropy gradients accumulated across all microbatches
            trainable_params: Trainable parameters
            optimizer: Current optimizer with up-to-date Adam states
            learning_rate: Current learning rate
            
        Returns:
            Dictionary with entropy change prediction and diagnostic metrics
        """
        if not self.enabled:
            return {}
            
        try:
            # All-reduce accumulated entropy gradients across ranks (like training gradients)
            if dist.is_initialized():
                dist.all_reduce(accumulated_entropy_grads, op=dist.ReduceOp.SUM)
            
            # Extract parameter updates: δθ_α (what optimizer just applied)
            param_updates = self._extract_parameter_updates(trainable_params, learning_rate)
            
            # Apply Adam preconditioning: P_α (using current Adam states)
            conditioned_updates = self._apply_preconditioning(param_updates, optimizer, trainable_params)
            
            # Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(accumulated_entropy_grads * conditioned_updates).item()
            
            if self.debug:
                print(f"[SimpleEntropyProbe] Final calculation:")
                print(f"  Accumulated entropy grad norm: {torch.norm(accumulated_entropy_grads).item():.6f}")
                print(f"  Param updates norm: {torch.norm(param_updates).item():.6f}")
                print(f"  Conditioned updates norm: {torch.norm(conditioned_updates).item():.6f}")
                print(f"  Delta H: {delta_h:.6f}")
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(accumulated_entropy_grads).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            self.step_counter += 1
            
            # Debug output
            if self.debug and (self.step_counter % self.log_every == 0):
                print(f"[SimpleEntropyProbe] Step {self.step_counter}:")
                print(f"  Predicted δH: {delta_h:.6f}")
                print(f"  Entropy grad norm: {self.last_entropy_gradient_norm:.6f}")
                print(f"  Param update norm: {self.last_param_update_norm:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in delta H calculation: {e}")
            return {}

    def compute_delta_h_prediction(
        self,
        rollouts: Any,  # RolloutBatch
        advantages: torch.Tensor,  # (B, G)
        log_probs: torch.Tensor,   # (B, G) sequence-level log probabilities
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        pad_id: int = 0
    ) -> Dict[str, float]:
        """
        Predict entropy change using simple δH = Σ_α (∂_α H) × (δθ_α) × P_α formula.
        
        Returns:
            Dictionary with entropy change prediction and diagnostic metrics
        """
        if not self.enabled:
            return {}
            
        try:
            # 1. Compute entropy gradients: ∂_α H = -E[(S-S̄) ∂_α S]
            entropy_gradients = self._compute_entropy_gradients(
                rollouts, log_probs, trainable_params, pad_id
            )
            
            # 2. Get parameter updates: δθ_α (what optimizer will apply)
            param_updates = self._extract_parameter_updates(trainable_params, learning_rate)
            
            # 3. Apply Adam preconditioning: P_α
            conditioned_updates = self._apply_preconditioning(param_updates, optimizer, trainable_params)
            
            # 4. Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(entropy_gradients * conditioned_updates).item()
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(entropy_gradients).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            # Compute current entropy for reference
            current_entropy = self._compute_current_entropy(log_probs, pad_id)
            self.entropy_history.append(current_entropy)
            if len(self.entropy_history) > 100:
                self.entropy_history.pop(0)
            
            self.step_counter += 1
            
            # Debug output
            if self.debug and (self.step_counter % self.log_every == 0):
                print(f"[SimpleEntropyProbe] Step {self.step_counter}:")
                print(f"  Predicted δH: {delta_h:.6f}")
                print(f"  Current entropy: {current_entropy:.6f}")
                print(f"  Entropy grad norm: {self.last_entropy_gradient_norm:.6f}")
                print(f"  Param update norm: {self.last_param_update_norm:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_current": current_entropy,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in step {self.step_counter}: {e}")
            return {}
    
    def _compute_entropy_gradients(
        self,
        rollouts: Any,
        log_probs: torch.Tensor,  # (B, G)
        trainable_params: List[torch.nn.Parameter],
        pad_id: int
    ) -> torch.Tensor:
        """
        Compute aggregated entropy gradients: ∂_α H = -E[(S-S̄) ∂_α S].
        
        This is much cheaper than computing per-sequence gradients.
        """
        B, G = log_probs.shape
        device = log_probs.device
        
        # Compute mean log probability for baseline (distributed)
        local_mean = log_probs.mean()
        if dist.is_initialized():
            # Average across all GPUs for global baseline
            global_mean = local_mean.clone()
            dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
            mean_log_prob = global_mean
        else:
            mean_log_prob = local_mean
        
        # Centered log probabilities: S(t) - S̄_global
        centered_log_probs = log_probs - mean_log_prob  # (B, G)
        
        # We need to compute: E[(S-S̄) ∂_α S] = E[(S-S̄) ∂_α log π(t)]
        # This requires computing gradients of log probabilities w.r.t. parameters
        
        # Clear existing gradients
        for param in trainable_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute weighted sum of log probabilities for backprop
        # This gives us ∂_α Σ_t (S(t) - S̄) S(t) = Σ_t (S(t) - S̄) ∂_α S(t)
        weighted_log_prob_sum = torch.sum(centered_log_probs * log_probs)
        
        # Backward pass to get gradients
        weighted_log_prob_sum.backward(retain_graph=True)
        
        # Collect gradients and normalize by global batch size
        entropy_grad_chunks = []
        
        # Get global batch size for proper normalization
        if dist.is_initialized():
            world_size = dist.get_world_size()
            global_batch_size = B * G * world_size
        else:
            global_batch_size = B * G
            
        for param in trainable_params:
            if param.grad is not None:
                # Entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N_global) Σ (S-S̄) ∂_α S
                entropy_grad = -param.grad.detach().flatten() / global_batch_size
                entropy_grad_chunks.append(entropy_grad)
            else:
                entropy_grad_chunks.append(torch.zeros(param.numel(), device=device))
        
        local_entropy_grad = torch.cat(entropy_grad_chunks)
        
        # Sum gradients across GPUs (they're already normalized by global batch size)
        if dist.is_initialized():
            global_entropy_grad = local_entropy_grad.clone()
            dist.all_reduce(global_entropy_grad, op=dist.ReduceOp.SUM)
            return global_entropy_grad
        else:
            return local_entropy_grad
    
    def _extract_parameter_updates(
        self,
        trainable_params: List[torch.nn.Parameter],
        learning_rate: float
    ) -> torch.Tensor:
        """Extract parameter updates δθ_α that optimizer will apply."""
        param_update_chunks = []
        
        for param in trainable_params:
            if param.grad is not None:
                # Basic SGD update: δθ = -η * ∇L
                param_update = -learning_rate * param.grad.detach().flatten()
                param_update_chunks.append(param_update)
            else:
                param_update_chunks.append(torch.zeros(param.numel(), device=param.device))
        
        return torch.cat(param_update_chunks)
    
    def _apply_preconditioning(
        self,
        param_updates: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        trainable_params: List[torch.nn.Parameter]
    ) -> torch.Tensor:
        """Apply Adam preconditioning factors P_α to parameter updates."""
        
        if self.preconditioning_mode == "none":
            return param_updates
        
        if self.preconditioning_mode == "previous_step":
            # Use Adam conditioning from previous step
            conditioning_chunks = []
            param_idx = 0
            
            for param in trainable_params:
                param_size = param.numel()
                
                if param in optimizer.state and 'exp_avg_sq' in optimizer.state[param]:
                    # Adam conditioning: 1 / (sqrt(v) + eps)
                    state = optimizer.state[param]
                    v = state['exp_avg_sq']
                    eps = optimizer.param_groups[0].get('eps', 1e-8)
                    conditioning = 1.0 / (torch.sqrt(v) + eps)
                    conditioning_chunks.append(conditioning.flatten())
                else:
                    # No conditioning available (early in training)
                    conditioning_chunks.append(torch.ones(param_size, device=param.device))
                
                param_idx += param_size
            
            conditioning_factors = torch.cat(conditioning_chunks)
            return param_updates * conditioning_factors
        
        return param_updates
    
    def _compute_current_entropy(self, log_probs: torch.Tensor, pad_id: int) -> float:
        """Compute current entropy H = -E[log π(t)] for reference."""
        return -log_probs.mean().item()
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics for logging."""
        metrics = {}
        
        if self.last_delta_h is not None:
            metrics.update({
                "simple_entropy_delta_h": self.last_delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_steps": self.step_counter,
            })
        
        # Add rolling statistics
        if len(self.delta_h_history) > 1:
            metrics.update({
                "simple_entropy_delta_h_mean": np.mean(self.delta_h_history[-10:]),  # Last 10 steps
                "simple_entropy_delta_h_std": np.std(self.delta_h_history[-10:]),
            })
        
        if len(self.entropy_history) > 1:
            metrics.update({
                "simple_entropy_current_mean": np.mean(self.entropy_history[-10:]),
                "simple_entropy_trend": self.entropy_history[-1] - self.entropy_history[0] if len(self.entropy_history) >= 2 else 0.0,
            })
        
        return metrics
    
    def reset(self) -> None:
        """Reset probe state."""
        self.last_delta_h = None
        self.last_entropy_gradient_norm = None
        self.last_param_update_norm = None
        
    def save_history(self, filepath: str) -> None:
        """Save entropy history for analysis."""
        if not self.enabled:
            return
            
        data = {
            "delta_h_history": self.delta_h_history,
            "entropy_history": self.entropy_history,
            "step_counter": self.step_counter,
            "config": {
                "preconditioning_mode": self.preconditioning_mode,
                "log_every": self.log_every,
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.debug:
            print(f"[SimpleEntropyProbe] Saved history to {filepath}")