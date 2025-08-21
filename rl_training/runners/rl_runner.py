# rl_training/runners/rl_runner.py
from __future__ import annotations

import os, json, math, gc, datetime, pathlib, shutil, yaml, torch, copy
import torch.distributed as dist
from collections import defaultdict
from subprocess import run as run_sync
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

from rl_training.runners.collect_rollouts import RolloutCollector
from rl_training.algs.dr_grpo import DRGRPO
from rl_training.algs.base import RolloutBatch
from rl_training.runners.eval_callback import EvalCallback

LOCALFS_ROOT = pathlib.Path(os.environ.get("LOCALFS_ROOT", "/lambda/nfs/localfs"))
TRAINING_ROOT = LOCALFS_ROOT / "training_runs"
EVAL_ROOT = LOCALFS_ROOT / "eval_runs"


def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(prompt_ids=rb.prompt_ids[sl],
                        gen_ids=rb.gen_ids[sl],
                        reward=rb.reward[sl],
                        logprobs=rb.logprobs[sl])

def _unwrap(model):
    # Changed 8/11: helper for DDP/unwrapped access
    return model.module if hasattr(model, "module") else model  # Changed 8/11


class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str, 
                 resume_optimizer_path: str = None,
                 resume_scheduler_path: str = None,
                 resume_info: dict = None):
        # Store original checkpoint path for reference model (KL computation)
        self.original_lora_ckpt = lora_ckpt
        # ─── distributed init ────────────────────────────────────────────
        # Changed 8/11: make DDP optional (works in single-GPU Colab)
        if os.environ.get("WORLD_SIZE", "1") != "1":
            dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"))  # Changed 8/11
            self.local_rank = int(os.getenv("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.rank = dist.get_rank()
            self.ddp = True  # Changed 8/11
        else:
            self.local_rank = 0
            self.rank = 0
            self.ddp = False  # Changed 8/11

        # ─── I/O setup ───────────────────────────────────────────────────
        self.cfg = yaml.safe_load(open(cfg_path))
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = TRAINING_ROOT / f"run_{stamp}"
        
        # Create directory structure
        if self.rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "logs").mkdir(exist_ok=True)
            (self.dir / "tensorboard").mkdir(exist_ok=True)
            (self.dir / "training_state").mkdir(exist_ok=True)
            EVAL_ROOT.mkdir(parents=True, exist_ok=True)
            shutil.copy(cfg_path, self.dir / "config.yaml")

        # Wait for rank 0 to create directories
        if hasattr(dist, 'is_initialized') and dist.is_initialized():
            dist.barrier()

        self.tb = SummaryWriter(str(self.dir / "tensorboard")) if self.rank == 0 else None
        self.save_every = self.cfg.get("save_every", 100)

        # -----GNS----------------------------------------------------------
        self.gns_cfg = self.cfg.get("gns_probe", {})
        self._gns_state = {
            "ema_y_small": None, "ema_y_large": None,
            "B_small": int(self.gns_cfg.get("small_B", 8)),
            "B_large": int(self.gns_cfg.get("large_B", 64)),
            "ema": float(self.gns_cfg.get("ema", 0.9)),
        }


        # ─── model & tokenizer ───────────────────────────────────────────
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(self.cfg["backbone"],
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb)
        base = prepare_model_for_kbit_training(base)  # PEFT/QLoRA prep  # (PEFT docs)  # Changed 8/11
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        model = PeftModel.from_pretrained(base, lora_ckpt, is_trainable=True)
        model.enable_input_require_grads()
        model = model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        if self.ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )  # Changed 8/11
        self.model = model

        self.tok = AutoTokenizer.from_pretrained(self.cfg["backbone"])
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if _unwrap(self.model).config.pad_token_id is None:
            _unwrap(self.model).config.pad_token_id = self.tok.pad_token_id
        self.pad_id = self.tok.pad_token_id
        self.tok.padding_side = "left"

        # Create reference model from original checkpoint (not current model state)
        self.ref_model = self._create_reference_model()
        self.ref_model = self.ref_model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # Create rollout collector (VLLM or standard)
        use_vllm = self.cfg.get("use_vllm", False)
        
        if use_vllm:
            # Import VLLM components only when needed to avoid CUDA initialization
            from rl_training.runners.collect_rollouts_vllm import create_rollout_collector
            vllm_kwargs = {
                "vllm_reload_every": self.cfg.get("vllm_reload_every", 10),
                "vllm_tensor_parallel": self.cfg.get("vllm_tensor_parallel", 1),
                "vllm_gpu_memory_utilization": self.cfg.get("vllm_gpu_memory_utilization", 0.4),
                "device": f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
            }
            self.collector = create_rollout_collector(
                self.model, self.tok, self.cfg,
                out_dir=self.dir / "logs",
                use_vllm=True,
                **vllm_kwargs
            )
        else:
            # Use standard rollout collector (no VLLM imports)
            self.collector = RolloutCollector(
                self.model, self.tok, self.cfg,
                out_dir=self.dir / "logs",
                device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
            )
        # Calculate grad_accum_steps automatically based on distributed setup
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.buffer_size = self.cfg["buffer_size"]
        microbatch_size = self.cfg["microbatch_size"]
        auto_grad_accum_steps = self.buffer_size // (world_size * microbatch_size)
        
        if self.rank == 0:
            print(f"[AUTO CONFIG] buffer_size={self.buffer_size}, world_size={world_size}, microbatch_size={microbatch_size}")
            config_grad_accum = self.cfg.get('grad_accum_steps', 'not specified')
            print(f"[AUTO CONFIG] Calculated grad_accum_steps={auto_grad_accum_steps} (config had {config_grad_accum})")
        
        self.algo = DRGRPO(self.model, self.cfg, pad_id=self.pad_id,
                           # ratio_log_path=self.dir / "logs" / "ratios.jsonl",  # Disabled for production
                           ratio_log_path=None,
                           grad_accum_steps=auto_grad_accum_steps)
        
        self.accum = auto_grad_accum_steps
        self.eval_cb = EvalCallback(self.dir, self.cfg)
        
        # Store resume information and paths
        self.resume_info = resume_info
        self.resume_optimizer_path = resume_optimizer_path
        self.resume_scheduler_path = resume_scheduler_path
        
        # Set starting step based on resume info
        if resume_info:
            self.step_id = resume_info["step"]
            if self.rank == 0:
                print(f"🔄 Resuming from step {self.step_id}")
        else:
            self.step_id = 0
        
        # Load optimizer and scheduler states if resuming
        if resume_optimizer_path and resume_scheduler_path:
            self._load_training_states()
        elif resume_optimizer_path or resume_scheduler_path:
            print("⚠️ Warning: Only one of optimizer/scheduler paths provided, both needed for proper resumption")

    @property
    def optimizer(self):
        """Access to the optimizer from the algorithm."""
        return self.algo.opt
    
    @property
    def scheduler(self):
        """Access to the scheduler from the algorithm."""
        return self.algo.lr_sched
    
    def _load_training_states(self):
        """Load optimizer and scheduler states for resumption."""
        if self.rank == 0:
            print(f"📥 Loading training states...")
            print(f"   Optimizer: {self.resume_optimizer_path}")
            print(f"   Scheduler: {self.resume_scheduler_path}")
        
        try:
            # Load optimizer state
            optimizer_state = torch.load(self.resume_optimizer_path, map_location=f"cuda:{self.local_rank}")
            self.optimizer.load_state_dict(optimizer_state)
            
            # Load scheduler state if scheduler exists
            if self.scheduler is not None:
                scheduler_state = torch.load(self.resume_scheduler_path, map_location=f"cuda:{self.local_rank}")
                self.scheduler.load_state_dict(scheduler_state)
            else:
                if self.rank == 0:
                    print("⚠️ No scheduler to load (scheduler is None)")
            
            if self.rank == 0:
                print(f"✅ Training states loaded successfully")
                
        except Exception as e:
            if self.rank == 0:
                print(f"❌ Failed to load training states: {e}")
            raise

    # ─── training loop ────────────────────────────────────────────────────
    def train(self, total_updates: int):
        K = self.cfg["ppo_epochs"]
        ga_steps = self.accum
        B = self.cfg["microbatch_size"]
        outer_loops = math.ceil(total_updates / K)

        world = dist.get_world_size() if dist.is_initialized() else 1
        per_rank = self.buffer_size // world
        # Handle remainder: rank 0 gets extra samples if buffer_size not evenly divisible
        if self.rank == 0:
            per_rank += self.buffer_size % world
        
        if self.rank == 0:
            print(f"Distributed collection: {world} ranks, buffer_size={self.buffer_size}, per_rank={per_rank}")

        for _ in trange(outer_loops, desc="outer collect loops", disable=(self.rank != 0)):
            # Memory monitoring before collection
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory before collection: {mem_before:.2f}GB")
            
            # ═══ DUAL-BUFFER ENTROPY PROBE COLLECTION ═══
            entropy_grads = None
            if self.algo.simple_entropy_probe.enabled and self.step_id > 0:  # Skip first step
                print(f"[DEBUG] Rank {self.rank} collecting ENTROPY buffer (step {self.step_id + 1})")
                
                # 1. Collect entropy buffer (half size)
                entropy_per_rank = per_rank // 2
                entropy_rb = self.collector.collect_batch(batch_prompts=entropy_per_rank)
                
                if torch.cuda.is_available():
                    mem_after_entropy = torch.cuda.memory_allocated() / 1024**3
                    print(f"[MEMORY] Rank {self.rank} GPU memory after entropy collection: {mem_after_entropy:.2f}GB")
                
                # 2. Compute entropy gradients and cleanup immediately
                try:
                    current_lr = self.algo.opt.param_groups[0]['lr']
                    ddp_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    
                    # Convert RolloutBuffer to RolloutBatch for entropy probe
                    entropy_batch = entropy_rb.to_batch(device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
                    
                    # Compute ∇H using entropy buffer with gradient accumulation
                    entropy_grads = self.algo.simple_entropy_probe.compute_entropy_gradients_only(
                        entropy_rollouts=entropy_batch,
                        trainable_params=ddp_trainable_params,
                        policy_model=self.model,
                        cfg=self.cfg,
                        step_number=self.step_id + 1,
                        microbatch_size=B  # Use same microbatch size as training
                    )
                    
                    print(f"[DEBUG] Rank {self.rank} computed entropy gradients, norm: {torch.norm(entropy_grads).item():.6f}")
                    
                except Exception as e:
                    print(f"[ENTROPY] Error computing entropy gradients: {e}")
                    entropy_grads = None
                
                # 3. Immediate cleanup of entropy buffer
                del entropy_rb
                torch.cuda.empty_cache()
                
                if torch.cuda.is_available():
                    mem_after_cleanup = torch.cuda.memory_allocated() / 1024**3
                    print(f"[MEMORY] Rank {self.rank} GPU memory after entropy cleanup: {mem_after_cleanup:.2f}GB")
            
            # 4. Collect training buffer (remaining half or full if no entropy probe)
            training_per_rank = per_rank // 2 if (self.algo.simple_entropy_probe.enabled and self.step_id > 0) else per_rank
            print(f"[DEBUG] Rank {self.rank} collecting TRAINING buffer")
            rb = self.collector.collect_batch(batch_prompts=training_per_rank)
            
            # Memory monitoring after collection
            if torch.cuda.is_available():
                mem_after_collect = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory after training collection: {mem_after_collect:.2f}GB")
            
            # Add barrier after collection to ensure both ranks finish before training
            if self.ddp:
                print(f"[DEBUG] Rank {self.rank} entering post-collection barrier")
                dist.barrier()
                print(f"[DEBUG] Rank {self.rank} exited post-collection barrier")
            print(f"[DEBUG] Rank {self.rank} completed rollout collection")
            print(f"[DEBUG] Rank {self.rank} about to start training on buffer")
            
            # each rank trains on its shard; DDP averages grads for you
            self._train_one_buffer(rb, K, ga_steps, B, entropy_grads)
            
            # ═══ SYNCHRONIZED PROBE PROCESSING (Option 2: Buffer Data Preservation) ═══
            # Rank 0 determines if GNS probe needs processing and broadcasts to all ranks
            has_gns_probe = False
            if self.rank == 0:
                has_gns_probe = hasattr(self, '_pending_gns_data') and self._pending_gns_data is not None
            
            # Broadcast probe status from rank 0 to all ranks in distributed mode
            if self.ddp:
                probe_tensor = torch.tensor([1 if has_gns_probe else 0], device=f"cuda:{self.local_rank}")
                dist.broadcast(probe_tensor, src=0)
                has_gns_probe = bool(probe_tensor.item())
            
            # Now all ranks know if GNS probe processing is needed
            if has_gns_probe:
                if self.rank == 0:
                    print(f"[DEBUG] Rank {self.rank} processing saved GNS probe data (step_id={self._pending_gns_data['step_id']})")
                    try:
                        # Wrap GNS probe in no_sync to prevent DDP hanging during gradient computation
                        from contextlib import nullcontext
                        ctx = self.model.no_sync() if hasattr(self.model, "no_sync") else nullcontext()
                        with ctx:
                            self._probe_gns_from_saved_data(self._pending_gns_data)
                        print(f"[DEBUG] Rank {self.rank} COMPLETED delayed GNS probe (step_id={self._pending_gns_data['step_id']})")
                    except Exception as e:
                        print(f"[GNS] delayed probe failed: {e}")
                    finally:
                        self._pending_gns_data = None  # Clear saved data
                else:
                    print(f"[DEBUG] Rank {self.rank} waiting for rank 0 to complete GNS probe")
                    if hasattr(self, '_pending_gns_data'):
                        self._pending_gns_data = None  # Clear any saved data on non-rank-0
                
                # Barrier: All ranks wait for rank 0 to finish GNS probe before proceeding
                if self.ddp:
                    print(f"[DEBUG] Rank {self.rank} entering post-GNS-probe barrier")
                    dist.barrier()
                    print(f"[DEBUG] Rank {self.rank} exited post-GNS-probe barrier")
            
            # Memory monitoring after cleanup (rb is cleaned up inside _train_one_buffer)
            if torch.cuda.is_available():
                mem_after_cleanup = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory after cleanup: {mem_after_cleanup:.2f}GB")
            
            if self.step_id % self.save_every == 0:
                self._save_ckpt()
        self._save_ckpt(final=True)

    def _train_one_buffer(self, rb, K, ga_steps, B, entropy_grads=None):
        print(f"[DEBUG] Rank {self.rank} entered _train_one_buffer with {len(rb)} prompts, K={K}, ga_steps={ga_steps}, B={B}")
        stats_sum, total_mb_cnt = defaultdict(float), 0
        for epoch in range(K):
            print(f"[DEBUG] Rank {self.rank} starting PPO epoch {epoch+1}/{K}")
            micro_cnt = 0
            for idx in rb.iter_minibatches(B, shuffle=True):
                print(f"[DEBUG] Rank {self.rank} processing microbatch {micro_cnt+1}, idx={len(idx)} samples")
                sync = ((micro_cnt + 1) % ga_steps == 0)
                print(f"[DEBUG] Rank {self.rank} microbatch {micro_cnt+1}: sync_grads={sync} (step_id will be {self.step_id + (1 if sync else 0)})")
                mb = rb.get_batch(idx, device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
                print(f"[DEBUG] Rank {self.rank} calling algo.step for microbatch {micro_cnt+1}")
                # Only call entropy probe on the final microbatch (when sync_grads=True)
                stats = self.algo.step(mb, self.ref_model, sync_grads=sync, call_entropy_probe=sync, entropy_grads=entropy_grads if sync else None)
                print(f"[DEBUG] Rank {self.rank} completed algo.step for microbatch {micro_cnt+1}")
                for k, v in stats.items():
                    stats_sum[k] += v
                micro_cnt += 1
                total_mb_cnt += 1
                if sync:
                    self.step_id += 1
                    print(f"[DEBUG] Rank {self.rank} incremented step_id to {self.step_id}")
                    
                    # Call entropy probe on full buffer after optimization step
                    if hasattr(self.algo, 'entropy_probe') and self.algo.entropy_probe.enabled:
                        print(f"[DEBUG] Rank {self.rank} calling entropy probe on full buffer")
                        full_buffer_batch = rb.get_all_as_batch(device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
                        self.algo.call_entropy_probe_on_buffer(full_buffer_batch)
                        print(f"[DEBUG] Rank {self.rank} completed entropy probe on full buffer")
                del mb, stats
                torch.cuda.empty_cache()
            print(f"[DEBUG] Rank {self.rank} completed PPO epoch {epoch+1}/{K}")
        stats_avg = {k: v / total_mb_cnt for k, v in stats_sum.items()}
        print(f"[DEBUG] Rank {self.rank} computed stats_avg, about to log")
        print(f"stats: {stats_avg}")
        print(f"[DEBUG] Rank {self.rank} calling _log with stats")
        self._log(stats_avg)
        print(f"[DEBUG] Rank {self.rank} completed _log, moving to probe section")

        # Save GNS probe data before cleanup for later processing (Option 2: Buffer Data Preservation)
        every = int(self.gns_cfg.get("every", 0))
        print(f"[DEBUG] Rank {self.rank} checking GNS probe: every={every}, step_id={self.step_id}, modulo={self.step_id % every if every > 0 else 'N/A'}")
        gns_saved_data = None
        if every > 0 and (self.step_id % every == 0) and self.rank == 0:
            print(f"[DEBUG] Rank {self.rank} saving GNS probe data before cleanup (step_id={self.step_id})")
            # Convert buffer to batch to access tensor data, then save minimal data needed for GNS probe
            batch = rb.to_batch(device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
            gns_saved_data = {
                'prompt_ids': batch.prompt_ids.clone().detach(),
                'gen_ids': batch.gen_ids.clone().detach(),
                'logprobs': batch.logprobs.clone().detach(), 
                'reward': batch.reward.clone().detach(),
                'tag_correct': batch.tag_correct.clone().detach(),
                'think_len': batch.think_len.clone().detach(),
                'step_id': self.step_id
            }
            print(f"[DEBUG] Rank {self.rank} saved GNS data: gen_ids.shape={gns_saved_data['gen_ids'].shape}")
            del batch  # Clean up the temporary batch
        elif every > 0 and (self.step_id % every == 0):
            print(f"[DEBUG] Rank {self.rank} skipping GNS probe data save (rank 0 only, step_id={self.step_id})")
        else:
            print(f"[DEBUG] Rank {self.rank} no GNS probe this step (step_id={self.step_id})")
        
        # Store GNS data for later processing (after cleanup)
        self._pending_gns_data = gns_saved_data
        # Cleanup rollout buffer and force garbage collection
        print(f"[DEBUG] Rank {self.rank} starting aggressive memory cleanup in _train_one_buffer")
        del rb
        torch.cuda.empty_cache()
        gc.collect()




    # ─── utils ────────────────────────────────────────────────────────────
    def _log(self, d):
        if self.rank != 0:
            return
        with open(self.dir / "logs" / "train_log.jsonl", "a") as f:
            f.write(json.dumps({"step": self.step_id, **d}) + "\n")
        if self.tb:
            for k, v in d.items():
                self.tb.add_scalar(k, float(v), self.step_id)
            self.tb.flush()

    def _save_ckpt(self, final: bool = False):
        tag = f"{self.step_id}" if not final else "final"
        save_dir = self.dir / "training_state" / f"step_{tag}"
        if self.rank == 0:
            # Save model (LoRA weights)
            model_dir = save_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            _unwrap(self.model).save_pretrained(model_dir)
            print(f"saved model to {model_dir}")
            
            # Save optimizer state
            optimizer_path = save_dir / "optimizer.pt"
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"saved optimizer state to {optimizer_path}")
            
            # Save scheduler state
            scheduler_path = save_dir / "scheduler.pt"
            torch.save(self.scheduler.state_dict(), scheduler_path)
            print(f"saved scheduler state to {scheduler_path}")
            
            # Save training info
            training_info = {
                "step": self.step_id,
                "global_step": getattr(self, 'global_step', self.step_id),
                "epoch": getattr(self, 'epoch', 0),
                "model_config": {
                    "backbone": self.cfg.get("backbone"),
                    "eval_backbone": self.cfg.get("eval_backbone")
                },
                "training_config": dict(self.cfg),
                "distributed_info": {
                    "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                    "rank": self.rank,
                    "local_rank": self.local_rank
                }
            }
            
            training_info_path = save_dir / "training_info.json"
            with open(training_info_path, 'w') as f:
                json.dump(training_info, f, indent=2)
            print(f"saved training info to {training_info_path}")
            
            # Create/update latest symlink
            latest_link = self.dir / "training_state" / "step_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(save_dir.name)
            print(f"updated latest symlink to {save_dir.name}")
            
            print(f"[CHECKPOINT] Full training state saved to {save_dir}")
        else:
            print(f"[DEBUG] Rank {self.rank} skipping checkpoint save (rank 0 only)")



    def _probe_gns(self, rb):
        """Measure ||g||^2 at two prompt-batch sizes, estimate B_simple, log it."""
        print(f"[GNS DEBUG] Rank {self.rank} entered _probe_gns")
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        if self.rank != 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns (not rank 0)")
            return
        if not self.gns_cfg or int(self.gns_cfg.get("every", 0)) <= 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns (config disabled)")
            return
        print(f"[GNS DEBUG] Rank {self.rank} starting GNS probe computation")

        import random
        B_small = self._gns_state["B_small"]
        B_large = self._gns_state["B_large"]
        ema = self._gns_state["ema"]
        print(f"[GNS DEBUG] B_small={B_small}, B_large={B_large}, ema={ema}")

        # sample prompt indices (without replacement) from the current rollout buffer
        N_prompts = len(rb)
        print(f"[GNS DEBUG] N_prompts={N_prompts}, need max({B_small}, {B_large})={max(B_small, B_large)}")
        if N_prompts < max(B_small, B_large):
            print(f"[GNS DEBUG] Not enough prompts for probe, exiting")
            return  # not enough to probe this time

        # index prompts we'll use for the two effective batches
        print(f"[GNS DEBUG] Sampling prompt indices")
        import random
        idx_small = random.sample(range(N_prompts), B_small)
        idx_large = random.sample(range(N_prompts), B_large)
        print(f"[GNS DEBUG] Sampled {len(idx_small)} small indices, {len(idx_large)} large indices")


        # helper: split a list of prompt indices into microbatches that fit in VRAM
        def _make_microbatches(idxs, micro_size):
            mbs = []
            for s in range(0, len(idxs), micro_size):
                mbs.append(rb.get_batch(idxs[s:s+micro_size], device=device))
            return mbs

        micro_size = int(self.cfg.get("prompts_per_microbatch", 1))
        print(f"[GNS DEBUG] Creating microbatches with micro_size={micro_size}")
        mbs_small = _make_microbatches(idx_small, micro_size)
        mbs_large = _make_microbatches(idx_large, micro_size)
        print(f"[GNS DEBUG] Created {len(mbs_small)} small microbatches, {len(mbs_large)} large microbatches")

        # emulate grad accumulation to measure ||g||^2 for the two effective batches
        print(f"[GNS DEBUG] Computing gradient norm for small batch")
        # NOTE: Removed old _grad_sq_norm_for_effective_batch method - using new GNS probe class instead
        # y_small = self.algo._grad_sq_norm_for_effective_batch(
        #     mbs_small, self.ref_model, avoid_ddp_allreduce=True
        # )
        y_small = 0.0  # Placeholder - this whole method is unused debug code
        print(f"[GNS DEBUG] Computed y_small={y_small}")
        print(f"[GNS DEBUG] Computing gradient norm for large batch")
        # y_large = self.algo._grad_sq_norm_for_effective_batch(
        #     mbs_large, self.ref_model, avoid_ddp_allreduce=True
        # )
        y_large = 0.0  # Placeholder - this whole method is unused debug code
        print(f"[GNS DEBUG] Computed y_large={y_large}")


        # EWMA for stability (Appendix A.1 suggests smoothing) 
        #   E[||g_B||^2] ≈ a + c/B  =>  solve from two points (B1,y1),(B2,y2)
        es = self._gns_state["ema_y_small"]
        el = self._gns_state["ema_y_large"]
        es = (ema * es + (1 - ema) * y_small) if es is not None else y_small
        el = (ema * el + (1 - ema) * y_large) if el is not None else y_large
        self._gns_state["ema_y_small"], self._gns_state["ema_y_large"] = es, el

        B1, y1 = float(B_small), float(es)
        B2, y2 = float(B_large), float(el)
        if abs(B1 - B2) < 1e-9:
            return

        # Solve:
        #   a = (B1*y1 - B2*y2) / (B1 - B2)
        #   c = B1*B2*(y2 - y1) / (B1 - B2)
        a_hat = (B1 * y1 - B2 * y2) / (B1 - B2)
        c_hat = (B1 * B2) * (y2 - y1) / (B1 - B2)
        B_simple = float("nan")
        if a_hat > 0 and c_hat > 0:
            B_simple = c_hat / a_hat

        rec = {
            "gns_y_small": y_small, "gns_y_large": y_large,
            "gns_y_small_ema": es, "gns_y_large_ema": el,
            "gns_B_small": B_small, "gns_B_large": B_large,
            "gns_a_hat": a_hat, "gns_c_hat": c_hat,
            "gns_B_simple": B_simple
        }
        # file + TB (add debugging around logging which might cause DDP hang)
        print(f"[GNS DEBUG] Rank {self.rank} about to log GNS results: {rec}")
        self._log(rec)
        print(f"[GNS DEBUG] Rank {self.rank} completed logging GNS results")




    def _probe_gns_from_saved_data(self, saved_data):
        """Measure ||g||^2 using saved tensor data instead of RolloutBuffer (Option 2 implementation)."""
        print(f"[GNS DEBUG] Rank {self.rank} entered _probe_gns_from_saved_data")
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        if self.rank != 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns_from_saved_data (not rank 0)")
            return
        if not self.gns_cfg or int(self.gns_cfg.get("every", 0)) <= 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns_from_saved_data (config disabled)")
            return
        print(f"[GNS DEBUG] Rank {self.rank} starting GNS probe computation from saved data")

        import random
        B_small = self._gns_state["B_small"]
        B_large = self._gns_state["B_large"]
        ema = self._gns_state["ema"]
        print(f"[GNS DEBUG] B_small={B_small}, B_large={B_large}, ema={ema}")

        # Extract saved data
        prompt_ids = saved_data['prompt_ids']  # (N, T_p)
        gen_ids = saved_data['gen_ids']  # (N, G, T_g)
        logprobs = saved_data['logprobs']  # (N, G, T_g)
        reward = saved_data['reward']  # (N, G)
        tag_correct = saved_data['tag_correct']  # (N, G)
        think_len = saved_data['think_len']  # (N, G)
        
        N_prompts = gen_ids.shape[0]
        print(f"[GNS DEBUG] N_prompts={N_prompts}, need max({B_small}, {B_large})={max(B_small, B_large)}")
        if N_prompts < max(B_small, B_large):
            print(f"[GNS DEBUG] Not enough prompts for probe, exiting")
            return

        # Sample prompt indices for the two effective batches
        print(f"[GNS DEBUG] Sampling prompt indices")
        idx_small = random.sample(range(N_prompts), B_small)
        idx_large = random.sample(range(N_prompts), B_large)
        print(f"[GNS DEBUG] Sampled {len(idx_small)} small indices, {len(idx_large)} large indices")

        # Helper: create microbatches from saved tensor data
        def _make_microbatches_from_tensors(idxs, micro_size):
            from rl_training.algs.base import RolloutBatch
            mbs = []
            for s in range(0, len(idxs), micro_size):
                batch_idxs = idxs[s:s+micro_size]
                # Create RolloutBatch objects from saved tensors
                mb = RolloutBatch(
                    prompt_ids=prompt_ids[batch_idxs].to(device),
                    gen_ids=gen_ids[batch_idxs].to(device),
                    reward=reward[batch_idxs].to(device),
                    logprobs=logprobs[batch_idxs].to(device),
                    tag_correct=tag_correct[batch_idxs].to(device),
                    think_len=think_len[batch_idxs].to(device)
                )
                mbs.append(mb)
            return mbs

        micro_size = int(self.cfg.get("prompts_per_microbatch", 1))
        print(f"[GNS DEBUG] Creating microbatches with micro_size={micro_size}")
        mbs_small = _make_microbatches_from_tensors(idx_small, micro_size)
        mbs_large = _make_microbatches_from_tensors(idx_large, micro_size)
        print(f"[GNS DEBUG] Created {len(mbs_small)} small microbatches, {len(mbs_large)} large microbatches")

        # Compute gradient norms for the two effective batches (same as original method)
        print(f"[GNS DEBUG] Computing gradient norm for small batch")
        # NOTE: Removed old _grad_sq_norm_for_effective_batch method - using new GNS probe class instead
        # y_small = self.algo._grad_sq_norm_for_effective_batch(
        #     mbs_small, self.ref_model, avoid_ddp_allreduce=True
        # )
        y_small = 0.0  # Placeholder - this whole method is unused debug code
        print(f"[GNS DEBUG] Computed y_small={y_small}")
        print(f"[GNS DEBUG] Computing gradient norm for large batch")
        # y_large = self.algo._grad_sq_norm_for_effective_batch(
        #     mbs_large, self.ref_model, avoid_ddp_allreduce=True
        # )
        y_large = 0.0  # Placeholder - this whole method is unused debug code
        print(f"[GNS DEBUG] Computed y_large={y_large}")

        # EWMA for stability (same as original method)
        es = self._gns_state["ema_y_small"]
        el = self._gns_state["ema_y_large"]
        es = (ema * es + (1 - ema) * y_small) if es is not None else y_small
        el = (ema * el + (1 - ema) * y_large) if el is not None else y_large
        self._gns_state["ema_y_small"], self._gns_state["ema_y_large"] = es, el

        B1, y1 = float(B_small), float(es)
        B2, y2 = float(B_large), float(el)
        if abs(B1 - B2) < 1e-9:
            return

        # Solve for B_simple (same as original method)
        a_hat = (B1 * y1 - B2 * y2) / (B1 - B2)
        c_hat = (B1 * B2) * (y2 - y1) / (B1 - B2)
        B_simple = float("nan")
        if a_hat > 0 and c_hat > 0:
            B_simple = c_hat / a_hat

        rec = {
            "gns_y_small": y_small, "gns_y_large": y_large,
            "gns_y_small_ema": es, "gns_y_large_ema": el,
            "gns_B_small": B_small, "gns_B_large": B_large,
            "gns_a_hat": a_hat, "gns_c_hat": c_hat,
            "gns_B_simple": B_simple
        }
        # file + TB (add debugging around logging which might cause DDP hang)
        print(f"[GNS DEBUG] Rank {self.rank} about to log GNS results: {rec}")
        self._log(rec)
        print(f"[GNS DEBUG] Rank {self.rank} completed logging GNS results")

    def cleanup(self):
        """Clean up resources, especially VLLM if used."""
        if hasattr(self.collector, 'cleanup'):
            print(f"[DEBUG] Rank {self.rank}: Cleaning up collector")
            self.collector.cleanup()
        
        # Clean up other resources
        if hasattr(self, 'ref_model'):
            del self.ref_model
        
        torch.cuda.empty_cache()
        print(f"[DEBUG] Rank {self.rank}: Cleanup completed")

    def _create_reference_model(self):
        """Create reference model from original checkpoint for consistent KL computation.
        
        This ensures KL divergence is always computed against the original fine-tuned 
        checkpoint, not the current model state (important for training resume).
        """
        print(f"[DEBUG] Creating reference model from original checkpoint: {self.original_lora_ckpt}")
        
        # Load the same base model setup
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(self.cfg["backbone"],
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb)
        base = prepare_model_for_kbit_training(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        # Load the original LoRA checkpoint (not current model state)
        ref_model = PeftModel.from_pretrained(base, self.original_lora_ckpt, is_trainable=False)
        
        # Reference model should be in eval mode and require no gradients
        ref_model = ref_model.eval().requires_grad_(False)
        
        print(f"[DEBUG] Reference model created successfully from {self.original_lora_ckpt}")
        return ref_model


# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint dir")
    # Resume training arguments
    p.add_argument("--resume-optimizer", help="Path to saved optimizer state (.pt file)")
    p.add_argument("--resume-scheduler", help="Path to saved scheduler state (.pt file)")
    p.add_argument("--resume-training-info", help="Path to saved training info (.json file)")
    # tolerate torchrun/launch variants; ignored by our code
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    # parse *known* only, to ignore any other launcher args
    args, _ = p.parse_known_args()

    cfg_dict = yaml.safe_load(open(args.cfg))
    
    # Determine if this is a resume operation
    resume_info = None
    if args.resume_training_info:
        import json
        with open(args.resume_training_info, 'r') as f:
            resume_info = json.load(f)
        print(f"🔄 Resuming training from step {resume_info['step']}")
    
    runner = RLRunner(args.cfg, args.ckpt, 
                      resume_optimizer_path=args.resume_optimizer,
                      resume_scheduler_path=args.resume_scheduler,
                      resume_info=resume_info)
    try:
        runner.train(total_updates=cfg_dict["total_steps"])
    finally:
        runner.cleanup()

