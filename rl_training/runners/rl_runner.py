# rl_training/runners/rl_runner.py
from __future__ import annotations
import json, math, pathlib, datetime, yaml, torch, shutil, gc
import os, textwrap
from subprocess import run as run_sync
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange
from rl_training.runners.collect_rollouts import RolloutCollector
from rl_training.runners.eval_callback import EvalCallback
#from rl_training.algs.grpo import GRPO
from rl_training.algs.dr_grpo import DRGRPO
from rl_training.algs.base import RolloutBatch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel, prepare_model_for_kbit_training
from copy import deepcopy
import torch.distributed as dist



RUN_ROOT = pathlib.Path(
    os.environ.get("RUN_ROOT", "./rl_runs")
)

def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(prompt_ids = rb.prompt_ids[sl],
                        gen_ids    = rb.gen_ids[sl],
                        reward     = rb.reward[sl],
                        logprobs   = rb.logprobs[sl])

class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str):

        # -------- Stuff for multi gpu -----------------------------------
        # 1) Init the process group
        dist.init_process_group(backend="nccl")

        # 2) Figure out this process’s GPU index
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.local_rank = local_rank
        self.rank = dist.get_rank()


        # ---------- I/O ---------------------------------------------------
        cfg = yaml.safe_load(open(cfg_path))
        self.cfg = cfg
        stamp     = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir  = RUN_ROOT / f"run_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, self.dir / "config.yaml")       # save config

        if self.rank == 0:
            self.tb = SummaryWriter(str(self.dir))
        else:
            self.tb = None

        self.save_every = cfg.get("save_every", 100) #updated
        self.step_id    = 0

        # --------- load model -------------------------------------------------
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        backbone_id = cfg["backbone"]
        base = AutoModelForCausalLM.from_pretrained(
            backbone_id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb,
        )
        # prepare for 4-bit LoRA training
        base = prepare_model_for_kbit_training(base)

        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        model = PeftModel.from_pretrained(
            base,
            lora_ckpt,
            is_trainable=True,
        )

        model = model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

        model.enable_input_require_grads()

        self.model = model

        # for debug
        #self.model.print_trainable_parameters()


        self.tok   = AutoTokenizer.from_pretrained(backbone_id)
        if self.tok.pad_token_id is None:
            # safest practice is to duplicate eos so you don't expand embeddings
            self.tok.pad_token = self.tok.eos_token 

        # keep model & GRPO in sync
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id

        self.pad_id = self.tok.pad_token_id 
        self.tok.padding_side = "left"

        # frozen model for KL
        self.ref_model = deepcopy(self.model).eval().requires_grad_(False)
        #DEBUG
        #self.ref_model.to("cpu")
        # ensure pads match
        self.ref_model.config.pad_token_id = self.model.config.pad_token_id


        # ---------- subsystems ------------------------------------------
        self.collector = RolloutCollector(self.model, self.tok, cfg,
                                          out_dir=self.dir, device=f"cuda:{self.local_rank}")
        ratio_log = self.dir / "ratios.jsonl"

        # Just stick with DRGRPO for now, add option later after fixing up ordinary grpo                  
        self.algo      = DRGRPO(self.model, cfg, pad_id=self.tok.pad_token_id, ratio_log_path=ratio_log)
        self.accum = cfg["grad_accum_steps"]

        self.buffer_size = cfg["buffer_size"] # multiple of self.accum * self.collector.batch_size

        # for eval
        self.eval_cb = EvalCallback(self.dir, cfg)
        # for debug
        assert isinstance(self.tok.pad_token_id, int), "pad_token_id must not be None"


    # ---------------- main training loop ------------------------------
    def train(self, total_updates: int = 1000):
        """total_updates == number of *optimizer* steps (same definition GRPO uses)."""
        K         = self.cfg["ppo_epochs"]
        ga_steps  = self.accum                     # == cfg["grad_accum_steps"]
        B         = self.cfg["microbatch_size"]      # == cfg["microbatch_size"]

        outer_loops = math.ceil(total_updates / K)
        p_per_outer = self.buffer_size
        
        for _ in trange(outer_loops, desc="outer collect loops", disable=(self.rank != 0)):
            rb = self.collector.collect_batch(batch_prompts=p_per_outer)

            torch.cuda.empty_cache()
            gc.collect()

            self._train_one_buffer(rb, K, ga_steps, B)
            del rb               # free references to all batch tensors
            torch.cuda.empty_cache()
            gc.collect()

            if self.step_id % self.save_every == 0:
                self._save_ckpt()

        self._save_ckpt(final=True)

    def _train_one_buffer(self, rb, K, ga_steps, B):
        stats_sum  = defaultdict(float)

        total_mb_cnt = 0

        for epoch in range(K):
            micro_cnt = 0
            # iter_minibatches returns list of B indices
            for idx in rb.iter_minibatches(B, shuffle=True): 
                sync = ((micro_cnt + 1) % ga_steps == 0)   # last micro-batch → step
                mb   = rb.get_batch(idx, device="cuda")

                stats = self.algo.step(mb, self.ref_model, sync_grads=sync)
                for k, v in stats.items():
                    stats_sum[k] += v
                micro_cnt += 1
                total_mb_cnt += 1
                if sync:
                    self.step_id += 1
                del mb, stats
                torch.cuda.empty_cache()
        

        # --- final average over *all* micro-batches processed ---
        stats_avg = {k: v / total_mb_cnt for k, v in stats_sum.items()}
        print(f"stats: {stats_avg}")
        self._log(stats_avg)


    def _log(self, d):
        if self.rank != 0:
            return
        json_out = {"step": self.step_id, **d}
        with open(self.dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(json_out) + "\n")

        for k, v in d.items():               # ← use d, not stats_avg
            self.tb.add_scalar(k, float(v), self.step_id)

        self.tb.flush()

    def _save_ckpt(self, final: bool = False):
        tag = f"{self.step_id}" if not final else "final"

        save_dir = self.dir / f"step_{self.step_id}"
        if self.rank == 0:
            # Only rank 0 writes the checkpoint
            self.model.module.save_pretrained(save_dir)
            print(f"saved model to {save_dir}")
        # barrier: wait for save to finish
        torch.distributed.barrier()

        # Evaluation: only rank 0 runs it, others wait
        if self.rank == 0:
            self._run_eval(save_dir)
        torch.distributed.barrier()


    def _run_eval(self, ckpt_dir: pathlib.Path):
        print(f"[Eval] starting eval for step {self.step_id} …")

        # 1) free GPU RAM from the training model
        self.model.to("cpu"); torch.cuda.empty_cache(); gc.collect()

        # 2) build eval command
        cmd = textwrap.dedent(f"""
            python -m evals.eval_runner
                --backbone {self.cfg['eval_backbone']}
                --ft_dataset {self.cfg["scheduler"]["dataset_name"]}
                --ckpt_path {ckpt_dir}
                --ckpt_step {self.step_id}
                --batch_size {self.cfg.get('eval_batch_size',8)}
                --subset_frac {self.cfg.get('eval_frac',1.0)}
                --eval_dataset {self.cfg["scheduler"]["dataset_name"]}
                --temperature {self.cfg.get('eval_temperature',0.7)} --top_p 1.0
                --runs_root /content/drive/MyDrive/RL_Practice_Files/eval_runs/rl_evals
        """).split()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # Improve to follow cmd structure from above
        subprocess.run(
            ["python", "evals/eval_runner.py", "--ckpt", str(save_dir), "--runs_root", str(self.eval_root)],
            env=env,
            check=True,
        )

        # 3) run *synchronously* (blocks until eval finishes)
        run_sync(cmd, env=env, check=True)

        # 4) clean up eval model & reload training model to GPU
        torch.cuda.empty_cache(); gc.collect()
        self.model.to("cuda")
        print(f"[Eval] finished, resuming training.")

# ------------------------------- CLI -------------------------------------- #
if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint dir")
    args = p.parse_args()
    cfg_dict = yaml.safe_load(open(args.cfg))
    runner = RLRunner(args.cfg, args.ckpt)
    runner.train(total_updates=cfg_dict["total_steps"])
