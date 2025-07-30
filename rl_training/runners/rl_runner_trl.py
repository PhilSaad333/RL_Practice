# rl_training/runners/rl_training_trl.py
import os, sys, re, argparse, torch
from pathlib import Path
import copy
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# ----------------------------- Data Utilities ------------------------------
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rlp_datasets.gsm8k_latex import build_gsm8k     # type: ignore
from math_verify import parse, verify

ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.MULTILINE)

def make_hf_dataset(split: str = "train"):
    examples = build_gsm8k(split)
    return Dataset.from_dict({
        "prompt":       [ex.question for ex in examples],
        "ground_truth": [ex.answer   for ex in examples],
    })

# ---------------------------------------------------------------------------
# ----------------------------- Reward Function -----------------------------
# ---------------------------------------------------------------------------

def tag_math_correct(prompts, completions, ground_truth, **kwargs):
    rewards = []
    for comp, gold in zip(completions, ground_truth):
        m = ANSWER_RE.search(comp)
        if not m:
            rewards.append(0.0)
            continue
        pred = m.group(1).strip()
        try:
            correct = verify(parse(gold), parse(pred))
        except Exception:
            correct = False
        rewards.append(1.0 if correct else 0.0)
    return rewards

# ---------------------------------------------------------------------------
# ------------------------------ Main Runner --------------------------------
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_ckpt",        type=str, required=True,
                   help="Path to the saved LoRA checkpoint directory.")
    p.add_argument("--out_dir",          type=str, required=True,
                   help="Where to write checkpoints / TB logs.")
    p.add_argument("--max_steps",        type=int, default=20,
                   help="Maximum total training steps (overrides epochs if set).")
    p.add_argument("--num_train_epochs", type=int, default=2,
                   help="Number of full-data epochs to train (TRL flag `--num_train_epochs`).")
    p.add_argument("--lr",               type=float, default=2e-5,
                   help="Learning rate for the optimizer.")
    p.add_argument("--num_gens",         type=int, default=6,
                   help="Num completions per prompt (G).")
    p.add_argument("--batch_size",       type=int, default=12,
                   help="Prompts per device per generation step.")
    p.add_argument("--accum",            type=int, default=16,
                   help="Gradient-accumulation steps.")
    args = p.parse_args()

    # ---------------------- Model & Tokenizer ------------------------------
    # 1) load base in 4-bit
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True)

    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base)   # <-- BEFORE adding LoRA
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    # 2) attach LoRA adapter *directory that contains adapter_config.json*
    #lora_dir = "/content/drive/MyDrive/RL_Practice_Files/finetuned/phi2_gsm8k_latex_lora/checkpoint-394"
    model = PeftModel.from_pretrained(base, args.lora_ckpt, is_trainable=True)

    # 3) sanity-check
    model.print_trainable_parameters() 

    tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ---------------------------- Dataset ----------------------------------
    train_ds = make_hf_dataset("train")

    # ----------------------- GRPO Configuration ---------------------------
    cfg = GRPOConfig(
        output_dir=args.out_dir,
        logging_steps=1,
        report_to="tensorboard",
        bf16=True,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,

        # NEW: control by epochs or steps
        num_train_epochs=args.num_train_epochs,  # TRL uses this exact name
        max_steps=args.max_steps,                # if >0, overrides epochs

        num_generations=args.num_gens,
        max_prompt_length=256,
        max_completion_length=128,

        beta=0.05,        # KL weight
        epsilon=0.2,      # PPO clip
        epsilon_high=0.5, # asymmetric upper clip
    )

    # ----------------------------- Trainer ---------------------------------
    model.train()           # allow gradients
    ref_model = copy.deepcopy(model).eval()
    trainer = GRPOTrainer(
        model=model,
        reference_model=ref_model,
        processing_class=tok,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=tag_math_correct,
    )

    # ---------------------------- Training ---------------------------------
    from transformers.utils import logging
    logging.set_verbosity_error()
    trainer.train()
    trainer.save_model(args.out_dir)


if __name__ == "__main__":
    main()
