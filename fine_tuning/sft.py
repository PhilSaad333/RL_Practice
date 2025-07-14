"""
Generic LoRA-SFT trainer.
Run, e.g.:

    python -m fine_tuning.sft \
        --config configs/sft_gsm8k_tinyllama.yaml
"""
from pathlib import Path
import yaml, tyro, torch
from datasets import load_dataset, load_from_disk
from models import load_model
from rlp_datasets import DATASET_REGISTRY
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, GenerationConfig
from trl import SFTTrainer

# -------------------------- dataclass for tyro --------------------------
from dataclasses import dataclass, field

@dataclass
class Config:
    # mandatory
    backbone: str
    dataset: str
    # optional (overrides template defaults)
    output_dir: str | None = None
    quantized: bool = False
    batch_size: int = 4
    lr: float = 2e-5
    epochs: int = 3
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    grad_accum: int = 1
    save_steps: int = 500
    log_steps: int = 100
    max_seq_len: int = 1024


# ------------------------------- helpers --------------------------------
def load_text_dataset(name: str, split: str):
    ds = DATASET_REGISTRY[name](split)
    # rlp_datasets yields an iterable of Example objects; wrap in HF Dataset
    tmp = [e.__dict__ for e in ds]          # convert dataclass -> dict
    return load_dataset("json", data_files={"train": tmp})["train"] \
           if split == "train" else \
           load_dataset("json", data_files={"test": tmp})["test"]


def main(cfg: Config):
    # 1) Data -------------------------------------------------------------
    train_ds = load_text_dataset(cfg.dataset, "train")
    val_ds   = load_text_dataset(cfg.dataset, "test")

    # 2) Model + Tok ------------------------------------------------------
    model, tok = load_model(cfg.backbone,
                            quantized=cfg.quantized,
                            device_map="auto")     # FP4 if quantized
    tok.pad_token = tok.eos_token     # safe default for most CAUSAL_LM

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )                                         # :contentReference[oaicite:1]{index=1}
    model = get_peft_model(model, lora_cfg)

    # 3) Training args ----------------------------------------------------
    out_dir = cfg.output_dir or f"outputs/{cfg.backbone}_{cfg.dataset}_lora"
    args = TrainingArguments(
        output_dir          = out_dir,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        gradient_accumulation_steps = cfg.grad_accum,
        learning_rate      = cfg.lr,
        num_train_epochs   = cfg.epochs,
        logging_steps      = cfg.log_steps,
        save_steps         = cfg.save_steps,
        evaluation_strategy= "steps",
        fp16               = torch.cuda.is_available(),
        bf16               = False,
        report_to          = "none",
    )                                          # HuggingFace trainer API :contentReference[oaicite:2]{index=2}

    # 4) Trainer ----------------------------------------------------------
    trainer = SFTTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        tokenizer        = tok,
        dataset_text_field= "text",
        max_seq_length   = cfg.max_seq_len,
    )                                          # TRL docs :contentReference[oaicite:3]{index=3}

    trainer.train()
    trainer.save_model(out_dir)                # LoRA weights only
    tok.save_pretrained(out_dir)
    print("✓ done →", out_dir)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
