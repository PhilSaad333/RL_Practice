"""
Generic LoRA-SFT trainer.

In colab, installs:

!pip uninstall -y -q gcsfs

# 2 · core deps (no implicit “extras”)
!pip install -q pyarrow_hotfix tyro
!pip install -q --no-deps fsspec==2024.2.0
!pip install -q --no-deps datasets==2.19.2
!pip install -q --no-deps transformers==4.53.0
!pip install -q --no-deps peft==0.10.0
!pip install -q --no-deps trl==0.8.6
!pip install -q --no-deps accelerate>=1.8.1

# 3 · 4-bit / LoRA helpers
!pip install -q --no-deps bitsandbytes==0.43.2 triton==3.1.0






Run, e.g.:

    python -m fine_tuning.sft \
        --config fine_tuning/configs/sft_gsm8k_tinyllama.yaml
"""
from pathlib import Path
import yaml, tyro, torch
from datasets import load_dataset, load_from_disk
from typing import Literal
from models import load_model
from rlp_datasets import DATASET_REGISTRY
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    max_seq_len: int = 256
    lr_scheduler_type: Literal["linear","cosine","constant"] = "linear"
    warmup_ratio: float = 0.05
    packing: bool = True



# ------------------------------- helpers --------------------------------
def load_text_dataset(name: str, split: str):
    """
    Instantiate your registry-backed loader (yields Example objects),
    convert to a list of dicts, and wrap in a HuggingFace Dataset.
    """
    from datasets import Dataset

    # 1) materialize all examples from your iterable loader
    examples = list(DATASET_REGISTRY[name](split))

    # 2) each Example has .text and .meta fields, so flatten:
    records = []
    for ex in examples:
        rec = {"text": ex.text}
        # if you want to keep metadata fields as columns:
        rec.update(ex.meta)
        records.append(rec)

    # 3) build and return the Dataset
    return Dataset.from_list(records)

def main(cfg: Config):
    # 1) Data -------------------------------------------------------------
    train_ds = load_text_dataset(cfg.dataset, "train")
    val_ds   = load_text_dataset(cfg.dataset, "test")

    # 2) Model + Tok ------------------------------------------------------
    model, tok = load_model(cfg.backbone,
                            quantized=cfg.quantized,
                            device_map="auto")     # FP4 if quantized
    tok.pad_token = tok.eos_token     # safe default for most CAUSAL_LM


    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # SPECIFIC TO PHI2 - CHANGE IF YOU USE A DIFFERENT MODEL!
    lora_cfg = LoraConfig(
        r               = cfg.lora_r,
        lora_alpha      = cfg.lora_alpha,
        lora_dropout    = cfg.lora_dropout,
        bias            = "none",
        task_type       = "CAUSAL_LM",
        target_modules=[
            "Wqkv",          # attention projections in Phi-2
            "out_proj",      # output projection
            "fc1", "fc2",    # MLP
            "dense"          # residual projection
        ],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters() #debug

    # 3) Training args ----------------------------------------------------
    out_dir = cfg.output_dir or f"outputs/{cfg.backbone}_{cfg.dataset}_lora"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=float(cfg.lr),
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.log_steps,
        gradient_checkpointing=True,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        label_names=["labels"],
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=1.0,            # carried over from your old defaults
        disable_tqdm=False,
        report_to="none",
    )                                        # HuggingFace trainer API :contentReference[oaicite:2]{index=2}

    # 4) Trainer ----------------------------------------------------------
    trainer = SFTTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        tokenizer        = tok,
        dataset_text_field= "text",
        max_seq_length   = cfg.max_seq_len,
        packing          = False, # explicit for now
    )                                          # TRL docs :contentReference[oaicite:3]{index=3}

    trainer.train()
    trainer.save_model(out_dir)                # LoRA weights only
    tok.save_pretrained(out_dir)
    print("✓ done →", out_dir)


if __name__ == "__main__":
    import argparse, yaml
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LoRA SFT trainer")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to a YAML config file")
    args = parser.parse_args()

    # 1) Load your specified YAML
    cfg_path = Path(args.config)
    cfg_dict = yaml.safe_load(cfg_path.read_text()) or {}

    # 2) Handle `inherit:` if present
    parent_key = cfg_dict.pop("inherit", None)
    if parent_key:
        parent_path = cfg_path.parent / parent_key
        parent_cfg = yaml.safe_load(parent_path.read_text()) or {}
        # Merge: parent < child
        parent_cfg.update(cfg_dict)
        cfg_dict = parent_cfg

    # 3) Now cfg_dict contains only fields that match Config(...)
    cfg = Config(**cfg_dict)
    print("✔ Loaded config:", cfg)

    # 4) Launch the trainer
    main(cfg)



