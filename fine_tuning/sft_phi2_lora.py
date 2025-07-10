"""
LoRA-based SFT of Phi-2 on a tag-wrapped GSM8K dataset.
Run:
    python -m RL_Practice.fine_tuning.sft_phi2_lora --config configs/sft_phi2.yaml
"""
import argparse, yaml, torch
from pathlib import Path
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()

def main():
    cfg = yaml.safe_load(Path(parse_args().config).read_text())

    # 1️⃣  Dataset -----------------------------------------------------------------
    def _load(ds_path):
        path = Path(ds_path)
        if path.is_dir():
            return load_from_disk(str(path))
        return load_dataset("json", data_files=str(path))["train"]  # JSONL
    train_ds = _load(cfg["dataset_train"])
    val_ds   = _load(cfg["dataset_val"])

    # 2️⃣  Model + Tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token    # Phi-2 has no explicit pad token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=cfg["load_in_4bit"],
        torch_dtype=torch.float16 if cfg["fp16"] else torch.float32,
    )

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)      # attaches adapters

    # 3️⃣  TrainingArguments -------------------------------------------------------
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        evaluation_strategy="steps",
        fp16=cfg["fp16"],
        report_to="none",
    )

    # 4️⃣  SFT-Trainer ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,   # Phi-2 was trained with 2048; 1k saves memory
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])        # saves LoRA adapter only
    tokenizer.save_pretrained(cfg["output_dir"])
    print("✓ Fine-tune complete →", cfg["output_dir"])

if __name__ == "__main__":
    main()
