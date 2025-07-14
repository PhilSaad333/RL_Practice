# datasets/GSM8K_dataset.py

# OLD, USE gsm8k.py

from datasets import load_dataset
from pathlib import Path

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/gsm8k_tagged")
PROC_DIR.mkdir(parents=True, exist_ok=True)        # cross-platform mkdir :contentReference[oaicite:4]{index=4}

def _wrap(example):
    question = example["question"]
    # GSM8K answer string is "... #### 42"
    rationale, final = example["answer"].split("####")
    return {"text": f"{question}\n<think>\n{rationale.strip()}\n</think>\n"
                    f"<answer>\n{final.strip()}\n</answer>"}

def build(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=RAW_DIR)  # downloads to datasets/raw/ :contentReference[oaicite:5]{index=5}
    ds = ds.map(_wrap, remove_columns=ds.column_names)                # map + drop old cols :contentReference[oaicite:6]{index=6}
    # 1️⃣ JSONL (small & diff-friendly)
    json_path = PROC_DIR / f"{split}.jsonl"
    ds.to_json(str(json_path))                                        # :contentReference[oaicite:7]{index=7}
    # 2️⃣ Arrow (fast reload for training)
    arrow_path = PROC_DIR / split
    ds.save_to_disk(str(arrow_path))                                  # :contentReference[oaicite:8]{index=8}
    print(f"✓ {split}: {len(ds):,} rows → {json_path} & {arrow_path}/")

if __name__ == "__main__":
    build("train")
    build("test")           # GSM8K “test” acts as dev/val split
