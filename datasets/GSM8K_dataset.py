# datasets/GSM8K_dataset.py
from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/gsm8k_tagged")
PROC_DIR.mkdir(parents=True, exist_ok=True)        # pathlib handles paths cross-OS  :contentReference[oaicite:6]{index=6}

def tag_format(example):
    return {
        "text": f"{example['question']}\n"
                f"<think>\n{example['answer'].split('####')[0].strip()}\n</think>\n"
                f"<answer>\n{example['answer'].split('####')[-1].strip()}\n</answer>"
    }

def build(split="train"):
    ds = load_dataset("openai/gsm8k", split=split, cache_dir=RAW_DIR)  #  :contentReference[oaicite:7]{index=7}
    ds = ds.map(tag_format, remove_columns=ds.column_names)
    outfile = PROC_DIR / f"{split}.jsonl"
    ds.to_json(str(outfile))                                           #  :contentReference[oaicite:8]{index=8}
    print(f"✓ saved {len(ds):,} examples to {outfile}")

if __name__ == "__main__":
    build("train")
    build("test")
