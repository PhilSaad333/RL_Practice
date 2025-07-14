# evals/evaluator.py
import json
import pandas as pd, importlib
from pathlib import Path
from typing import Union
from datetime import datetime
from evals.records import EvalRecord, save_records
from typing import Iterable, Callable, List

class Evaluator:
    def __init__(self,
                 backbone: str,
                 ft_dataset: str,
                 ckpt_step: str | None,
                 eval_dataset: str,
                 batch_size: int = 8,
                 runs_root: Union[str,Path] = "eval_runs"
                 model_path: str | None = None,
                 **gen_kwargs):
        """
        backbone      : registry key, e.g. 'tinyllama' or 'phi2'
        ft_dataset    : dataset used during fine-tune (gsm8k, math, …)
        ckpt_step     : '500', '1000', … (None if evaluating base model)
        eval_dataset  : dataset name for prompts
        model_path    : explicit local path (overrides backbone)
        """
        
        root = Path(runs_root)
        self.run_dir = (
            root
            / f"{backbone}_{ft_dataset}_finetuned"
            / f"step_{ckpt_step or 'base'}_{eval_dataset}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)  # pathlib trick :contentReference[oaicite:1]{index=1}

        target = model_path or backbone
        self.model, self.tok, self.prompts, self.golds, self.stopper = (
            load_everything(target, eval_dataset)
        )
        
        self.batch_size = batch_size
        self.gen_cfg = gen_kwargs
        self.timestamp = datetime.utcnow().isoformat()

    def run(self):
        # 1) save raw records once
        raw_path = self.out_dir / "records.jsonl.gz"
        save_records(self.record_iter, raw_path)

        # 2) compute & merge metric dataframes
        dfs = [pd.DataFrame(fn(self.record_iter)) for fn in self.metric_fns]
        df  = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, on="q_idx", how="left")

        df.to_csv(self.out_dir / "metrics.csv", index=False)
        print(f"✓ wrote metrics.csv with {len(df)} rows to {self.out_dir}")

        # 3) save global metadata
        meta = {
            "ckpt_dir": str(self.out_dir.parents[1] / f"checkpoint-{self.record_iter[0].step}"),
            "step":     self.record_iter[0].step,
            "subset_frac":  self.subset_frac,
            "batch_size":   self.batch_size,
            "decoding_cfg": self.record_iter[0].cfg,
        }
        Path(self.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))