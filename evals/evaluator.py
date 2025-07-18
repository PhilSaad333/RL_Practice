# evals/evaluator.py
import json
import pandas as pd, importlib
from pathlib import Path
from typing import Union
from datetime import datetime
from evals.records import EvalRecord, save_records
from typing import Iterable, Callable, List
from evals.utils_io import load_everything

class Evaluator:
    def __init__(self,
                 backbone: str,
                 ft_dataset: str,
                 ckpt_step: str | None,
                 eval_dataset: str,
                 batch_size: int = 8,
                 subset_frac: float = 1.0,
                 runs_root: Union[str,Path] = "eval_runs",
                 model_path: str | None = None,
                 **gen_kwargs):
        """
        backbone      : registry key, e.g. 'tinyllama' or 'phi2'
        ft_dataset    : dataset used during fine-tune (gsm8k, math, …)
        ckpt_step     : '500', '1000', … (None if evaluating base model)
        eval_dataset  : dataset name for prompts
        model_path    : explicit local path (overrides backbone)
        **gen_kwargs  : decoding params (temperature, top_p, num_return_sequences, etc.)
        """
        
        root = Path(runs_root)
        base_dir = (
            root
            / f"{backbone}_{ft_dataset}_finetuned"
            / f"step_{ckpt_step or 'base'}_{eval_dataset}"
        )

        temp = gen_kwargs.get("temperature", "NA")
        top_p = gen_kwargs.get("top_p",       "NA")
        nret  = gen_kwargs.get("num_return_sequences", "NA")
        params_folder = f"temp{temp}_p{top_p}_r{nret}"

        self.run_dir = base_dir / params_folder
        self.run_dir.mkdir(parents=True, exist_ok=True)

        target = model_path or backbone
        self.model, self.tok, self.prompts, self.golds, self.stopper = (
            load_everything(target, eval_dataset, ckpt_path=model_path)
        )
        
        self.batch_size    = batch_size
        self.subset_frac   = subset_frac
        self.gen_cfg       = gen_kwargs
        self.timestamp     = datetime.utcnow().isoformat()

        # 4 metric functions to apply (must match your imports)
        from evals.metrics.tag_format import tag_format_metrics
        from evals.metrics.passk import passk_metrics
        from evals.metrics.response_len import response_len_metrics
        from evals.metrics.entropy import entropy_metrics
        self.metric_fns    = [
            tag_format_metrics,
            passk_metrics,
            response_len_metrics,
            entropy_metrics,
        ]

        # placeholder for the actual EvalRecord list
        self.record_iter   = []

        # alias for convenience (run() writes to self.out_dir)
        self.out_dir       = self.run_dir


    def run(self):
        # 1) save raw records once
        raw_path = self.run_dir / "records.jsonl.gz"
        save_records(self.record_iter, raw_path)

        # 2) compute & merge metric dataframes
        dfs = [pd.DataFrame(fn(self.record_iter)) for fn in self.metric_fns]
        df  = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, on="q_idx", how="left")

        df.to_csv(self.run_dir / "metrics.csv", index=False)
        print(f"✓ wrote metrics.csv with {len(df)} rows to {self.run_dir}")

        # 3) save global metadata
        meta = {
            "ckpt_dir": str(self.run_dir.parents[1] / f"checkpoint-{self.record_iter[0].step}"),
            "step":     self.record_iter[0].step,
            "subset_frac":  self.subset_frac,
            "batch_size":   self.batch_size,
            "decoding_cfg": self.record_iter[0].cfg,
        }
        Path(self.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))