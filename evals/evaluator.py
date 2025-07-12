# evals/evaluator.py
import json
import pandas as pd, importlib
from pathlib import Path
from evals.records import EvalRecord, save_records
from typing import Iterable, Callable, List

class Evaluator:
    def __init__(self,
                 record_iter: Iterable[EvalRecord],
                 metric_fns:  List[Callable],
                 out_dir: str,
                 subset_frac: float,
                 batch_size: int,
                 ):

        self.record_iter = list(record_iter)        # recs is normally list anyways
        self.metric_fns  = metric_fns
        self.out_dir     = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.subset_frac = subset_frac
        self.batch_size  = batch_size

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