# evals/evaluator.py
import pandas as pd, importlib
from pathlib import Path
from evals.records import EvalRecord, save_records
from typing import Iterable, Callable, List

class Evaluator:
    def __init__(self,
                 record_iter: Iterable[EvalRecord],
                 metric_fns:  List[Callable],
                 out_dir: str):
        self.record_iter = list(record_iter)       # consume generator
        self.metric_fns  = metric_fns               # list of callables
        self.out_dir     = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

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
