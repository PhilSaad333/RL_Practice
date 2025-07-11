class Evaluator:
    def __init__(self, ckpt_dir, data_dir, gen_cfg, metrics, hooks=None):
        self.metrics = metrics          # list of callables
        self.hooks   = hooks or []

    def run(self):
        records = list(self._generate_all())      # or stream
        dfs     = [m(records) for m in self.metrics]
        full    = reduce(lambda a,b: a.merge(b,on="q_idx"), dfs)
        full.to_parquet(f"{self.ckpt_dir}/eval.parquet")
