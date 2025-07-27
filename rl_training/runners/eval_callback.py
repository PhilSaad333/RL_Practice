# rl_training/runners/eval_callback.py
import subprocess, shlex, pathlib

class EvalCallback:
    def __init__(self, run_dir: pathlib.Path, cfg: dict):
        self.run_dir   = run_dir
        self.every     = cfg.get("eval_every", 0)      # 0 = off
        self.script    = (pathlib.Path(__file__).parent
                          / "../../tools/eval_ckpt.py").resolve()

    def maybe_eval(self, step: int):
        if self.every and step % self.every == 0:
            cmd = f"python {self.script} --run_dir {self.run_dir} --step {step}"
            # fire-and-forget – no blocking
            subprocess.Popen(shlex.split(cmd),
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT)
