# evals/cli.py
from pathlib import Path
import importlib
import tyro

def main(
    analysis: str,          # e.g. "summarystats" or "entropy"
    base_root: Path,        # directory that holds step_* dirs
    # forward-through flags for your analyses:
    show: bool = False,
    save_dir: Path = None,
    gens_per_prompt: int = 8,
    **kwargs,               # forwarded to the analysis.main(...)
):
    """
    Dispatch to evals.analyses.<analysis>.main().
    Example:
        python -m evals.cli summarystats /path --show
    """
    mod_name = f"evals.analyses.{analysis}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise SystemExit(f"[ERROR] Unknown analysis '{analysis}'. "
                         "Check evals/analyses/ or tyro --help") from e

    if not hasattr(mod, "main"):
        raise SystemExit(f"[ERROR] {mod_name} has no `main` callable")

    return mod.main(base_root=base_root, show=show, save_dir=save_dir, gens_per_prompt=gens_per_prompt, **kwargs)

if __name__ == "__main__":
    tyro.cli(main, description="Unified entry-point for evaluation analyses.")
