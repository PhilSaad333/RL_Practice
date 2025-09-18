# utils/eval_ckpt.py
import argparse, subprocess, shlex, pathlib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--step", required=True, type=int)
    args = p.parse_args()

    run   = pathlib.Path(args.run_dir)
    ckpt  = run / f"checkpoint-{args.step}"
    assert ckpt.exists(), f"{ckpt} missing"

    cmd = f"""
    python -m evals.eval_runner \
        --backbone phi2 \
        --ft_dataset gsm8k_latex \
        --ckpt_path {ckpt} \
        --ckpt_step {args.step} \
        --batch_size 12 \
        --subset_frac 1.0 \
        --eval_dataset gsm8k_latex \
        --temperature 0.7 --top_p 1.0 \
        --runs_root /content/drive/MyDrive/RL_Practice_Files/eval_runs/rl_evals
    """

    # launch **detached** so RL keeps running
    subprocess.Popen(shlex.split(cmd),
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.STDOUT)

if __name__ == "__main__":
    main()
