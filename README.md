├── env.txt
├── evals
│   ├── analyses
│   │   ├── high_entropy_tokens.py
│   │   ├── __init__.py
│   │   └── plot_summary_metrics.py
│   ├── cli.py
│   ├── eval_runner.py
│   ├── evaluator.py
│   ├── __init__.py
│   ├── inspect_repl.py
│   ├── metrics
│   │   ├── entropy.py
│   │   ├── __init__.py
│   │   ├── passk.py
│   │   ├── response_len.py
│   │   └── tag_format.py
│   ├── records.py
│   └── utils_io.py
├── fine_tuning
│   ├── configs
│   │   ├── sft_gsm8k_qwen2.yaml
│   │   ├── sft_gsm8k_tinyllama.yaml
│   │   ├── sft_math_phi2.yaml
│   │   └── _template_sft.yaml
│   ├── __init__.py
│   └── sft.py
├── __init__.py
├── main.py
├── models
│   └── __init__.py
├── README.md
├── rlp_datasets
│   ├── gsm8k_latex.py
│   ├── gsm8k.py
│   ├── __init__.py
│   ├── local_paths.py
│   ├── mathmix.py
│   ├── math.py
│   ├── registry.py
│   └── short.py
└── rl_training
    ├── algs
    │   ├── base.py
    │   ├── drgrpo.py
    │   └── grpo.py
    ├── cfg
    │   ├── grpo_gsm8k_phi2.yaml
    │   ├── grpo_gsm8k_qwen2.yaml
    │   └── testconfig.yaml
    ├── rewards
    │   ├── dummy_zero.py
    │   ├── __init__.py
    │   ├── tag_math_correct.py
    │   └── tag_pref.py
    ├── runners
    │   ├── collect_rollouts_old.py
    │   ├── collect_rollouts.py
    │   ├── eval_callback.py
    │   ├── rl_runner.py
    │   └── rl_runner_trl.py
    ├── schedulers
    │   └── mix_passrate.py
    └── utils
        ├── eval_ckpt.py
        └── rollout_buffer.py