├── env.txt
├── evals
│   ├── analyses
│   │   ├── high_entropy_tokens.py
│   │   ├── __init__.py
│   │   ├── plot_summary_metrics.py
│   │   └── token_dump.py
│   ├── cli.py
│   ├── eval_runner.py
│   ├── evaluator.py
│   ├── __init__.py
│   ├── inspect_repl.py
│   ├── metrics
│   │   ├── entropy.py
│   │   ├── __init__.py
│   │   ├── max_correct_len.py
│   │   ├── passk.py
│   │   ├── response_len.py
│   │   └── tag_format.py
│   ├── records.py
│   ├── token_utils.py
│   └── utils_io.py
├── fine_tuning
│   ├── configs
│   │   ├── sft_gsm8k_gemma.yaml
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
├── requirements.txt
├── rlp_datasets
│   ├── gsm8k_latex.py
│   ├── gsm8k.py
│   ├── gsm8k_r1_template.py
│   ├── __init__.py
│   ├── local_paths.py
│   ├── mathmix.py
│   ├── math.py
│   ├── processed
│   │   ├── gsm8k_latex_test.jsonl
│   │   └── gsm8k_latex_train.jsonl
│   ├── registry.py
│   └── short.py
└── rl_training
    ├── algs
    │   ├── base.py
    │   └── dr_grpo.py
    ├── cfg
    │   ├── grpo_gsm8k_phi2.yaml
    │   ├── qwen2_5_15.yaml
    │   ├── qwen2.yaml
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
    │   ├── rl_runner_old.py
    │   ├── rl_runner.py
    │   └── rl_runner_trl.py
    ├── schedulers
    │   └── mix_passrate.py
    └── utils
        ├── eval_ckpt.py
        ├── local_paths.py
        ├── logprob_entropy.py
        └── rollout_buffer.py