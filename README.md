# RL_Practice
Getting intuition for RL dynamics in the case of teaching LLMs to do math better

# Contents

RL_Practice/
├── README.md
├── main.py
├── env.txt # pip/conda environment spec (unused in Colab)
├── .gitignore
├── fine_tuning/
│ └── stf_phi2_lora.py
├── configs/
│ └── sft_phi2.yaml
├── datasets/
│ ├── GSM8K_dataset.py
│ └── processed/
│ └── gsm8k_tagged/
│ ├── test/
│ └── train/
├── evals/
│ ├── eval_runner.py
│ ├── evaluator.py
│ ├── records.py
│ ├── inspect_repl.py
│ ├── utils_io.py
│ └── metrics/
│ ├── tag_format.py
│ ├── passk.py
│ ├── response_len.py
│ └── entropy.py
└── rl_training/
└── (empty)
