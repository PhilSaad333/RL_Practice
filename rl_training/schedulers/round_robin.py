# rl_training/schedulers/round_robin.py
def get_prompt_sampler(cfg):
    prompts = ["2+2=", "1+3=", "6*7=", "5-2="]
    while True:
        for p in prompts:
            yield p
