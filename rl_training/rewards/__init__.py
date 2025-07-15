# rl_training/rewards/__init__.py

# To be used in rl_training/runners/collect_rollouts.py
def get_reward_fns(names: list[str]):
    from importlib import import_module
    fns = []
    for n in names:
        mod = import_module(f"rl_training.rewards.{n}")
        fns.append(mod.reward_fn)   # each module exposes `reward_fn(prompt, answers)->Tensor[G]`
    return fns
