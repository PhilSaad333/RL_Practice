import re
TAG_RGX = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)

def has_good_tags(txt: str) -> bool:
    return bool(TAG_RGX.search(txt))
