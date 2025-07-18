import random
import re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from rlp_datasets import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/mathmix_tagged")
SUBJECTS = [
    "algebra", "counting_and_probability", "prealgebra",
]
# Here just for reference, if I want to change the above
ALL_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra",
    "precalculus",
]
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

BAD_WORDS = [
    'draw', 'graph', 'array', 'diagram',
    'plot', 'chart', 'table', 'matrix',
    'tikz', 'coordinate', 'circle', 'ellipse', 'polygon',
]
# compile a regex that matches any word in the list, case-insensitive
_BAD_RX = re.compile(r'\b(?:' + r'|'.join(BAD_WORDS) + r')\b', re.IGNORECASE)
# also catch explicit TikZ environments
_TIKZ_RX = re.compile(r'\\begin\{tikzpicture\}')


@register("mathmix")
class MathMix(BaseDataset):
    """Streaming mix of GSM8K and Hendrycks MATH, with <think>/<answer> tags."""
    def __init__(self, split, gsm_weight=1.0, math_weight=1.0, seed=42):
        super().__init__(split)
        self.gsm_weight  = gsm_weight
        self.math_weight = math_weight
        self.seed        = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        # 1) get the two HF datasets
        gsm_ds = load_dataset(
            "openai/gsm8k", "main",
            split=self.split, cache_dir=RAW_DIR
        )
        # tag-split GSM8K answer into rationale vs final
        def fmt_gsm(row):
            q, a = row["question"].strip(), row["answer"].split("####")
            rationale, final = a[0].strip(), a[1].strip()
            return Example(
                text=f"{q}\n<think>\n{rationale}\n</think>\n"
                     f"<answer>\n{final}\n</answer>",
                question=q,
                answer=final,
                meta={"dataset": "gsm8k"}
            )

        # build a generator for GSM8K
        def gsm_iter():
            for row in gsm_ds:
                yield fmt_gsm(row)



        # 1) build Hendrycks Math in the same style
        math_splits = []
        for subj in SUBJECTS:
            # first load the split into a temporary var
            # 1) load the Hendrycks MATH split
            ds_subj = load_dataset(
                "EleutherAI/hendrycks_math",
                name=subj,
                split=self.split,
                cache_dir=RAW_DIR,
            )
            # 2) now that ds_subj exists, safely compute its length
            ds_subj = ds_subj.add_column(
                "subject",
                [subj] * len(ds_subj)
            )
            math_splits.append(ds_subj)

        # 2) concatenate once after the loop
        math_ds = concatenate_datasets(math_splits)

        def fmt_math(row):
            prob = row["problem"].strip()
            sol  = row["solution"].strip()
            m    = BOXED_RE.search(sol)
            final = m.group(1).strip() if m else ""
            return Example(
                text=f"{prob}\n<think>\n{sol}\n</think>\n"
                     f"<answer>\n{final}\n</answer>",
                question=prob,
                answer=final,
                meta={"dataset": "math", "subject": row["subject"]},
            )

        def math_iter():
            for row in math_ds:
                prob = row["problem"].strip()
                sol  = row["solution"].strip()
                # 1) skip if any bad word in the problem text
                if _BAD_RX.search(prob) or _TIKZ_RX.search(sol):
                    continue
                # 2) extract final answer as before
                m     = BOXED_RE.search(sol)
                final = m.group(1).strip() if m else ""
                yield Example(
                    text=f"{prob}\n<think>\n{sol}\n</think>\n"
                        f"<answer>\n{final}\n</answer>",
                    question=prob,
                    answer=final,
                    meta={"dataset": "math", "subject": row["subject"]},
                )


        # 3) Interleave them by weighted random sampling
        gsm_gen  = gsm_iter()
        math_gen = math_iter()
        # We'll stop when both are exhausted
        gsm_exhausted = math_exhausted = False

        while not (gsm_exhausted and math_exhausted):
            # pick which corpus to draw from
            total = 0.0
            if not gsm_exhausted:
                total += self.gsm_weight
            if not math_exhausted:
                total += self.math_weight
            pick = rng.random() * total

            if not gsm_exhausted and pick < self.gsm_weight:
                try:
                    yield next(gsm_gen)
                except StopIteration:
                    gsm_exhausted = True
            else:
                if not math_exhausted:
                    try:
                        yield next(math_gen)
                    except StopIteration:
                        math_exhausted = True
                else:
                    # if math is dry but GSM still has examples
                    try:
                        yield next(gsm_gen)
                    except StopIteration:
                        gsm_exhausted = True
