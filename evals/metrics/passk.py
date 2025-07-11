from evals.metrics.tag_format import has_good_tags
from math_verify import parse, verify

def passk(records, k_vals=(1,2,4,8)):
    rows = []
    for r in records:
        flags = []
        for g in r.generations:
            ok = has_good_tags(g)
            flags.append(ok and verify(parse(f"${r.prompt.split('$')[-1]}$"),
                                       parse(f"${g.split('</answer>')[0].split('<answer>')[-1].strip()}$")))
        cum = 0
        row = {"q_idx": r.q_idx}
        for k in k_vals:
            cum = cum or any(flags[:k])
            row[f"pass@{k}"] = sum(flags[:k]) / k
            row[f"pass_any@{k}"] = int(any(flags[:k]))
        rows.append(row)
    return rows
