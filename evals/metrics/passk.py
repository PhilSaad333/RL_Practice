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
            row[f"pass@{k}"] = int(cum)
        row["tag_ok_first"] = int(has_good_tags(r.generations[0]))
        row["tag_ok_any"]   = int(any(has_good_tags(g) for g in r.generations))
        rows.append(row)
    return rows
