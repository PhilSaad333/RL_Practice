from .tag_format import has_good_tags    # reuse regex criterion
def pass_at_k(records, k=(1,2,4,8)):
    out = []
    for r in records:
        flags = [has_good_tags(t) and verify_gold(r,q,t) for t in r.generations]
        row   = {"q_idx": r.q_idx}
        cum = 0
        for kk in k:
            cum = cum or any(flags[:kk])
            row[f"pass@{kk}"] = int(cum)
        out.append(row)
    return pd.DataFrame(out)
