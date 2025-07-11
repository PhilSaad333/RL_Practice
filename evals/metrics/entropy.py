
def token_entropy(logp):
    p = np.exp(logp)
    return -(p * logp).sum()            # per-token H


def avg_entropy(records):
    rows = []
    for r in records:
        entropies = [token_entropy(lp).mean() for lp in r.logprobs]
        rows.append({"q_idx": r.q_idx,
                     "H_avg_first":    entropies[0],
                     "H_avg_samples":  np.mean(entropies)})
    return pd.DataFrame(rows)
