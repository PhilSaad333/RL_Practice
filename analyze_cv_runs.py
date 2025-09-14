#!/usr/bin/env python3
# analyze_cv_runs.py
import json, os, math, argparse
from pathlib import Path
from typing import List, Dict
import csv

def load_index(root: Path) -> List[Dict]:
    idx = json.loads((root / "index.json").read_text())
    return idx

def load_summary(path: str) -> Dict:
    return json.loads(Path(path).read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="Root folder created by run_control_variate_analysis.py (…/<label>_<ts>)")
    p.add_argument("--out_csv", type=str, default="cv_runs_aggregate.csv")
    p.add_argument("--out_json", type=str, default="cv_runs_aggregate.json")
    args = p.parse_args()

    root = Path(args.root)
    idx = load_index(root)

    rows = []
    for entry in idx:
        if "error" in entry:  # skip failed runs
            continue
        summ_path = entry.get("cv_summary") or entry.get("summary", {}).get("paths", {}).get("summary_path")
        if not summ_path: 
            continue
        s = load_summary(summ_path)
        diag = s.get("diagnostics", {})
        g = float(diag.get("gdotv_batch", float("nan")))
        se = float(diag.get("se_batch", float("nan")))
        se_after = float(diag.get("se_batch_after_cv_joint", float("nan")))
        vr = se_after / se if (se > 0 and math.isfinite(se_after)) else float("nan")
        snr = abs(g) / se if (se > 0 and math.isfinite(g)) else float("nan")
        snr_after = abs(g) / se_after if (se_after > 0 and math.isfinite(g)) else float("nan")

        rows.append({
            "run_dir": entry.get("run_dir"),
            "gdotv_batch": g,
            "se_batch": se,
            "se_batch_after": se_after,
            "VR_SE": vr,
            "SNR": snr,
            "SNR_after": snr_after,
            "features": ",".join(s.get("diagnostics", {}).get("features", [])),
            "B_total": s.get("diagnostics", {}).get("meta", {}).get("B_total"),
            "T_total": s.get("diagnostics", {}).get("meta", {}).get("T_total"),
        })

    # Write CSV
    out_csv = root / args.out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Write JSON summary (means/medians)
    import statistics as st
    def safe(vals, fn):
        vals = [v for v in vals if v == v and math.isfinite(v)]
        return fn(vals) if vals else float("nan")

    agg = {
        "n_runs": len(rows),
        "mean_abs_g": safe([abs(r["gdotv_batch"]) for r in rows], st.mean),
        "median_abs_g": safe([abs(r["gdotv_batch"]) for r in rows], st.median),
        "mean_se": safe([r["se_batch"] for r in rows], st.mean),
        "mean_se_after": safe([r["se_batch_after"] for r in rows], st.mean),
        "mean_VR_SE": safe([r["VR_SE"] for r in rows], st.mean),
        "median_VR_SE": safe([r["VR_SE"] for r in rows], st.median),
        "mean_SNR": safe([r["SNR"] for r in rows], st.mean),
        "mean_SNR_after": safe([r["SNR_after"] for r in rows], st.mean),
        "rows": rows,
    }
    out_json = root / args.out_json
    out_json.write_text(json.dumps(agg, indent=2))
    print("Wrote:\n ", out_csv, "\n ", out_json)

if __name__ == "__main__":
    main()
