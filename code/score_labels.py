r"""
Score your hand-labelled sample: precision of the Netflix <-> TMDB matches.

Run from the netflix-bda folder:
    python code\score_labels.py                 (just print the results)
    python code\score_labels.py --write         (also store them in dashboard\facts.json for the dashboard)

Reads data\label_sample.csv. Only 'yes' / 'no' (or y / n, any capitalisation) count; blank or other
values are reported and left out. Confidence intervals are Wilson 95% intervals.
"""
import json, math, sys
import pandas as pd

CSV = "data/label_sample.csv"
FACTS = "dashboard/facts.json"

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def read(path):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            err = e
    raise err

if __name__ == "__main__":
    facts_path = FACTS
    if "--facts" in sys.argv:
        facts_path = sys.argv[sys.argv.index("--facts") + 1]
    df = read(CSV)
    df.columns = [c.strip() for c in df.columns]
    if "correct_yes_no" not in df.columns:
        sys.exit(f"Column 'correct_yes_no' not found. Columns are: {list(df.columns)}")
    lab = df["correct_yes_no"].astype(str).str.strip().str.lower().replace({"nan": ""})
    lab = lab.replace({"y": "yes", "n": "no"})
    ok = lab.isin(["yes", "no"])
    print(f"rows in file: {len(df)}   labelled yes/no: {int(ok.sum())}   blank or unrecognised: {int((~ok).sum())}")
    if (~ok).any():
        bad = df[~ok]
        print("  unlabelled rows (movie_id, nf_title):", list(zip(bad.get("movie_id", []), bad.get("nf_title", [])))[:10])
    d = df[ok].assign(correct=lab[ok] == "yes")
    n, k = len(d), int(d["correct"].sum())
    lo, hi = wilson(k, n)
    print(f"\nOVERALL precision: {k}/{n} = {100*k/n:.1f}%   (95% CI {100*lo:.1f}% to {100*hi:.1f}%)")
    by = {}
    if "method" in d.columns:
        print("\nby method:")
        for m, g in d.groupby("method"):
            kk, nn = int(g["correct"].sum()), len(g)
            l, h = wilson(kk, nn)
            by[str(m)] = {"n": nn, "yes": kk}
            print(f"  {m:6} {kk}/{nn} = {100*kk/nn:.1f}%   (95% CI {100*l:.1f}% to {100*h:.1f}%)")
        if any(v["n"] < 20 for v in by.values()):
            print("  note: a method with fewer than 20 labelled rows has a very wide interval; its precision is not reliable.")
    wrong = d[~d["correct"]]
    cols = [c for c in ["movie_id", "nf_title", "nf_year", "tm_title", "tm_year", "method", "dist"] if c in wrong.columns]
    if len(wrong):
        print("\nmatches you marked 'no':")
        print(wrong[cols].to_string(index=False))
    if "--write" in sys.argv:
        f = json.load(open(facts_path, encoding="utf-8"))
        f["matching"]["precision"] = round(k / n, 4)
        f["matching"]["precision_n"] = n
        f["matching"]["precision_ci"] = [round(lo, 4), round(hi, 4)]
        f["matching"]["precision_by_method"] = by
        json.dump(f, open(facts_path, "w", encoding="utf-8"), indent=2)
        print(f"\nwritten to {facts_path}")
