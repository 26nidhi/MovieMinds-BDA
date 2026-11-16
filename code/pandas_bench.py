"""
Step 3C (Pandas side). Runs INSIDE a memory-capped Docker container.
Loads the first <fraction> of the ratings Parquet files (by bytes, files in sorted order)
into ONE Pandas DataFrame and runs the same 3 queries as spark_bench.py.
We help Pandas as much as we can: only 3 columns are read and rating is stored as int8.

Usage: python pandas_bench.py 0.1
Appends one line to pandas_results.csv. If the container runs out of memory it is killed by
Docker before it can write anything; the PowerShell loop then adds a KILLED_OOM line.
"""
import os, sys, time, resource
import pandas as pd
import pyarrow.parquet as pq

BASE = os.environ.get("BASE", "/data/ratings_parquet")
OUT = os.environ.get("RESULTS", "/data/pandas_results.csv")
frac = float(sys.argv[1])

files = []
for root, _, names in os.walk(BASE):
    for n in names:
        if n.endswith(".parquet"):
            p = os.path.join(root, n)
            files.append((os.path.relpath(p, BASE).replace("\\", "/"), os.path.getsize(p), p))
files.sort()
total_bytes = sum(f[1] for f in files)
chosen, acc = [], 0
for rel, size, p in files:
    if acc >= frac * total_bytes and chosen:
        break
    chosen.append(p); acc += size

def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

status, rows, load_s, qs = "ok", 0, 0.0, [0.0, 0.0, 0.0]
print(f"fraction {frac}: {len(chosen)} of {len(files)} files, {acc/2**20:.0f} MB on disk", flush=True)
try:
    t0 = time.time()
    parts = []
    for p in chosen:
        t = pq.read_table(p, columns=["movie_id", "user_id", "rating"]).to_pandas()
        t["rating"] = t["rating"].astype("int8")
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    del parts
    rows = len(df)
    load_s = time.time() - t0
    print(f"  loaded {rows:,} rows in {load_s:.1f}s, peak {peak_mb():.0f} MB", flush=True)
    t = time.time(); df.groupby("movie_id")["rating"].agg(["mean", "count"]); qs[0] = time.time() - t
    print(f"  q1 avg rating per movie: {qs[0]:.1f}s, peak {peak_mb():.0f} MB", flush=True)
    t = time.time(); df["user_id"].value_counts().head(10); qs[1] = time.time() - t
    print(f"  q2 top users: {qs[1]:.1f}s, peak {peak_mb():.0f} MB", flush=True)
    t = time.time(); df["rating"].value_counts(); qs[2] = time.time() - t
    print(f"  q3 histogram: {qs[2]:.1f}s, peak {peak_mb():.0f} MB", flush=True)
except MemoryError:
    status = "MemoryError"
    print("  MemoryError", flush=True)

total = load_s + sum(qs)
new = not os.path.exists(OUT)
with open(OUT, "a") as f:
    if new:
        f.write("fraction,files,rows,load_s,q1_s,q2_s,q3_s,total_s,peak_mb,status\n")
    f.write(f"{frac},{len(chosen)},{rows},{load_s:.1f},{qs[0]:.1f},{qs[1]:.1f},{qs[2]:.1f},{total:.1f},{peak_mb():.0f},{status}\n")
print(f"  TOTAL (load + 3 queries): {total:.1f}s  status={status}")
