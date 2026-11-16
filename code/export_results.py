"""
Export every small result table in HDFS (/netflix/results/*) to CSV files, so they can be
looked at, shared, and used by the dashboard.

Output folder (inside the container): /code/export_csv
That is your code folder on the laptop, so a new folder  export_csv  appears next to the .py files.

Big tables are cut to the first MAX_ROWS rows (movie_stats, user_stats, rank_scores etc.).
Long text cells (physical plans) are cut to MAX_CELL characters.

Run:
docker exec -e HOME=/tmp spark-master /opt/bitnami/spark/bin/spark-submit --master spark://spark-master:7077 --conf spark.jars.ivy=/tmp/.ivy2 --driver-memory 1g /code/export_results.py
"""
import os, csv
from pyspark.sql import SparkSession

HD = "hdfs://namenode:9000"
OUT = "/code/export_csv"
MAX_ROWS = 20000
MAX_CELL = 1500

spark = SparkSession.builder.appName("export-results").config("spark.hadoop.dfs.replication", "2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext
os.makedirs(OUT, exist_ok=True)


def clean(v):
    if v is None:
        return ""
    s = str(v)
    return s if len(s) <= MAX_CELL else s[:MAX_CELL] + " ...[cut]"


def save(name, df):
    total = df.count()
    rows = df.limit(MAX_ROWS).collect()
    with open(f"{OUT}/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(df.columns)
        for r in rows:
            w.writerow([clean(v) for v in r])
    print(f"[export] {name}: {len(rows)} of {total} rows", flush=True)


# 1. every folder under /netflix/results
base = sc._jvm.org.apache.hadoop.fs.Path(f"{HD}/netflix/results")
fs = base.getFileSystem(sc._jsc.hadoopConfiguration())
names = sorted(s.getPath().getName() for s in fs.listStatus(base) if s.isDirectory())
print(f"[export] found {len(names)} result tables: {names}", flush=True)
for n in names:
    try:
        save(n, spark.read.parquet(f"{HD}/netflix/results/{n}"))
    except Exception as e:
        print(f"[export] FAILED {n}: {str(e)[:200]}", flush=True)

# 2. matching benchmark (saved as JSON, not Parquet)
for n in ("benchmark", "benchmark_full"):
    try:
        save(f"matching_{n}", spark.read.json(f"{HD}/netflix/meta/{n}"))
    except Exception as e:
        print(f"[export] skipped matching_{n}: {str(e)[:120]}", flush=True)

# 3. TMDB matching summary: how many movies matched by each method
try:
    m = spark.read.parquet(f"{HD}/netflix/meta/matches")
    save("matching_summary", m.groupBy("method").count())
except Exception as e:
    print(f"[export] skipped matching_summary: {str(e)[:120]}", flush=True)

print(f"\n[export] done. Files are in {OUT} (the export_csv folder inside your code folder).", flush=True)
spark.stop()
