"""
Step 3C (Spark side). Same data subsets, same 3 queries as pandas_bench.py.
Each query reads the Parquet files itself (no caching), so Spark pays the I/O cost 3 times
while Pandas pays it once. That is deliberately unfavourable to Spark.

Usage: spark-submit [--total-executor-cores N] spark_bench.py [fractions]
       fractions: comma list, default 0.01,0.1,0.5,1.0
The number of cores used is in the saved result name, so you can run it with 1, 2 and 4 cores.
"""
import sys, time
from pyspark.sql import SparkSession, functions as F

HD = "hdfs://namenode:9000"
BASE = f"{HD}/netflix/parquet/ratings"
spark = SparkSession.builder.appName("spark-bench").config("spark.hadoop.dfs.replication", "2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
cores = sc.getConf().get("spark.cores.max", "all")
fracs = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0.01,0.1,0.5,1.0").split(",")]

# list the Parquet files exactly like pandas_bench.py does (sorted by relative path, cumulative bytes)
jp = sc._jvm.org.apache.hadoop.fs.Path(BASE)
fs = jp.getFileSystem(sc._jsc.hadoopConfiguration())
base_path = jp.toUri().getPath()
it = fs.listFiles(jp, True)
files = []
while it.hasNext():
    st = it.next()
    full = st.getPath().toString()
    if full.endswith(".parquet"):
        files.append((st.getPath().toUri().getPath()[len(base_path):].lstrip("/"), st.getLen(), full))
files.sort()
total_bytes = sum(f[1] for f in files)

def choose(frac):
    chosen, acc = [], 0
    for rel, size, full in files:
        if acc >= frac * total_bytes and chosen:
            break
        chosen.append(full); acc += size
    return chosen, acc

def timed(df):
    t = time.time(); df.write.format("noop").mode("overwrite").save(); return time.time() - t

rows_out = []
for frac in fracs:
    paths, acc = choose(frac)
    df = spark.read.parquet(*paths).select("movie_id", "user_id", "rating")
    n = df.count()
    q1 = timed(df.groupBy("movie_id").agg(F.avg("rating"), F.count("*")))
    q2 = timed(df.groupBy("user_id").count().orderBy(F.desc("count")).limit(10))
    q3 = timed(df.groupBy("rating").count())
    tot = q1 + q2 + q3
    rows_out.append((frac, len(paths), n, round(q1, 1), round(q2, 1), round(q3, 1), round(tot, 1), cores))
    print(f"[spark cores={cores}] fraction {frac}: {len(paths)} files, {n:,} rows | q1 {q1:.1f}s q2 {q2:.1f}s q3 {q3:.1f}s | total {tot:.1f}s", flush=True)

name = f"spark_bench_{cores}cores"
spark.createDataFrame(rows_out, "fraction double, files int, rows long, q1_s double, q2_s double, q3_s double, total_s double, cores string") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/{name}")
print(f"\n========== SPARK BENCH (cores={cores}) ==========")
for r in rows_out:
    print(f"fraction {r[0]:<5} rows {r[2]:>12,}  total {r[6]:>7}s")
print("=============================================")
spark.stop()
