"""
Step 3B: Optimisation lab. Each experiment changes ONE thing and times the same query.

  1. Caching      : no cache vs MEMORY_ONLY vs MEMORY_AND_DISK (full data and a 20% sample)
  2. Join strategy: broadcast vs sort-merge vs shuffle-hash (ratings JOIN movie table)
  3. AQE          : adaptive query execution off/on, 200 vs 2000 shuffle partitions
  4. Skew check   : is there real key skew here? (measured, not assumed)
  5. Bucketing    : (optional, add the word  bucketing  after the script name) pre-bucketed join

Every timing is the median of 3 runs. Results go to /netflix/results/optimization_lab and
the physical plans to /netflix/results/optimization_plans (for the dashboard).

Run:  spark-submit ... /code/optimization_lab.py            (1-4)
      spark-submit ... /code/optimization_lab.py bucketing  (1-5)
"""
import sys, time, statistics
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

HD = "hdfs://namenode:9000"
spark = (SparkSession.builder.appName("optimization-lab")
         .config("spark.hadoop.dfs.replication", "2")
         .config("spark.sql.warehouse.dir", f"{HD}/netflix/warehouse")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("ERROR")

ratings = spark.read.parquet(f"{HD}/netflix/parquet/ratings")
ratings.createOrReplaceTempView("ratings")
raw = spark.read.option("encoding", "ISO-8859-1").option("sep", "\u0001").csv(f"{HD}/netflix/meta/movie_titles.csv")
(raw.select(F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 1).cast("int").alias("movie_id"),
            F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 2).alias("year_s"))
    .withColumn("release_year", F.when(F.col("year_s") != "NULL", F.col("year_s").cast("int")))
    .drop("year_s").createOrReplaceTempView("titles"))

results, plans = [], []          # results: (experiment, variant, first_s, median_s, detail)

def timed(df):
    t = time.time(); df.write.format("noop").mode("overwrite").save(); return time.time() - t

def plan_of(df):
    return df._jdf.queryExecution().executedPlan().toString()

def record(exp, variant, first, runs, detail=""):
    med = round(statistics.median(runs), 1)
    results.append((exp, variant, round(first, 1), med, detail))
    print(f"[{exp}] {variant:34} first={first:6.1f}s  median(3)={med:6.1f}s  {detail}", flush=True)

def cache_info():
    parts = cached = mem = disk = 0
    for i in sc._jsc.sc().getRDDStorageInfo():
        parts += i.numPartitions(); cached += i.numCachedPartitions()
        mem += i.memSize(); disk += i.diskSize()
    return parts, cached, mem, disk

# ===================== 1. CACHING =====================
spark.conf.set("spark.sql.shuffle.partitions", "64")
def q_cache(d): return d.groupBy("movie_id").agg(F.avg("rating").alias("a"), F.count("*").alias("n"))
for scen, frac in (("full data", 1.0), ("20% sample", 0.2)):
    base = ratings.select("movie_id", "user_id", "rating", "year")
    if frac < 1.0:
        base = base.sample(False, frac, seed=7)
    for label, level in (("no cache", None), ("MEMORY_ONLY", StorageLevel.MEMORY_ONLY),
                         ("MEMORY_AND_DISK", StorageLevel.MEMORY_AND_DISK)):
        d = base
        if level is not None:
            d = base.persist(level)
        first = timed(q_cache(d))                      # for cached runs this includes building the cache
        detail = ""
        if level is not None:
            p, c, m, k = cache_info()
            detail = f"cached {c}/{p} partitions, mem {m/2**20:.0f}MB, disk {k/2**20:.0f}MB"
        runs = [timed(q_cache(d)) for _ in range(3)]
        record("cache " + scen, label, first, runs, detail)
        if level is not None:
            base.unpersist(True)

# ===================== 2. JOIN STRATEGY =====================
spark.conf.set("spark.sql.adaptive.enabled", "false")      # so Spark cannot change the strategy on its own
JQ = """SELECT /*+ {hint}(m) */ (m.release_year DIV 10) * 10 AS decade, COUNT(*) AS n, AVG(r.rating) AS avg_rating
        FROM ratings r JOIN titles m ON r.movie_id = m.movie_id GROUP BY 1"""
for variant, hint, want in (("broadcast hash join", "BROADCAST", "BroadcastHashJoin"),
                            ("sort-merge join", "MERGE", "SortMergeJoin"),
                            ("shuffled hash join", "SHUFFLE_HASH", "ShuffledHashJoin")):
    df = spark.sql(JQ.format(hint=hint))
    pl = plan_of(df)
    ok = "plan confirms " + want if want in pl else "WARNING: plan does not show " + want
    plans.append(("join: " + variant, pl))
    first = timed(df)
    runs = [timed(spark.sql(JQ.format(hint=hint))) for _ in range(3)]
    record("join strategy", variant, first, runs, ok)

# ===================== 3. AQE =====================
AQ = "SELECT user_id, COUNT(*) AS n, AVG(rating) AS a FROM ratings GROUP BY user_id"
for variant, parts, aqe in (("AQE off, 200 partitions", 200, "false"),
                            ("AQE off, 2000 partitions", 2000, "false"),
                            ("AQE on,  200 partitions", 200, "true"),
                            ("AQE on,  2000 partitions", 2000, "true")):
    spark.conf.set("spark.sql.adaptive.enabled", aqe)
    spark.conf.set("spark.sql.shuffle.partitions", str(parts))
    df = spark.sql(AQ)
    plans.append(("aqe: " + variant, plan_of(df)))
    first = timed(df)
    runs = [timed(spark.sql(AQ)) for _ in range(3)]
    record("aqe", variant, first, runs)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "64")

# ===================== 4. SKEW CHECK =====================
ms = spark.read.parquet(f"{HD}/netflix/results/movie_stats")
us = spark.read.parquet(f"{HD}/netflix/results/user_stats")
total = ratings.count()
avg_part = total / 64.0
skew_rows = []
for key, df in (("movie_id", ms), ("user_id", us)):
    top = df.agg(F.max("n_ratings")).first()[0]
    skew_rows.append((key, float(top), round(top / avg_part, 4)))
    print(f"[skew] busiest {key} has {top} ratings = {top/avg_part:.3f} x the size of an average 64-way shuffle partition", flush=True)
print("       a key must be several times bigger than an average partition before skew handling pays off", flush=True)
spark.createDataFrame(skew_rows, "key string, max_key_ratings double, ratio_to_avg_partition double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/skew_check")

# ===================== 5. BUCKETING (optional) =====================
if "bucketing" in sys.argv:
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")   # force a shuffle join so bucketing matters
    BQ = "SELECT COUNT(*) AS n, AVG(r.rating - s.avg_rating) AS dev FROM {r} r JOIN {s} s ON r.movie_id = s.movie_id"
    # baseline: plain Parquet files, both sides must be shuffled
    ms.select("movie_id", "avg_rating").createOrReplaceTempView("stats_plain")
    ratings.select("movie_id", "rating").createOrReplaceTempView("ratings_plain")
    df = spark.sql(BQ.format(r="ratings_plain", s="stats_plain"))
    pl = plan_of(df); plans.append(("bucketing: not bucketed", pl))
    first = timed(df); runs = [timed(spark.sql(BQ.format(r="ratings_plain", s="stats_plain"))) for _ in range(3)]
    record("bucketing", "not bucketed (shuffle join)", first, runs, f"Exchange nodes in plan: {pl.count('Exchange')}")
    # one-time cost: write both tables bucketed by movie_id
    t0 = time.time()
    (ratings.select("movie_id", "rating").write.mode("overwrite").format("parquet")
        .bucketBy(64, "movie_id").sortBy("movie_id").saveAsTable("ratings_b"))
    (ms.select("movie_id", "avg_rating").write.mode("overwrite").format("parquet")
        .bucketBy(64, "movie_id").sortBy("movie_id").saveAsTable("stats_b"))
    t_write = time.time() - t0
    print(f"[bucketing] one-time cost of writing the bucketed tables: {t_write:.0f}s", flush=True)
    df = spark.sql(BQ.format(r="ratings_b", s="stats_b"))
    pl = plan_of(df); plans.append(("bucketing: bucketed", pl))
    first = timed(df); runs = [timed(spark.sql(BQ.format(r="ratings_b", s="stats_b"))) for _ in range(3)]
    record("bucketing", "bucketed by movie_id (no shuffle)", first, runs,
           f"Exchange nodes in plan: {pl.count('Exchange')}; one-time write {t_write:.0f}s")

# ===================== SAVE + SUMMARY =====================
spark.createDataFrame(results, "experiment string, variant string, first_run_s double, median_s double, detail string") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/optimization_lab")
spark.createDataFrame(plans, "name string, plan string").coalesce(1) \
     .write.mode("overwrite").parquet(f"{HD}/netflix/results/optimization_plans")
print("\n========== OPTIMISATION LAB (median of 3 runs) ==========")
for e, v, f, m, d in results:
    print(f"{e:18}{v:36}first {f:>7}s   median {m:>7}s   {d}")
print("==========================================================")
spark.stop()
