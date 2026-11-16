"""
Step 4E: user-clustered bootstrap confidence intervals.

Why: ratings from the same user are correlated, so intervals that treat every rating as independent
(as in hybrid_eval.py) are too narrow. Here we resample USERS with replacement and recompute each
statistic, which respects that correlation.

PART A  rating error: for each pair of models, the difference in test RMSE (and MSE).
PART B  ranking: for each pair of recommenders, the difference in per-user P@10 and NDCG@10.

Needs the outputs of als_train.py, content_model.py and hybrid_eval.py (it re-uses /netflix/model/rank_scores).
Usage: spark-submit ... bootstrap_ci.py [resamples=2000]
"""
import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

HD = "hdfs://namenode:9000"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TOPK = 10
spark = SparkSession.builder.appName("bootstrap-ci").config("spark.sql.shuffle.partitions", "64").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))
rng = np.random.default_rng(2024)

def boot(arr, stat, chunk=100):
    """arr: users x columns of per-user sums. stat(sums) -> value. Returns (estimate, lo, hi)."""
    U = arr.shape[0]; est = stat(arr.sum(axis=0)); vals = []
    p = np.full(U, 1.0 / U)
    for _ in range(0, B, chunk):
        counts = rng.multinomial(U, p, size=min(chunk, B - len(vals)))      # how often each user is drawn
        sums = counts @ arr
        vals.extend(stat(s) for s in sums)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, lo, hi

# ======================= PART A: rating error =======================
pa = spark.read.parquet(f"{HD}/netflix/model/pred_als_test").select("user_id", "movie_id", "rating", "pred_als_filled", "pred_bias")
pc = spark.read.parquet(f"{HD}/netflix/model/pred_content_test").select("user_id", "movie_id", "pred_content")
j = pa.join(pc, ["user_id", "movie_id"]).withColumn("pc", F.coalesce("pred_content", "pred_bias"))
models = {"bias": F.col("pred_bias"), "ALS": F.col("pred_als_filled"), "content": F.col("pc"),
          "blend 0.25": 0.25 * F.col("pred_als_filled") + 0.75 * F.col("pc"),
          "blend 0.50": 0.5 * F.col("pred_als_filled") + 0.5 * F.col("pc"),
          "blend 0.75": 0.75 * F.col("pred_als_filled") + 0.25 * F.col("pc")}
names = list(models)
per_user = j.groupBy("user_id").agg(F.count("*").alias("n"), *[F.sum((clip(e) - F.col("rating")) ** 2).alias(f"se{i}") for i, e in enumerate(models.values())])
rows = per_user.collect()
arrA = np.array([[r["n"]] + [r[f"se{i}"] for i in range(len(names))] for r in rows], dtype=float)
col = {n: i + 1 for i, n in enumerate(names)}
rmse = lambda s, k: np.sqrt(s[col[k]] / s[0])
pairsA = [("ALS", "bias"), ("content", "bias"), ("blend 0.50", "bias"), ("blend 0.50", "ALS"), ("blend 0.50", "content")]
print(f"\n[A] {int(arrA[:,0].sum()):,} test pairs from {len(rows):,} users, {B} user-level resamples")
resA = []
for a, b in pairsA:
    est, lo, hi = boot(arrA, lambda s, a=a, b=b: rmse(s, a) - rmse(s, b))
    resA.append((f"{a} minus {b}", float(est), float(lo), float(hi)))
    print(f"[A] RMSE({a}) - RMSE({b}) = {est:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}", flush=True)

# ======================= PART B: ranking =======================
S = spark.read.parquet(f"{HD}/netflix/model/rank_scores")
test = spark.read.parquet(f"{HD}/netflix/model/test").select("user_id", "movie_id", "rating")
uid = S.select("user_id").distinct()
relevant = test.filter(F.col("rating") >= 4).join(uid, "user_id").select("user_id", "movie_id").withColumn("hit", F.lit(1))
methods = [("Popularity", "n_movie", False), ("Best-rated, pool", "b_m", True), ("ALS, all", "als", False),
           ("ALS, pool", "als", True), ("Content, pool", "content", True), ("Hybrid, pool", "hybrid", True)]
users = sorted(r["user_id"] for r in uid.collect()); pos = {u: i for i, u in enumerate(users)}
metric = {}
for name, c, use_pool in methods:
    d = S.filter(F.col(c).isNotNull())
    if use_pool:
        d = d.filter(F.col("in_pool") == 1)
    top = (d.withColumn("rk", F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc(c), F.asc("movie_id"))))
             .filter(F.col("rk") <= TOPK).select("user_id", "movie_id", "rk", "R"))
    pu = (top.join(relevant, ["user_id", "movie_id"], "left").groupBy("user_id", "R")
             .agg(F.sum(F.coalesce("hit", F.lit(0))).alias("hits"), F.sum(F.coalesce("hit", F.lit(0)) / F.log2(F.col("rk") + 1)).alias("dcg"))
             .withColumn("idcg", F.expr(f"aggregate(sequence(1, least({TOPK}, R)), cast(0 as double), (acc, i) -> acc + 1 / log2(i + 1))")))
    arr = np.zeros((len(users), 2))
    for r in pu.collect():
        arr[pos[r["user_id"]]] = (r["hits"] / TOPK, r["dcg"] / r["idcg"])
    metric[name] = arr
pairsB = [("ALS, pool", "Popularity"), ("ALS, pool", "Best-rated, pool"), ("Hybrid, pool", "ALS, pool"),
          ("Content, pool", "Best-rated, pool"), ("Hybrid, pool", "Popularity")]
print(f"\n[B] {len(users):,} users, {B} user-level resamples (paired differences per user)")
resB = []
for a, b in pairsB:
    arr = np.hstack([metric[a], metric[b]])                     # columns: P_a, NDCG_a, P_b, NDCG_b
    for k, nm in ((0, "P@10"), (1, "NDCG@10")):
        est, lo, hi = boot(np.hstack([arr, np.ones((len(users), 1))]), lambda s, k=k: (s[k] - s[k + 2]) / s[4])
        resB.append((f"{nm}: {a} minus {b}", float(est), float(lo), float(hi)))
        print(f"[B] {nm:8} {a} - {b} = {est:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}", flush=True)

spark.createDataFrame(resA + resB, "comparison string, estimate double, ci_low double, ci_high double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/bootstrap_ci")
spark.stop()
