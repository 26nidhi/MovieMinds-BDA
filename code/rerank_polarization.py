"""
Step 5B: Polarization-aware re-ranking (Novelty 3).

A movie is "polarizing" when people disagree about it: its ratings have a large spread (std dev).
The re-ranker adjusts each candidate's predicted rating by its polarization:

        score(u, m) = predicted_rating(u, m) - lambda * z(m)

  z(m) = standardised rating spread of movie m (computed from TRAIN+VAL ratings only, so no future
         information leaks into the test evaluation)
  lambda > 0 : "safe picks"  (pushes polarizing movies down)
  lambda < 0 : "risky picks" (pushes them up)
  lambda = 0 : the original ranking

The lambda values are fixed in advance. This is a trade-off curve for a user-facing toggle, NOT a tuned
setting: we do not pick the best lambda on the test set. Each row compares with lambda = 0 using
user-clustered bootstrap intervals.

Metrics per list of 10: P@10 and NDCG@10 (hit = rated >= 4 in the test period), distinct movies
recommended, mean polarization of the list, and "bad rate" = share of recommended movies the user
actually rated in the test period that got <= 2 stars.

Needs /netflix/model/rank_scores (from hybrid_eval.py).
Usage: spark-submit ... rerank_polarization.py [resamples=2000]
"""
import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

HD = "hdfs://namenode:9000"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TOPK = 10
LAMBDAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
BASES = [("ALS, popular pool", "als"), ("Hybrid 50/50, popular pool", "hybrid")]
spark = SparkSession.builder.appName("rerank-polarization").config("spark.sql.shuffle.partitions", "64").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
rd = lambda p: spark.read.parquet(f"{HD}{p}")
rng = np.random.default_rng(2024)

S = rd("/netflix/model/rank_scores").filter(F.col("in_pool") == 1)
sel = lambda df: df.select("user_id", "movie_id", F.col("rating").cast("double").alias("rating"))
full = sel(rd("/netflix/model/train")).unionByName(sel(rd("/netflix/model/val")))
test = sel(rd("/netflix/model/test")).withColumnRenamed("rating", "tr")

pol = full.groupBy("movie_id").agg(F.stddev("rating").alias("pol"))
pool_pol = pol.join(S.select("movie_id").distinct(), "movie_id")
mu_p, sd_p = pool_pol.agg(F.avg("pol"), F.stddev("pol")).first()
pol = pol.withColumn("pol", F.coalesce("pol", F.lit(mu_p))).withColumn("z", (F.col("pol") - F.lit(mu_p)) / F.lit(sd_p))
S = S.join(pol, "movie_id", "left").withColumn("z", F.coalesce("z", F.lit(0.0))).withColumn("pol", F.coalesce("pol", F.lit(mu_p))).cache()
users = sorted(r["user_id"] for r in S.select("user_id").distinct().collect()); pos = {u: i for i, u in enumerate(users)}
n_pool = S.select("movie_id").distinct().count()
print(f"[pol] {len(users)} users, pool of {n_pool} movies, polarization (std dev of ratings): mean {mu_p:.3f}, sd across pool {sd_p:.3f}", flush=True)

def evaluate(base, lam):
    d = S.filter(F.col(base).isNotNull()).withColumn("score", F.col(base) - F.lit(lam) * F.col("z"))
    top = (d.withColumn("rk", F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc("score"), F.asc("movie_id"))))
             .filter(F.col("rk") <= TOPK).select("user_id", "movie_id", "rk", "R", "pol"))
    pu = (top.join(test, ["user_id", "movie_id"], "left").groupBy("user_id", "R")
             .agg(F.sum(F.when(F.col("tr") >= 4, 1).otherwise(0)).alias("hits"),
                  F.sum(F.when(F.col("tr") >= 4, 1 / F.log2(F.col("rk") + 1)).otherwise(0.0)).alias("dcg"),
                  F.count("tr").alias("rated"), F.sum(F.when(F.col("tr") <= 2, 1).otherwise(0)).alias("bad"),
                  F.avg("pol").alias("avg_pol"))
             .withColumn("idcg", F.expr(f"aggregate(sequence(1, least({TOPK}, R)), cast(0 as double), (acc, i) -> acc + 1 / log2(i + 1))")))
    arr = np.zeros((len(users), 2)); rated = bad = 0; pol_sum = 0.0
    for r in pu.collect():
        arr[pos[r["user_id"]]] = (r["hits"] / TOPK, r["dcg"] / r["idcg"])
        rated += r["rated"]; bad += r["bad"]; pol_sum += r["avg_pol"]
    distinct = top.select("movie_id").distinct().count()
    return arr, distinct, pol_sum / len(users), (bad / rated if rated else float("nan")), rated

def paired_ci(diff):
    U = len(diff); vals = []; p = np.full(U, 1.0 / U)
    while len(vals) < B:
        counts = rng.multinomial(U, p, size=min(100, B - len(vals))); vals.extend(counts @ diff / U)
    return diff.mean(), *np.percentile(vals, [2.5, 97.5])

out = []
print("\n========== STEP 5B RESULTS (test period, 95% CIs resample users) ==========")
for bname, base in BASES:
    res = {lam: evaluate(base, lam) for lam in LAMBDAS}
    base_arr = res[0.0][0]
    print(f"\nbase ranker: {bname}   (lambda>0 safe, lambda<0 risky)")
    print(f"{'lambda':>7}{'P@10':>8}{'NDCG@10':>9}{'distinct':>10}{'mean pol':>10}{'bad rate':>10}   change in P@10 vs lambda=0 (95% CI)")
    for lam in LAMBDAS:
        arr, distinct, mpol, bad_rate, rated = res[lam]
        p, nd = arr[:, 0].mean(), arr[:, 1].mean()
        if lam == 0.0:
            ci_txt = "(reference)"; dP = loP = hiP = None
        else:
            dP, loP, hiP = paired_ci(arr[:, 0] - base_arr[:, 0])
            ci_txt = f"{dP:+.4f} [{loP:+.4f}, {hiP:+.4f}] {'excludes 0' if loP > 0 or hiP < 0 else 'includes 0'}"
        print(f"{lam:>7.2f}{p:>8.4f}{nd:>9.4f}{distinct:>10}{mpol:>10.3f}{bad_rate:>10.3f}   {ci_txt}")
        out.append((bname, float(lam), float(p), float(nd), int(distinct), float(mpol), float(bad_rate), int(rated),
                    None if dP is None else float(dP), None if dP is None else float(loP), None if dP is None else float(hiP)))
print("=" * 72)
print("bad rate = share of the recommended movies the user later rated that got <= 2 stars (only counts movies that were rated)")
spark.createDataFrame(out, "base string, lam double, p10 double, ndcg10 double, distinct_movies long, mean_polarization double, "
                           "bad_rate double, rated_recs long, d_p10 double, d_p10_lo double, d_p10_hi double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/rerank_polarization")
spark.stop()
