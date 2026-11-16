"""
Step 4D: (1) Top-10 ranking metrics for 6 recommenders, (2) rating-prediction hybrid with confidence intervals.

PART 1  Ranking (Precision@10, Recall@10, NDCG@10, coverage)
  Users: a random sample of test-period users who have >= 5 earlier ratings and >= 1 test rating >= 4.
  Relevant item = a movie the user rated >= 4 in the TEST period (the future). Items the user already
  rated before the test period are never recommended. Every method ranks the same candidate movies.
    Popularity            : most-rated movies in train+val (non-personalised)
    Best-rated, pool      : damped movie mean, restricted to the popular pool (non-personalised)
    ALS, all movies       : predicted rating from the saved ALS model
    ALS, popular pool     : same, restricted to the POOL most-rated movies
    Content kNN, pool     : bias + content-neighbour residuals, restricted to the pool
    Hybrid 50/50, pool    : average of the ALS and content-kNN predicted ratings (weights fixed, not tuned)
  Why a pool: ALS is trained to predict the rating of movies a user DOES rate, so it can rank obscure,
  rarely rated movies first; those are rarely rated in the future, so they are rarely 'hits'.

PART 2  Rating RMSE of blends  alpha*ALS + (1-alpha)*content  vs the bias baseline, with a 95% CI on the
  paired difference in MSE (treats test pairs as independent; ratings of one user are correlated, so the true
  interval is wider than printed).

Usage: spark-submit ... hybrid_eval.py [n_users=1000] [pool=1000]
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALSModel

HD = "hdfs://namenode:9000"
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TOPK, LAMBDA = 10, 3.0
spark = (SparkSession.builder.appName("hybrid-eval")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))

train = spark.read.parquet(f"{HD}/netflix/model/train").select("user_id", "movie_id", F.col("rating").cast("double"))
val = spark.read.parquet(f"{HD}/netflix/model/val").select("user_id", "movie_id", F.col("rating").cast("double"))
test = spark.read.parquet(f"{HD}/netflix/model/test").select("user_id", "movie_id", F.col("rating").cast("double")).cache()
full = train.unionByName(val).cache()
mu = full.agg(F.avg("rating")).first()[0]
bm = (full.groupBy("movie_id").agg(F.sum(F.col("rating") - mu).alias("s"), F.count("*").alias("n"))
          .select("movie_id", (F.col("s") / (F.col("n") + 10.0)).alias("b_m"), F.col("n").alias("n_movie"))).cache()
bu = (full.join(bm, "movie_id").withColumn("res", F.col("rating") - mu - F.col("b_m"))
          .groupBy("user_id").agg(F.sum("res").alias("s"), F.count("*").alias("n"))
          .select("user_id", (F.col("s") / (F.col("n") + 25.0)).alias("b_u"), F.col("n").alias("n_hist"))).cache()
nb = spark.read.parquet(f"{HD}/netflix/model/content_neighbors")
model = ALSModel.load(f"{HD}/netflix/model/als_best")

# ============================ PART 1: ranking ============================
rel_all = test.filter(F.col("rating") >= 4).select("user_id", "movie_id")
R = rel_all.groupBy("user_id").agg(F.count("*").alias("R"))
users = (R.join(bu.filter(F.col("n_hist") >= 5).select("user_id", "b_u"), "user_id")
          .orderBy(F.rand(11)).limit(N_USERS)).cache()
n_users = users.count()
cands = bm.select("movie_id", "b_m", "n_movie").cache()                     # every movie seen in train+val
pool_ids = cands.orderBy(F.desc("n_movie")).limit(POOL).select("movie_id", F.lit(1).alias("in_pool"))
seen = full.join(users.select("user_id"), "user_id").select("user_id", "movie_id").withColumn("seen", F.lit(1))
relevant = rel_all.join(users.select("user_id"), "user_id").withColumn("hit", F.lit(1))

uf = model.userFactors.join(users, model.userFactors.id == users.user_id).select("user_id", F.col("features").alias("uvec"), "b_u", "R")
itf = model.itemFactors.select(F.col("id").alias("movie_id"), F.col("features").alias("ivec"))
resid = (full.join(users.select("user_id"), "user_id").join(bm, "movie_id").join(bu.select("user_id", "b_u"), "user_id")
             .select("user_id", F.col("movie_id").alias("neighbor"), (F.col("rating") - mu - F.col("b_m") - F.col("b_u")).alias("res")))
contrib = (resid.join(nb, "neighbor").groupBy("user_id", F.col("movie_id"))
                .agg(F.sum(F.col("sim") * F.col("res")).alias("num"), F.sum("sim").alias("den")))

S = (uf.crossJoin(cands).join(itf, "movie_id", "left").join(pool_ids, "movie_id", "left")
       .join(seen, ["user_id", "movie_id"], "left").filter(F.col("seen").isNull())
       .join(contrib, ["user_id", "movie_id"], "left")
       .withColumn("als", clip(F.when(F.col("ivec").isNotNull(),
            F.expr("aggregate(zip_with(uvec, ivec, (x, y) -> x * y), cast(0 as double), (acc, v) -> acc + v)"))))
       .withColumn("bias_pred", clip(F.lit(mu) + F.col("b_m") + F.col("b_u")))
       .withColumn("content", clip(F.col("bias_pred") + F.coalesce(F.col("num") / (F.col("den") + F.lit(LAMBDA)), F.lit(0.0))))
       .withColumn("hybrid", F.when(F.col("als").isNotNull(), 0.5 * F.col("als") + 0.5 * F.col("content")).otherwise(F.col("content")))
       .withColumn("in_pool", F.coalesce("in_pool", F.lit(0)))
       .select("user_id", "movie_id", "R", "n_movie", "b_m", "in_pool", "als", "content", "hybrid"))
S.write.mode("overwrite").parquet(f"{HD}/netflix/model/rank_scores")
S = spark.read.parquet(f"{HD}/netflix/model/rank_scores")

methods = [("Popularity (most rated)", "n_movie", False), ("Best-rated (damped mean), pool", "b_m", True),
           ("ALS, all movies", "als", False), ("ALS, popular pool", "als", True),
           ("Content kNN, popular pool", "content", True), ("Hybrid 50/50, popular pool", "hybrid", True)]
n_pool = pool_ids.count()
rank_rows = []
for name, col, use_pool in methods:
    d = S.filter(F.col(col).isNotNull())
    if use_pool:
        d = d.filter(F.col("in_pool") == 1)
    top = (d.withColumn("rk", F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc(col), F.asc("movie_id"))))
             .filter(F.col("rk") <= TOPK).select("user_id", "movie_id", "rk", "R").cache())
    per_user = (top.join(relevant, ["user_id", "movie_id"], "left")
                   .groupBy("user_id", "R")
                   .agg(F.sum(F.coalesce("hit", F.lit(0))).alias("hits"),
                        F.sum(F.coalesce("hit", F.lit(0)) / (F.log2(F.col("rk") + 1))).alias("dcg"))
                   .withColumn("idcg", F.expr(f"aggregate(sequence(1, least({TOPK}, R)), cast(0 as double), (acc, i) -> acc + 1 / log2(i + 1))")))
    agg = per_user.agg(F.avg(F.col("hits") / TOPK).alias("p"), F.avg(F.col("hits") / F.col("R")).alias("rec"),
                       F.avg(F.col("dcg") / F.col("idcg")).alias("ndcg"), F.count("*").alias("n")).first()
    cov = top.select("movie_id").distinct().count() / (n_pool if use_pool else cands.count())
    rank_rows.append((name, float(agg["p"]), float(agg["rec"]), float(agg["ndcg"]), float(cov), int(agg["n"])))
    print(f"[ranking] {name:34} P@{TOPK}={agg['p']:.4f}  R@{TOPK}={agg['rec']:.4f}  NDCG@{TOPK}={agg['ndcg']:.4f}  coverage={cov:.3f}", flush=True)
    top.unpersist()
spark.createDataFrame(rank_rows, "method string, precision_at_10 double, recall_at_10 double, ndcg_at_10 double, coverage double, users long") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/ranking_metrics")

# ============================ PART 2: rating blends with CI ============================
pa = spark.read.parquet(f"{HD}/netflix/model/pred_als_test").select("user_id", "movie_id", "rating", "pred_als_filled", "pred_bias")
pc = spark.read.parquet(f"{HD}/netflix/model/pred_content_test").select("user_id", "movie_id", "pred_content")
j = (pa.join(pc, ["user_id", "movie_id"]).withColumn("pc", F.coalesce("pred_content", "pred_bias"))
       .withColumn("e_bias", (F.col("pred_bias") - F.col("rating")) ** 2).cache())
n_pairs = j.count()
models = [("user+movie bias (baseline)", F.col("pred_bias")), ("ALS (+bias fallback)", F.col("pred_als_filled")),
          ("content kNN (+bias fallback)", F.col("pc"))] + \
         [(f"blend alpha={a} ALS + {1-a:.2f} content", a * F.col("pred_als_filled") + (1 - a) * F.col("pc")) for a in (0.25, 0.5, 0.75)]
hy_rows = []
for name, expr in models:
    e = j.withColumn("e", (clip(expr) - F.col("rating")) ** 2).withColumn("d", F.col("e") - F.col("e_bias"))
    r = e.agg(F.sqrt(F.avg("e")).alias("rmse"), F.avg("d").alias("dm"), F.stddev("d").alias("ds")).first()
    half = 1.96 * (r["ds"] or 0.0) / (n_pairs ** 0.5)
    lo, hi = r["dm"] - half, r["dm"] + half
    hy_rows.append((name, float(r["rmse"]), float(r["dm"]), float(lo), float(hi)))
spark.createDataFrame(hy_rows, "model string, rmse double, mse_diff_vs_bias double, ci_low double, ci_high double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/hybrid_rmse")

print("\n========== STEP 4D RESULTS ==========")
print(f"PART 1: top-{TOPK} ranking on {n_users} sampled test users, candidates = all {cands.count()} movies seen in training, pool = {n_pool} most-rated")
print(f"{'method':36}{'P@10':>8}{'R@10':>8}{'NDCG@10':>9}{'coverage':>10}")
for n_, p_, r_, g_, c_, u_ in rank_rows:
    print(f"{n_:36}{p_:>8.4f}{r_:>8.4f}{g_:>9.4f}{c_:>10.3f}")
print(f"\nPART 2: rating RMSE on {n_pairs:,} test pairs (difference = MSE of model minus MSE of bias; negative = better)")
print(f"{'model':36}{'RMSE':>8}{'MSE diff':>11}{'95% CI':>24}{'CI excludes 0':>15}")
for n_, rm, dm, lo, hi in hy_rows[1:]:
    print(f"{n_:36}{rm:>8.4f}{dm:>11.5f}   [{lo:>9.5f}, {hi:>9.5f}]{'yes' if (lo > 0 or hi < 0) else 'no':>12}")
print(f"{hy_rows[0][0]:36}{hy_rows[0][1]:>8.4f}")
print("=====================================")
spark.stop()
