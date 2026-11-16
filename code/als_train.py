"""
Step 4B: Baselines + ALS (Spark MLlib) with a temporal tuning protocol.

  1. Baselines on the test set: global mean, movie mean, and a damped user+movie bias model.
  2. ALS grid search: fit on train, score on val  (temporal validation, NOT random folds,
     because random CrossValidator folds would mix past and future ratings).
  3. Refit the best setting on train+val, score on test.

All predictions are clipped to [1, 5]. We report two ALS numbers: RMSE on the pairs ALS could
predict (users and movies seen in training) and RMSE over ALL test pairs, where ALS falls back
to the bias baseline for unseen users/movies.

Usage: spark-submit ... als_train.py
"""
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

HD = "hdfs://namenode:9000"
GRID = [(10, 0.1), (10, 0.2), (10, 0.4), (20, 0.2)]       # (rank, regParam); v1 found the best reg at the edge (0.1), so v2 looks higher
MAX_ITER, BLOCKS = 10, 16
spark = (SparkSession.builder.appName("als-train")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())
spark.sparkContext.setCheckpointDir(f"{HD}/netflix/checkpoints")
spark.sparkContext.setLogLevel("ERROR")

train = spark.read.parquet(f"{HD}/netflix/model/train").select("user_id", "movie_id", F.col("rating").cast("float"))
val = spark.read.parquet(f"{HD}/netflix/model/val").select("user_id", "movie_id", F.col("rating").cast("float"))
test = spark.read.parquet(f"{HD}/netflix/model/test").select("user_id", "movie_id", F.col("rating").cast("float"))
full = train.unionByName(val).cache()
test = test.cache(); n_test = test.count()
rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="pred")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))
metrics = []                                     # (model, rmse_all_pairs, rmse_predictable, coverage, seconds)

# ---------------- 1. baselines (fit on train+val) ----------------
mu = full.agg(F.avg("rating")).first()[0]
L_M, L_U = 10.0, 25.0                            # damping: shrink biases of rarely seen movies/users to 0
bm = (full.groupBy("movie_id").agg(F.sum(F.col("rating") - mu).alias("s"), F.count("*").alias("n"))
          .select("movie_id", (F.col("s") / (F.col("n") + L_M)).alias("b_m"), F.col("n").alias("n_movie")))
fu = full.join(bm, "movie_id").withColumn("res", F.col("rating") - mu - F.col("b_m"))
bu = (fu.groupBy("user_id").agg(F.sum("res").alias("s"), F.count("*").alias("n"))
        .select("user_id", (F.col("s") / (F.col("n") + L_U)).alias("b_u")))
bm.cache(); bu.cache()
base = (test.join(bm, "movie_id", "left").join(bu, "user_id", "left")
            .withColumn("pred_bias", clip(F.lit(mu) + F.coalesce("b_m", F.lit(0.0)) + F.coalesce("b_u", F.lit(0.0))))
            .withColumn("pred_movie_mean", clip(F.lit(mu) + F.coalesce("b_m", F.lit(0.0)))).cache())
for name, col in (("global mean", None), ("movie mean (damped)", "pred_movie_mean"), ("user+movie bias (damped)", "pred_bias")):
    d = base.withColumn("pred", F.lit(mu) if col is None else F.col(col))
    r = rmse_eval.evaluate(d)
    metrics.append((name, float(r), float(r), 1.0, 0.0))
    print(f"[baseline] {name:28} test RMSE {r:.4f}", flush=True)

# ---------------- 2. ALS grid on train -> val ----------------
def make_als(rank, reg):
    return ALS(rank=rank, regParam=reg, maxIter=MAX_ITER, userCol="user_id", itemCol="movie_id", ratingCol="rating",
               coldStartStrategy="drop", numUserBlocks=BLOCKS, numItemBlocks=BLOCKS, checkpointInterval=2, seed=42)
grid_rows, best = [], None
n_val = val.count()
for rank, reg in GRID:
    t0 = time.time()
    m = make_als(rank, reg).fit(train)
    p = m.transform(val).withColumn("pred", clip(F.col("prediction")))
    r = rmse_eval.evaluate(p); cov = p.count() / n_val; dt = time.time() - t0
    grid_rows.append((rank, reg, float(r), float(cov), round(dt, 1)))
    print(f"[als grid] rank={rank:<3} reg={reg:<5} val RMSE {r:.4f}  coverage {cov:.3f}  ({dt:.0f}s)", flush=True)
    if best is None or r < best[0]:
        best = (r, rank, reg)
spark.createDataFrame(grid_rows, "rank int, reg double, val_rmse double, val_coverage double, seconds double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/als_grid")

# ---------------- 3. refit best on train+val -> test ----------------
_, rank, reg = best
t0 = time.time()
model = make_als(rank, reg).fit(full)
fit_s = time.time() - t0
model.write().overwrite().save(f"{HD}/netflix/model/als_best")
model.setColdStartStrategy("nan")                # keep every test row; unseen users/movies get NaN
pt = model.transform(test).withColumn("pred_als", F.when(F.isnan("prediction"), None).otherwise(F.col("prediction")))
p = (pt.join(bm.select("movie_id", "b_m"), "movie_id", "left").join(bu, "user_id", "left")
       .withColumn("pred_bias", clip(F.lit(mu) + F.coalesce("b_m", F.lit(0.0)) + F.coalesce("b_u", F.lit(0.0))))
       .withColumn("pred_als_filled", clip(F.coalesce("pred_als", F.col("pred_bias")))).cache())
got = p.filter(F.col("pred_als").isNotNull()).withColumn("pred", clip(F.col("pred_als")))
r_pred = rmse_eval.evaluate(got); cov = got.count() / n_test
r_all = rmse_eval.evaluate(p.withColumn("pred", F.col("pred_als_filled")))
# like-for-like: how does the BIAS baseline do on exactly the pairs ALS can predict, and on the rest?
same = p.filter(F.col("pred_als").isNotNull())
r_bias_same = rmse_eval.evaluate(same.withColumn("pred", F.col("pred_bias")))
rest = p.filter(F.col("pred_als").isNull())
r_rest = rmse_eval.evaluate(rest.withColumn("pred", F.col("pred_bias"))) if rest.count() else float("nan")
metrics.append(("user+movie bias, SAME predictable pairs as ALS", float(r_bias_same), float(r_bias_same), float(cov), 0.0))
metrics.append(("user+movie bias, pairs ALS cannot predict", float(r_rest), float(r_rest), float(1 - cov), 0.0))
metrics.append((f"ALS rank={rank} reg={reg} (predictable pairs only)", float(r_pred), float(r_pred), float(cov), round(fit_s, 1)))
metrics.append((f"ALS rank={rank} reg={reg} (+bias fallback, all pairs)", float(r_all), float(r_pred), 1.0, round(fit_s, 1)))
p.select("user_id", "movie_id", "rating", "pred_als", "pred_bias", "pred_als_filled", "b_m") \
 .write.mode("overwrite").parquet(f"{HD}/netflix/model/pred_als_test")
bm.write.mode("overwrite").parquet(f"{HD}/netflix/model/movie_bias")
spark.createDataFrame(metrics, "model string, rmse_all_pairs double, rmse_predictable_pairs double, coverage double, fit_seconds double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/model_metrics_als")

print("\n========== STEP 4B RESULTS (test set, newest 10% of ratings) ==========")
print(f"{'model':58}{'RMSE':>8}{'coverage':>10}")
for name, ra, rp, c, s in metrics:
    print(f"{name:58}{ra:>8.4f}{c:>10.3f}")
print(f"ALS vs bias on the SAME {100*cov:.1f}% of test pairs: {r_pred:.4f} vs {r_bias_same:.4f}")
print(f"best ALS: rank={rank}, regParam={reg}, maxIter={MAX_ITER}; final fit {fit_s:.0f}s")
print("=======================================================================")
spark.stop()