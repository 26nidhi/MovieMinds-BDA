"""
Step 5A: two novelties, tuned on the VALIDATION slice only, judged once on the TEST slice.

  Novelty 1  Time-decayed models. Every rating gets weight exp(-age/tau) when estimating the user/movie
             biases and the content-kNN residuals. We pick tau on validation (tau = 0 means no decay).
  Novelty 2  Adaptive blend weight. Instead of a fixed 50/50 mix of ALS and content, alpha depends on
             context (user history length, movie popularity, content-evidence strength). For each bucket
             alpha has a closed form: the least-squares optimum  alpha = sum((r-c)(a-c)) / sum((a-c)^2),
             clipped to [0,1], fitted on validation pairs. We pick the best rule by validation RMSE.

Validation setup: models see only TRAIN, predict VAL. Test setup: models see TRAIN+VAL, predict TEST
(same as steps 4B-4D). Test is never used to choose tau, the rule, or any weight.

Ablation (test RMSE, user-clustered bootstrap CIs on every step):
  bias -> ALS -> +content (fixed 50/50) -> +time decay -> +adaptive alpha

Needs outputs of prepare_split.py, als_train.py, content_model.py.
Usage: spark-submit ... adaptive_blend.py [resamples=2000] [taus=0,540,270,120]
"""
import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS

HD = "hdfs://namenode:9000"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TAUS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 540, 270, 120]   # days; 0 = no decay
LAMBDA, L_M, L_U = 3.0, 10.0, 25.0
spark = (SparkSession.builder.appName("adaptive-blend")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir(f"{HD}/netflix/checkpoints")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))
rd = lambda p: spark.read.parquet(f"{HD}{p}")
rng = np.random.default_rng(2024)

sel = lambda df: df.select("user_id", "movie_id", F.col("rating").cast("double").alias("rating"), "day")
train, val, test = sel(rd("/netflix/model/train")), sel(rd("/netflix/model/val")), sel(rd("/netflix/model/test"))
full = train.unionByName(val)
nb = rd("/netflix/model/content_neighbors").withColumnRenamed("movie_id", "m")        # m, neighbor, sim

def features(hist, target, tau):
    """Bias + content-kNN predictions for the target pairs, using only the history `hist`."""
    tref = hist.agg(F.max("day")).first()[0]
    w = F.exp((F.col("day") - F.lit(tref)) / F.lit(float(tau))) if tau > 0 else F.lit(1.0)
    h = hist.withColumn("w", w)
    mu = h.agg(F.sum(F.col("w") * F.col("rating")) / F.sum("w")).first()[0]
    bm = (h.groupBy("movie_id").agg(F.sum(F.col("w") * (F.col("rating") - mu)).alias("s"), F.sum("w").alias("sw"),
                                   F.count("*").alias("n_movie"))
            .select("movie_id", (F.col("s") / (F.col("sw") + L_M)).alias("b_m"), "n_movie"))
    hh = h.join(bm, "movie_id").withColumn("r0", F.col("rating") - mu - F.col("b_m"))
    bu = (hh.groupBy("user_id").agg(F.sum(F.col("w") * F.col("r0")).alias("s"), F.sum("w").alias("sw"), F.count("*").alias("n_hist"))
             .select("user_id", (F.col("s") / (F.col("sw") + L_U)).alias("b_u"), "n_hist"))
    resid = hh.join(bu, "user_id").select("user_id", F.col("movie_id").alias("neighbor"), (F.col("r0") - F.col("b_u")).alias("res"), "w")
    cand = (target.select("user_id", "movie_id").join(F.broadcast(nb), F.col("movie_id") == F.col("m"))
                  .select("user_id", "movie_id", "neighbor", "sim").join(resid, ["user_id", "neighbor"]))
    agg = cand.groupBy("user_id", "movie_id").agg(F.sum(F.col("sim") * F.col("w") * F.col("res")).alias("num"),
                                                  F.sum(F.col("sim") * F.col("w")).alias("den"))
    out = (target.join(bm, "movie_id", "left").join(bu, "user_id", "left").join(agg, ["user_id", "movie_id"], "left")
             .withColumn("n_movie", F.coalesce("n_movie", F.lit(0))).withColumn("n_hist", F.coalesce("n_hist", F.lit(0)))
             .withColumn("pred_bias", clip(F.lit(mu) + F.coalesce("b_m", F.lit(0.0)) + F.coalesce("b_u", F.lit(0.0))))
             .withColumn("pred_content", F.when(F.col("den").isNotNull(),
                         clip(F.col("pred_bias") + F.col("num") / (F.col("den") + F.lit(LAMBDA))))))
    return out.select("user_id", "movie_id", "rating", "pred_bias", "pred_content", "den", "n_hist", "n_movie")

def assemble(feat, als_df):
    d = feat.join(als_df, ["user_id", "movie_id"], "left")
    return d.withColumn("a", clip(F.coalesce("pred_als", "pred_bias"))).withColumn("c", F.coalesce("pred_content", "pred_bias"))

def add_buckets(d):
    return (d.withColumn("b_user", F.when(F.col("n_hist") == 0, "0").when(F.col("n_hist") < 20, "1-19").when(F.col("n_hist") < 100, "20-99")
                                    .when(F.col("n_hist") < 500, "100-499").otherwise("500+"))
             .withColumn("b_movie", F.when(F.col("n_movie") == 0, "0").when(F.col("n_movie") < 10, "1-9").when(F.col("n_movie") < 100, "10-99")
                                     .when(F.col("n_movie") < 1000, "100-999").otherwise("1000+"))
             .withColumn("b_conf", F.when(F.col("den").isNull(), "none").when(F.col("den") < 0.5, "<0.5").when(F.col("den") < 1.5, "0.5-1.5")
                                    .when(F.col("den") < 3, "1.5-3").otherwise("3+")))

rmse = lambda df, col: df.agg(F.sqrt(F.avg((F.col(col) - F.col("rating")) ** 2))).first()[0]

# ---------------- ALS fitted on TRAIN only (for validation predictions) ----------------
g = rd("/netflix/results/als_grid").orderBy("val_rmse").first()
als = ALS(rank=int(g["rank"]), regParam=float(g["reg"]), maxIter=10, userCol="user_id", itemCol="movie_id", ratingCol="rating",
          coldStartStrategy="nan", numUserBlocks=16, numItemBlocks=16, checkpointInterval=2, seed=42) \
        .fit(train.select("user_id", "movie_id", F.col("rating").cast("float")))
va_als = (als.transform(val.select("user_id", "movie_id"))
             .withColumn("pred_als", F.when(F.isnan("prediction"), None).otherwise(F.col("prediction")))
             .select("user_id", "movie_id", "pred_als")).cache()
te_als = rd("/netflix/model/pred_als_test").select("user_id", "movie_id", "pred_als")
print(f"[als] validation predictions from ALS rank={g['rank']} reg={g['reg']} fitted on train only", flush=True)

# ---------------- Novelty 1: choose tau on validation ----------------
decay_rows, best = [], None
for tau in TAUS:
    fv = assemble(features(train, val, tau), va_als).withColumn("blend", clip(0.5 * F.col("a") + 0.5 * F.col("c"))).cache()
    r = (rmse(fv, "pred_bias"), rmse(fv, "c"), rmse(fv, "blend"))
    decay_rows.append((float(tau), *map(float, r)))
    print(f"[decay] tau={tau:>4} days  val RMSE: bias {r[0]:.4f}   content {r[1]:.4f}   blend50 {r[2]:.4f}", flush=True)
    if best is None or r[2] < best[0]:
        best = (r[2], tau)
    fv.unpersist()
TAU = best[1]
spark.createDataFrame(decay_rows, "tau_days double, val_rmse_bias double, val_rmse_content double, val_rmse_blend50 double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/decay_search")
print(f"[decay] chosen tau = {TAU} days (lowest validation RMSE of the 50/50 blend)", flush=True)

# ---------------- Novelty 2: fit alpha rules on validation ----------------
fv = add_buckets(assemble(features(train, val, TAU), va_als)).cache()
rules = {"fixed 0.5": [], "by user history": ["b_user"], "by movie popularity": ["b_movie"],
         "by content evidence": ["b_conf"], "user history x popularity": ["b_user", "b_movie"]}

def fit_alpha(df, keys):
    gg = df.groupBy(*keys).agg(F.sum((F.col("rating") - F.col("c")) * (F.col("a") - F.col("c"))).alias("num"),
                               F.sum((F.col("a") - F.col("c")) ** 2).alias("dd"), F.count("*").alias("n"))
    return gg.withColumn("alpha", F.when(F.col("dd") > 1e-9, F.least(F.greatest(F.col("num") / F.col("dd"), F.lit(0.0)), F.lit(1.0)))
                                   .otherwise(F.lit(0.5))).select(*keys, "alpha", "n").cache()

def apply_rule(df, keys, table):
    if not keys:
        return df.withColumn("alpha", F.lit(0.5)).withColumn("adaptive", clip(0.5 * F.col("a") + 0.5 * F.col("c")))
    d = df.join(F.broadcast(table.select(*keys, "alpha")), keys, "left").withColumn("alpha", F.coalesce("alpha", F.lit(0.5)))
    return d.withColumn("adaptive", clip(F.col("alpha") * F.col("a") + (1 - F.col("alpha")) * F.col("c")))

tables, rule_rows, best_rule = {}, [], None
for name, keys in rules.items():
    tables[name] = fit_alpha(fv, keys) if keys else None
    r = rmse(apply_rule(fv, keys, tables[name]), "adaptive")
    print(f"[rule] {name:28} validation RMSE {r:.4f}", flush=True)
    rule_rows.append((name, float(r)))
    if best_rule is None or r < best_rule[0]:
        best_rule = (r, name)
RULE = best_rule[1]
print(f"[rule] chosen: '{RULE}' (lowest validation RMSE; in-sample on {fv.count():,} val pairs with at most 25 fitted weights)", flush=True)
if rules[RULE]:
    rows = tables[RULE].orderBy(*rules[RULE]).collect()
    print("[rule] fitted alpha (weight on ALS; 1-alpha on content):")
    for r in rows:
        print("        " + ", ".join(f"{k}={r[k]}" for k in rules[RULE]) + f"   alpha={r['alpha']:.2f}   (val pairs: {r['n']:,})")
    tables[RULE].coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/alpha_table")
spark.createDataFrame(rule_rows, "rule string, val_rmse double").coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/alpha_rules")

# ---------------- Test: ablation with user-clustered bootstrap ----------------
t0 = assemble(features(full, test, 0), te_als).select("user_id", "movie_id", "rating", F.col("pred_bias").alias("bias0"),
                                                      F.col("a").alias("a0"), F.col("c").alias("c0"))
ts = apply_rule(add_buckets(assemble(features(full, test, TAU), te_als)), rules[RULE], tables[RULE])
d = ts.join(t0.drop("rating"), ["user_id", "movie_id"])
models = [("bias baseline", F.col("bias0")),
          ("ALS", clip(F.col("a0"))),
          ("ALS + content (fixed 50/50, no decay)", clip(0.5 * F.col("a0") + 0.5 * F.col("c0"))),
          (f"+ time decay (tau={TAU}d)", clip(0.5 * F.col("a") + 0.5 * F.col("c"))),
          (f"+ adaptive alpha ({RULE})", F.col("adaptive"))]
per_user = d.groupBy("user_id").agg(F.count("*").alias("n"), *[F.sum((e - F.col("rating")) ** 2).alias(f"se{i}") for i, (_, e) in enumerate(models)])
arr = np.array([[r["n"]] + [r[f"se{i}"] for i in range(len(models))] for r in per_user.collect()], dtype=float)
rm = lambda s, i: np.sqrt(s[i + 1] / s[0])

def boot(stat, chunk=100):
    U = arr.shape[0]; est = stat(arr.sum(axis=0)); vals = []; p = np.full(U, 1.0 / U)
    while len(vals) < B:
        counts = rng.multinomial(U, p, size=min(chunk, B - len(vals)))
        vals.extend(stat(s) for s in counts @ arr)
    lo, hi = np.percentile(vals, [2.5, 97.5]); return est, lo, hi

out_rows = []
print("\n========== STEP 5A RESULTS (test set) ==========")
print(f"{int(arr[:,0].sum()):,} test pairs, {arr.shape[0]:,} users, {B} user-level resamples; tau and alpha rule chosen on validation only")
print(f"{'model':46}{'RMSE':>8}   {'change vs previous row (95% CI)':<38}")
for i, (name, _) in enumerate(models):
    est = float(rm(arr.sum(axis=0), i))
    if i == 0:
        print(f"{name:46}{est:>8.4f}"); out_rows.append((name, est, None, None, None)); continue
    de, lo, hi = boot(lambda s, i=i: rm(s, i) - rm(s, i - 1))
    flag = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
    print(f"{name:46}{est:>8.4f}   {de:+.4f} [{lo:+.4f}, {hi:+.4f}] {flag}")
    out_rows.append((name, est, float(de), float(lo), float(hi)))
de, lo, hi = boot(lambda s: rm(s, len(models) - 1) - rm(s, 0))
print(f"{'full model vs bias baseline':46}{'':>8}   {de:+.4f} [{lo:+.4f}, {hi:+.4f}] {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")
print("=================================================")
spark.createDataFrame(out_rows, "model string, rmse double, delta_vs_prev double, ci_low double, ci_high double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/ablation_rmse")
spark.stop()
