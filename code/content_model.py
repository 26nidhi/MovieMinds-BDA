"""
Step 4C: Content-based recommender from TMDB plots + genres.

  1. TF-IDF per movie, written as plain Spark SQL (no Python UDFs, so no extra packages on workers).
  2. Exact cosine similarity between movies with an inverted-index join; keep the 20 most similar
     movies per movie  ->  /netflix/model/content_neighbors
  3. (optional, add the word  lsh ) MinHash LSH approximate neighbours, compared with the exact ones
     for speed and recall
  4. Rating predictor = item-kNN on content similarity, working on residuals of the bias baseline:
        pred(u,m) = bias(u,m) + sum_j sim(m,j) * resid(u,j) / (sum_j sim(m,j) + LAMBDA)
     where j runs over the movies user u rated in training that are content-neighbours of m.
  5. Like-for-like evaluation against the bias baseline on the same test pairs, by movie popularity.

Only movies matched to TMDB have content, so coverage is limited (stated in the output).

Usage: spark-submit ... content_model.py [lsh]
"""
import sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, MinHashLSH

HD = "hdfs://namenode:9000"
MAXDF = 0.05          # drop words that appear in more than 5% of plots (too generic to be useful)
K = 20                # neighbours kept per movie
LAMBDA = 3.0          # shrinkage: fixed in advance, NOT tuned on the test set
MIN_SIM = 0.05
spark = (SparkSession.builder.appName("content-model")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))

# ---------------- 1. TF-IDF in SQL ----------------
m = spark.read.parquet(f"{HD}/netflix/meta/matches").select("movie_id", "nf_title", "tm_title", "overview", "genres")
docs = (m.withColumn("text", F.concat_ws(" ", F.col("tm_title"), F.coalesce("overview", F.lit("")), F.concat_ws(" ", "genres")))
          .filter(F.length("text") > 0).select("movie_id", "nf_title", "text")).cache()
n_docs = docs.count()
tok = RegexTokenizer(inputCol="text", outputCol="tok", pattern="\\W+", minTokenLength=3).transform(docs)
tok = StopWordsRemover(inputCol="tok", outputCol="words").transform(tok).cache()
tf = tok.select("movie_id", F.explode("words").alias("term")).groupBy("movie_id", "term").agg(F.count("*").alias("tf"))
dfc = tf.groupBy("term").agg(F.count("*").alias("df")).filter((F.col("df") >= 2) & (F.col("df") <= MAXDF * n_docs))
w = tf.join(dfc, "term").withColumn("w", (1 + F.log("tf")) * (F.log((n_docs + 1.0) / (F.col("df") + 1)) + 1))
nrm = w.groupBy("movie_id").agg(F.sqrt(F.sum(F.col("w") * F.col("w"))).alias("nrm"))
tw = w.join(nrm, "movie_id").select("movie_id", "term", (F.col("w") / F.col("nrm")).alias("w")).cache()
n_vocab = tw.select("term").distinct().count(); n_tw_movies = tw.select("movie_id").distinct().count()

# ---------------- 2. exact cosine neighbours (inverted-index join) ----------------
t0 = time.time()
a = tw.select(F.col("movie_id").alias("m1"), "term", F.col("w").alias("w1"))
b = tw.select(F.col("movie_id").alias("m2"), "term", F.col("w").alias("w2"))
pairs = (a.join(b, (a.term == b.term) & (a.m1 < b.m2)).groupBy("m1", "m2")
          .agg(F.sum(F.col("w1") * F.col("w2")).alias("sim")).filter(F.col("sim") >= MIN_SIM))
sym = (pairs.select(F.col("m1").alias("movie_id"), F.col("m2").alias("neighbor"), "sim")
       .unionByName(pairs.select(F.col("m2").alias("movie_id"), F.col("m1").alias("neighbor"), "sim")))
nb = (sym.withColumn("rk", F.row_number().over(Window.partitionBy("movie_id").orderBy(F.desc("sim"))))
         .filter(F.col("rk") <= K).drop("rk"))
nb.write.mode("overwrite").parquet(f"{HD}/netflix/model/content_neighbors")
nb = spark.read.parquet(f"{HD}/netflix/model/content_neighbors").cache()
n_nb = nb.count(); t_exact = time.time() - t0
print(f"[content] {n_docs} movies with text, {n_tw_movies} with usable words, vocabulary {n_vocab}", flush=True)
print(f"[content] exact cosine neighbours: {n_nb} rows ({nb.select('movie_id').distinct().count()} movies) in {t_exact:.0f}s", flush=True)

# eyeball check: nearest neighbours of a few titles
titles = docs.select("movie_id", "nf_title")
for pat in ("%Fellowship of the Ring%", "Toy Story%", "The Matrix%"):
    row = titles.filter(F.col("nf_title").like(pat)).first()
    if row:
        out = (nb.filter(F.col("movie_id") == row["movie_id"]).orderBy(F.desc("sim")).limit(5)
                 .join(titles.withColumnRenamed("movie_id", "neighbor").withColumnRenamed("nf_title", "nb_title"), "neighbor")
                 .orderBy(F.desc("sim")).collect())
        print(f"[neighbours of '{row['nf_title']}'] " + "; ".join(f"{r['nb_title']} ({r['sim']:.2f})" for r in out), flush=True)

# ---------------- 3. optional LSH ----------------
lsh_line = None
if "lsh" in sys.argv:
    t0 = time.time()
    ids = tw.select("movie_id").distinct()
    d = tok.join(ids, "movie_id").select("movie_id", "words")
    cv = CountVectorizer(inputCol="words", outputCol="bin", binary=True, minDF=2.0, maxDF=MAXDF).fit(d)
    dv = cv.transform(d).select("movie_id", "bin").cache(); dv.count()
    lm = MinHashLSH(inputCol="bin", outputCol="h", numHashTables=5, seed=1).fit(dv)
    j = lm.approxSimilarityJoin(dv, dv, 0.95, distCol="jd").select(
        F.col("datasetA.movie_id").alias("movie_id"), F.col("datasetB.movie_id").alias("neighbor"), "jd"
    ).filter("movie_id != neighbor")
    lnb = (j.withColumn("rk", F.row_number().over(Window.partitionBy("movie_id").orderBy("jd")))
             .filter(F.col("rk") <= K).select("movie_id", "neighbor")).cache()
    n_l = lnb.count(); t_lsh = time.time() - t0
    top5 = (nb.withColumn("rk", F.row_number().over(Window.partitionBy("movie_id").orderBy(F.desc("sim"))))
              .filter(F.col("rk") <= 5).select("movie_id", "neighbor"))
    found = top5.join(lnb.withColumn("hit", F.lit(1)), ["movie_id", "neighbor"], "left").agg(
        F.count("*").alias("n"), F.sum(F.coalesce("hit", F.lit(0))).alias("h")).first()
    lsh_line = (f"MinHash LSH ({t_lsh:.0f}s, {n_l} neighbour rows) vs exact cosine ({t_exact:.0f}s): "
                f"{found['h']}/{found['n']} = {100*found['h']/max(found['n'],1):.1f}% of each movie's 5 true nearest neighbours "
                f"appear in its LSH top-{K} (LSH ranks by Jaccard on word sets, exact by TF-IDF cosine)")
    print("[lsh] " + lsh_line, flush=True)

# ---------------- 4. rating prediction on the test set ----------------
train = spark.read.parquet(f"{HD}/netflix/model/train").select("user_id", "movie_id", F.col("rating").cast("double"))
val = spark.read.parquet(f"{HD}/netflix/model/val").select("user_id", "movie_id", F.col("rating").cast("double"))
test = spark.read.parquet(f"{HD}/netflix/model/test").select("user_id", "movie_id", F.col("rating").cast("double")).cache()
full = train.unionByName(val).cache()
mu = full.agg(F.avg("rating")).first()[0]
bm = (full.groupBy("movie_id").agg(F.sum(F.col("rating") - mu).alias("s"), F.count("*").alias("n"))
          .select("movie_id", (F.col("s") / (F.col("n") + 10.0)).alias("b_m"), F.col("n").alias("n_movie"))).cache()
fu = full.join(bm, "movie_id").withColumn("res", F.col("rating") - mu - F.col("b_m"))
bu = (fu.groupBy("user_id").agg(F.sum("res").alias("s"), F.count("*").alias("n"))
        .select("user_id", (F.col("s") / (F.col("n") + 25.0)).alias("b_u"))).cache()
resid = (full.join(bm, "movie_id").join(bu, "user_id")
             .select("user_id", F.col("movie_id").alias("neighbor"), (F.col("rating") - mu - F.col("b_m") - F.col("b_u")).alias("res")))
cand = (test.join(F.broadcast(nb.withColumnRenamed("movie_id", "m")), test.movie_id == F.col("m"))
            .select("user_id", F.col("m").alias("movie_id"), "neighbor", "sim")
            .join(resid, ["user_id", "neighbor"]))
agg = cand.groupBy("user_id", "movie_id").agg(F.sum(F.col("sim") * F.col("res")).alias("num"),
                                               F.sum("sim").alias("den"), F.count("*").alias("n_nb_rated"))
p = (test.join(bm, "movie_id", "left").join(bu, "user_id", "left").join(agg, ["user_id", "movie_id"], "left")
         .withColumn("n_movie", F.coalesce("n_movie", F.lit(0)))
         .withColumn("pred_bias", clip(F.lit(mu) + F.coalesce("b_m", F.lit(0.0)) + F.coalesce("b_u", F.lit(0.0))))
         .withColumn("pred_content", F.when(F.col("den").isNotNull(),
                     clip(F.col("pred_bias") + F.col("num") / (F.col("den") + F.lit(LAMBDA)))))
         .cache())
p.select("user_id", "movie_id", "rating", "pred_bias", "pred_content", "den", "n_nb_rated", "n_movie") \
 .write.mode("overwrite").parquet(f"{HD}/netflix/model/pred_content_test")

def rmse(df, col):
    return df.agg(F.sqrt(F.avg((F.col(col) - F.col("rating")) ** 2))).first()[0]
n_test = p.count()
cov_df = p.filter(F.col("pred_content").isNotNull()).cache(); n_cov = cov_df.count()
r_bias_all = rmse(p, "pred_bias")
r_bias_cov, r_cont_cov = rmse(cov_df, "pred_bias"), rmse(cov_df, "pred_content")
r_cont_all = rmse(p.withColumn("pc", F.coalesce("pred_content", "pred_bias")), "pc")
buckets = [("0 (movie unseen)", 0, 0), ("1-9", 1, 9), ("10-99", 10, 99), ("100-999", 100, 999), ("1000+", 1000, 10**9)]
brow = []
for name, lo, hi in buckets:
    allb = p.filter((F.col("n_movie") >= lo) & (F.col("n_movie") <= hi))
    cb = cov_df.filter((F.col("n_movie") >= lo) & (F.col("n_movie") <= hi))
    na, nc = allb.count(), cb.count()
    brow.append((name, na, nc, rmse(cb, "pred_bias") if nc else None, rmse(cb, "pred_content") if nc else None))
spark.createDataFrame(brow, "bucket string, test_pairs long, content_covered long, rmse_bias double, rmse_content double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/content_by_bucket")
spark.createDataFrame([("user+movie bias, all pairs", r_bias_all, 1.0), ("content kNN, covered pairs", r_cont_cov, n_cov / n_test),
                       ("bias on the same covered pairs", r_bias_cov, n_cov / n_test), ("content kNN + bias fallback, all pairs", r_cont_all, 1.0)],
                      "model string, rmse double, coverage double").coalesce(1).write.mode("overwrite") \
     .parquet(f"{HD}/netflix/results/model_metrics_content")

print("\n========== STEP 4C RESULTS (test set) ==========")
print(f"movies with content: {n_tw_movies} (only TMDB-matched movies); lambda={LAMBDA}, K={K}")
print(f"content model can predict {n_cov:,} of {n_test:,} test pairs = {100*n_cov/n_test:.1f}%")
print(f"RMSE on those pairs:  content kNN {r_cont_cov:.4f}   vs   bias baseline {r_bias_cov:.4f}")
print(f"RMSE on all pairs:    content kNN + bias fallback {r_cont_all:.4f}   vs   bias baseline {r_bias_all:.4f}")
print(f"{'movie popularity (train ratings)':34}{'test pairs':>11}{'covered':>9}{'bias':>8}{'content':>9}")
for name, na, nc, rb, rc in brow:
    print(f"{name:34}{na:>11}{nc:>9}{'' if rb is None else format(rb, '.4f'):>8}{'' if rc is None else format(rc, '.4f'):>9}")
if lsh_line:
    print(lsh_line)
print("================================================")
spark.stop()
