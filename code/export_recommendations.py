"""
Step 6B (data for the Recommender page): per-user candidate lists with everything the page needs.

For each of the sampled users (from hybrid_eval.py -> rank_scores) we keep the top-60 candidates by ALS
and the top-60 by the adaptive hybrid (union), from the popular pool, and attach:
  title/year, TMDB poster path, genres, ALS and content predicted ratings, the adaptive weight alpha
  (share of the prediction that is collaborative), polarization z (from TRAIN+VAL ratings only),
  a "because you rated X" reason from content neighbours, and whether the user later rated it >= 4.
Writes /netflix/results/rec_candidates and /netflix/results/rec_users (small tables for the dashboard).

Usage: spark-submit ... export_recommendations.py
"""
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

HD = "hdfs://namenode:9000"
N_PER = 60
spark = (SparkSession.builder.appName("export-recommendations")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
rd = lambda p: spark.read.parquet(f"{HD}{p}")
clip = lambda c: F.least(F.greatest(c, F.lit(1.0)), F.lit(5.0))
sel = lambda df: df.select("user_id", "movie_id", F.col("rating").cast("double").alias("rating"), "day")
train, val, test = sel(rd("/netflix/model/train")), sel(rd("/netflix/model/val")), sel(rd("/netflix/model/test"))
full = train.unionByName(val)

S = rd("/netflix/model/rank_scores").filter((F.col("in_pool") == 1) & F.col("als").isNotNull())
users = S.select("user_id", "R").distinct()
nb = rd("/netflix/model/content_neighbors").withColumnRenamed("movie_id", "m")            # m, neighbor, sim

# ---- titles for all movies, TMDB metadata for matched ones
raw = spark.read.option("encoding", "ISO-8859-1").option("sep", "\u0001").csv(f"{HD}/netflix/meta/movie_titles.csv")
titles = (raw.select(F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 1).cast("int").alias("movie_id"),
                     F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 2).alias("year_s"),
                     F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 3).alias("title"))
             .withColumn("year", F.when(F.col("year_s") != "NULL", F.col("year_s").cast("int"))).drop("year_s"))
meta = rd("/netflix/meta/matches").select("movie_id", F.col("nf_title").alias("m_title"), "poster_path",
                                          F.concat_ws(", ", "genres").alias("genres"), "overview")
tmeta = titles.join(meta, "movie_id", "left").withColumn("title", F.coalesce("title", "m_title")).drop("m_title")

# ---- user history and content-neighbour evidence
hist = full.join(users.select("user_id"), "user_id")
hist_n = hist.groupBy("user_id").agg(F.count("*").alias("n_hist"), F.round(F.avg("rating"), 2).alias("avg_hist_rating"))
contrib = (hist.select("user_id", F.col("movie_id").alias("neighbor"), F.col("rating").alias("r_j"))
               .join(nb, "neighbor").select("user_id", F.col("m").alias("movie_id"), "neighbor", "sim", "r_j"))
den = contrib.groupBy("user_id", "movie_id").agg(F.sum("sim").alias("den"))

# ---- adaptive weight alpha (the rule fitted in adaptive_blend.py)
at = rd("/netflix/results/alpha_table"); akeys = [c for c in at.columns if c not in ("alpha", "n")]
S2 = (S.join(hist_n.select("user_id", "n_hist"), "user_id", "left").join(den, ["user_id", "movie_id"], "left")
        .withColumn("n_hist", F.coalesce("n_hist", F.lit(0)))
        .withColumn("b_user", F.when(F.col("n_hist") == 0, "0").when(F.col("n_hist") < 20, "1-19").when(F.col("n_hist") < 100, "20-99")
                               .when(F.col("n_hist") < 500, "100-499").otherwise("500+"))
        .withColumn("b_movie", F.when(F.col("n_movie") == 0, "0").when(F.col("n_movie") < 10, "1-9").when(F.col("n_movie") < 100, "10-99")
                                .when(F.col("n_movie") < 1000, "100-999").otherwise("1000+"))
        .withColumn("b_conf", F.when(F.col("den").isNull(), "none").when(F.col("den") < 0.5, "<0.5").when(F.col("den") < 1.5, "0.5-1.5")
                               .when(F.col("den") < 3, "1.5-3").otherwise("3+")))
S2 = S2.join(F.broadcast(at.select(*akeys, "alpha")), akeys, "left").withColumn("alpha", F.coalesce("alpha", F.lit(0.5)))
S2 = S2.withColumn("adaptive", clip(F.col("alpha") * F.col("als") + (1 - F.col("alpha")) * F.col("content")))

# ---- polarization z from TRAIN+VAL only, standardised over the pool
pol = full.groupBy("movie_id").agg(F.stddev("rating").alias("pol"))
mu_p, sd_p = pol.join(S.select("movie_id").distinct(), "movie_id").agg(F.avg("pol"), F.stddev("pol")).first()
pol = pol.withColumn("pol", F.coalesce("pol", F.lit(mu_p))).withColumn("z", (F.col("pol") - F.lit(mu_p)) / F.lit(sd_p))

# ---- candidate selection
w_als = Window.partitionBy("user_id").orderBy(F.desc("als"), F.asc("movie_id"))
w_adp = Window.partitionBy("user_id").orderBy(F.desc("adaptive"), F.asc("movie_id"))
C = (S2.withColumn("rank_als", F.row_number().over(w_als)).withColumn("rank_adaptive", F.row_number().over(w_adp))
       .filter((F.col("rank_als") <= N_PER) | (F.col("rank_adaptive") <= N_PER)))

# ---- "because you rated X": best neighbour the user rated (prefer liked ones, then most similar)
w_b = Window.partitionBy("user_id", "movie_id").orderBy((F.col("r_j") >= 4).cast("int").desc(), F.desc("sim"))
because = (contrib.join(C.select("user_id", "movie_id"), ["user_id", "movie_id"])
                  .withColumn("rn", F.row_number().over(w_b)).filter("rn = 1")
                  .join(titles.select(F.col("movie_id").alias("neighbor"), F.col("title").alias("because_title")), "neighbor")
                  .select("user_id", "movie_id", "because_title", F.col("r_j").alias("because_rating"), F.col("sim").alias("because_sim")))

out = (C.join(tmeta.select("movie_id", "title", "year", "poster_path", "genres", "overview"), "movie_id", "left")
        .join(pol.select("movie_id", "pol", "z"), "movie_id", "left")
        .join(because, ["user_id", "movie_id"], "left")
        .join(test.select("user_id", "movie_id", F.col("rating").alias("test_rating")), ["user_id", "movie_id"], "left")
        .withColumn("z", F.coalesce("z", F.lit(0.0)))
        .select("user_id", "movie_id", "title", "year", "poster_path", "genres", "overview", "als", "content", "adaptive", "alpha",
                "pol", "z", "n_movie", "rank_als", "rank_adaptive", "because_title", "because_rating", "because_sim", "test_rating"))
out.coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/rec_candidates")

# ---- user table: history summary and what they liked
w_h = Window.partitionBy("user_id").orderBy(F.desc("rating"), F.desc("day"))
liked = (hist.withColumn("rn", F.row_number().over(w_h)).filter("rn <= 5")
             .join(titles.select("movie_id", "title"), "movie_id", "left")
             .withColumn("s", F.concat(F.coalesce("title", F.lit("movie ")), F.lit(" ("), F.col("rating").cast("int").cast("string"), F.lit("★)")))
             .groupBy("user_id").agg(F.concat_ws("; ", F.expr("transform(sort_array(collect_list(struct(rn, s))), x -> x.s)")).alias("liked_before")))
(users.join(hist_n, "user_id", "left").join(liked, "user_id", "left")
      .withColumnRenamed("R", "n_liked_in_test").coalesce(1)
      .write.mode("overwrite").parquet(f"{HD}/netflix/results/rec_users"))
print(f"[export] candidates: {spark.read.parquet(HD + '/netflix/results/rec_candidates').count():,} rows for "
      f"{spark.read.parquet(HD + '/netflix/results/rec_users').count():,} users; alpha keys: {akeys}")
spark.stop()
