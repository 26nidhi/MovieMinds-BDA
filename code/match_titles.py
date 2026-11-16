"""
Step 2b: Match Netflix movies to TMDB by title + year, fully in Spark.

Stage 1 (exact):  normalised title equal and release years within +-1.
Stage 2 (fuzzy):  MinHash LSH on character 3-grams of the titles that stage 1 missed.
Then: benchmark LSH against a year-blocked Levenshtein baseline on a sample,
and write a 100-row sample for hand-labelling precision.

Run:  spark-submit ... /code/match_titles.py
"""
import time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH

HD = "hdfs://namenode:9000"
spark = (SparkSession.builder.appName("match-titles")
         .config("spark.hadoop.dfs.replication", "2").getOrCreate())

def normalise(col):
    c = F.lower(col)
    c = F.regexp_replace(c, r"\(.*?\)", " ")         # drop "(Director's Cut)"
    c = F.regexp_replace(c, r"&", " and ")
    c = F.regexp_replace(c, r"^(the|a|an) ", "")      # drop leading article
    c = F.regexp_replace(c, r"[^a-z0-9 ]", " ")       # punctuation -> space
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

# ---------- Netflix titles: movie_id,year,title (title may contain commas, file is latin-1)
raw = (spark.read.option("encoding", "ISO-8859-1").option("sep", "\u0001")
       .csv(f"{HD}/netflix/meta/movie_titles.csv"))
nf = (raw.select(F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 1).cast("int").alias("movie_id"),
                 F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 2).alias("year_s"),
                 F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 3).alias("nf_title"))
      .withColumn("nf_year", F.when(F.col("year_s") != "NULL", F.col("year_s").cast("int")))
      .drop("year_s")
      .withColumn("nf_norm", normalise(F.col("nf_title"))))
n_total = nf.count()
nf_noyear = nf.filter(F.col("nf_year").isNull()).count()
nf = nf.filter(F.col("nf_year").isNotNull() & (F.length("nf_norm") > 0)).cache()

# ---------- TMDB catalogue
tm = (spark.read.json(f"{HD}/netflix/meta/tmdb_movies.jsonl")
      .withColumn("tm_year", F.substring("release_date", 1, 4).cast("int"))
      .filter(F.col("tm_year").isNotNull())
      .withColumn("tm_norm", normalise(F.col("title")))
      .filter(F.length("tm_norm") > 0)
      .select("tmdb_id", F.col("title").alias("tm_title"), "tm_norm", "tm_year",
              "overview", "genres", "poster_path",
              F.coalesce("vote_count", F.lit(0)).alias("vote_count")).cache())
n_tmdb = tm.count()

def best_per_movie(df, order_cols):
    w = Window.partitionBy("movie_id").orderBy(*order_cols)
    return df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

# ---------- Stage 1: exact normalised title, year within +-1
t0 = time.time()
cand = nf.join(tm, (nf.nf_norm == tm.tm_norm) & (F.abs(nf.nf_year - tm.tm_year) <= 1))
exact = best_per_movie(cand.withColumn("dist", F.lit(0.0)).withColumn("method", F.lit("exact")),
                       [F.abs(F.col("nf_year") - F.col("tm_year")), F.col("vote_count").desc()])
exact = exact.cache(); n_exact = exact.count()
t_exact = time.time() - t0

# ---------- Stage 2: LSH on the rest
def shingle(df, col):
    padded = F.concat(F.lit("##"), F.col(col), F.lit("##"))
    d = df.withColumn("chars", F.array_remove(F.split(padded, ""), ""))
    d = NGram(n=3, inputCol="chars", outputCol="grams").transform(d)
    return HashingTF(inputCol="grams", outputCol="feat", numFeatures=1 << 18, binary=True).transform(d)

def lsh_match(nf_df, tm_df, threshold=0.5):
    a, b = shingle(nf_df, "nf_norm"), shingle(tm_df, "tm_norm")
    model = MinHashLSH(inputCol="feat", outputCol="h", numHashTables=5).fit(a)
    j = model.approxSimilarityJoin(a, b, threshold, distCol="dist")
    j = j.select("datasetA.movie_id", "datasetA.nf_title", "datasetA.nf_norm", "datasetA.nf_year",
                 "datasetB.tmdb_id", "datasetB.tm_title", "datasetB.tm_norm", "datasetB.tm_year",
                 "datasetB.overview", "datasetB.genres", "datasetB.poster_path",
                 "datasetB.vote_count", "dist")
    j = j.filter(F.abs(F.col("nf_year") - F.col("tm_year")) <= 1)        # year blocking
    return best_per_movie(j.withColumn("method", F.lit("lsh")),
                          [F.col("dist"), F.abs(F.col("nf_year") - F.col("tm_year")),
                           F.col("vote_count").desc()])

t0 = time.time()
rest = nf.join(exact.select("movie_id"), "movie_id", "left_anti")
fuzzy = lsh_match(rest, tm).cache(); n_fuzzy = fuzzy.count()
t_fuzzy = time.time() - t0

cols = ["movie_id", "nf_title", "nf_year", "tmdb_id", "tm_title", "tm_year",
        "dist", "method", "overview", "genres", "poster_path"]
matches = exact.select(*cols).unionByName(fuzzy.select(*cols))
matches.write.mode("overwrite").parquet(f"{HD}/netflix/meta/matches")

# ---------- Hand-labelling sample (100 random matches, fuzzy ones oversampled by taking all methods)
(spark.read.parquet(f"{HD}/netflix/meta/matches")
   .select("movie_id", "nf_title", "nf_year", "tm_title", "tm_year", "method", "dist")
   .orderBy(F.rand(7)).limit(100).withColumn("correct_yes_no", F.lit(""))
   .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{HD}/netflix/meta/label_sample"))

# ---------- Benchmark: LSH vs year-blocked Levenshtein on the same 1000-movie sample
sample = nf.orderBy(F.rand(1)).limit(1000).cache(); sample.count()
t0 = time.time()
naive = (sample.join(tm, F.abs(sample.nf_year - tm.tm_year) <= 1)
         .withColumn("lev", F.levenshtein("nf_norm", "tm_norm") /
                            F.greatest(F.length("nf_norm"), F.length("tm_norm")))
         .filter("lev <= 0.3"))
n_naive_pairs = naive.count(); t_naive = time.time() - t0
t0 = time.time(); n_lsh_sample = lsh_match(sample, tm).count(); t_lsh = time.time() - t0

n_matched = n_exact + n_fuzzy
print("\n========== STEP 2 RESULTS ==========")
print(f"Netflix movies total            : {n_total}")
print(f"  without release year (skipped): {nf_noyear}")
print(f"TMDB catalogue size             : {n_tmdb}")
print(f"Exact matches                   : {n_exact}  ({t_exact:.0f}s)")
print(f"LSH fuzzy matches               : {n_fuzzy}  ({t_fuzzy:.0f}s)")
print(f"TOTAL matched                   : {n_matched}  = {100*n_matched/n_total:.1f}% of all Netflix movies")
print(f"Benchmark (1000-movie sample)   : year-blocked Levenshtein {t_naive:.0f}s vs LSH {t_lsh:.0f}s")
print(f"   pairs kept by Levenshtein<=0.3: {n_naive_pairs}; movies matched by LSH: {n_lsh_sample}")
print("====================================")
spark.stop()
