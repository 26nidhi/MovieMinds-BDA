"""
Step 2c: Fair benchmark, LSH vs year-blocked Levenshtein, at growing sample sizes.

Fixes the earlier unfair test:
  * TMDB features/hashes are built ONCE and cached; that cost is timed separately.
  * Only the matching join itself is timed for each sample size.
  * Quality is compared on the same 1000 movies (how often both pick the same TMDB movie).
Results are also saved to /netflix/meta/benchmark for the dashboard.
"""
import sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH

HD = "hdfs://namenode:9000"
LSH_ONLY = "lsh_only" in sys.argv          # python script lsh_only -> skip the slow naive runs
NAIVE_SIZES = [] if LSH_ONLY else [500, 1000, 2000]   # naive gets slow fast, so it stops early
LSH_SIZES = [500, 1000, 2000, 5000, 17770] if not LSH_ONLY else [17770]  # 17770 = full list
QUALITY_N = 1000

spark = (SparkSession.builder.appName("benchmark-matching")
         .config("spark.hadoop.dfs.replication", "2").getOrCreate())

def normalise(col):
    c = F.lower(col)
    c = F.regexp_replace(c, r"\(.*?\)", " ")
    c = F.regexp_replace(c, r"&", " and ")
    c = F.regexp_replace(c, r"^(the|a|an) ", "")
    c = F.regexp_replace(c, r"[^a-z0-9 ]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

raw = (spark.read.option("encoding", "ISO-8859-1").option("sep", "\u0001")
       .csv(f"{HD}/netflix/meta/movie_titles.csv"))
nf = (raw.select(F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 1).cast("int").alias("movie_id"),
                 F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 2).alias("year_s"),
                 F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 3).alias("nf_title"))
      .withColumn("nf_year", F.when(F.col("year_s") != "NULL", F.col("year_s").cast("int")))
      .withColumn("nf_norm", normalise(F.col("nf_title")))
      .filter(F.col("nf_year").isNotNull() & (F.length("nf_norm") > 0))
      .select("movie_id", "nf_norm", "nf_year"))
tm = (spark.read.json(f"{HD}/netflix/meta/tmdb_movies.jsonl")
      .withColumn("tm_year", F.substring("release_date", 1, 4).cast("int"))
      .filter(F.col("tm_year").isNotNull())
      .withColumn("tm_norm", normalise(F.col("title")))
      .filter(F.length("tm_norm") > 0)
      .select("tmdb_id", "tm_norm", "tm_year",
              F.coalesce("vote_count", F.lit(0)).alias("vote_count")).cache())
n_tmdb = tm.count()
nmax = min(max(LSH_SIZES), nf.count())
pool = nf.orderBy(F.rand(1)).limit(nmax).cache(); pool.count()

def shingle(df, col):
    d = df.withColumn("chars", F.array_remove(F.split(F.concat(F.lit("##"), F.col(col), F.lit("##")), ""), ""))
    d = NGram(n=3, inputCol="chars", outputCol="grams").transform(d)
    return HashingTF(inputCol="grams", outputCol="feat", numFeatures=1 << 18, binary=True).transform(d)

def best(df, order):
    w = Window.partitionBy("movie_id").orderBy(*order)
    return df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

results = []

# ---- one-time LSH build cost (shingle + hash all TMDB titles)
t0 = time.time()
lsh = MinHashLSH(inputCol="feat", outputCol="h", numHashTables=5)
tm_f = shingle(tm, "tm_norm")
model = lsh.fit(tm_f)
tm_h = model.transform(tm_f).select("tmdb_id", "tm_year", "vote_count", "feat", "h").cache()
tm_h.count()
t_build = time.time() - t0
results.append(("lsh_build_tmdb_once", n_tmdb, round(t_build, 1), None))

def run_lsh(sample):
    a = model.transform(shingle(sample, "nf_norm")).cache(); a.count()   # not timed
    t = time.time()
    j = model.approxSimilarityJoin(a, tm_h, 0.5, distCol="dist")
    j = (j.select("datasetA.movie_id", "datasetA.nf_year", "datasetB.tmdb_id",
                  "datasetB.tm_year", "datasetB.vote_count", "dist")
          .filter(F.abs(F.col("nf_year") - F.col("tm_year")) <= 1))
    out = best(j, [F.col("dist"), F.abs(F.col("nf_year") - F.col("tm_year")),
                   F.col("vote_count").desc()]).cache()
    n = out.count()
    return out, n, time.time() - t

def run_naive(sample):
    t = time.time()
    j = (sample.join(tm, F.abs(sample.nf_year - tm.tm_year) <= 1)
         .withColumn("lev", F.levenshtein("nf_norm", "tm_norm") /
                            F.greatest(F.length("nf_norm"), F.length("tm_norm")))
         .filter("lev <= 0.3"))
    out = best(j, [F.col("lev"), F.col("vote_count").desc()]).cache()
    n = out.count()
    return out, n, time.time() - t

keep = {}
for req in sorted(set(NAIVE_SIZES + LSH_SIZES)):
    n = min(req, nmax)          # 17770 requested = "all Netflix movies that have a year"
    s = pool.limit(n).cache(); s.count()
    if req in NAIVE_SIZES:
        o, k, t = run_naive(s); results.append(("naive_levenshtein", n, round(t, 1), k))
        print(f"naive  n={n}: {t:.0f}s, matched {k}", flush=True)
        if n == QUALITY_N: keep["naive"] = o
    if req in LSH_SIZES:
        o, k, t = run_lsh(s); results.append(("lsh_join_only", n, round(t, 1), k))
        print(f"lsh    n={n}: {t:.0f}s, matched {k}", flush=True)
        if n == QUALITY_N: keep["lsh"] = o

print("\n========== BENCHMARK RESULTS ==========")
print(f"TMDB titles: {n_tmdb}; one-time LSH build (shingle+hash TMDB): {t_build:.0f}s")
print(f"{'method':22}{'movies':>8}{'seconds':>10}{'matched':>9}")
for m, n, t, k in results:
    print(f"{m:22}{n:>8}{t:>10}{'' if k is None else k:>9}")
if "naive" in keep and "lsh" in keep:
    nv = keep["naive"].select("movie_id", F.col("tmdb_id").alias("t_naive"))
    ls = keep["lsh"].select("movie_id", F.col("tmdb_id").alias("t_lsh"))
    both = nv.join(ls, "movie_id")
    n_both = both.count(); n_same = both.filter("t_naive = t_lsh").count()
    print(f"\nQuality on the same {QUALITY_N} movies (different cut-offs: Levenshtein<=0.3, Jaccard<=0.5):")
    print(f"  matched by naive: {keep['naive'].count()}, by LSH: {keep['lsh'].count()}, by both: {n_both}")
    print(f"  of those matched by both, same TMDB movie chosen: {n_same} ({100*n_same/max(n_both,1):.1f}%)")
print("=======================================")

(spark.createDataFrame(results, "method string, movies int, seconds double, matched int")
      .coalesce(1).write.mode("overwrite").json(f"{HD}/netflix/meta/" + ("benchmark_full" if LSH_ONLY else "benchmark")))
spark.stop()