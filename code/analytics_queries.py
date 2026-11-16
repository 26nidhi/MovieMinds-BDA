"""
Step 3A: Spark SQL analytics over all 100M ratings.
Every query is plain Spark SQL on the partitioned Parquet table. Each result is a small
aggregate saved to /netflix/results/<name> (the dashboard reads these, not the big table),
and every query is timed so we can show job timings on the dashboard.

Run:  spark-submit ... /code/analytics_queries.py
"""
import sys, time
from pyspark.sql import SparkSession, functions as F

HD = "hdfs://namenode:9000"
MIN_N = 1000          # a movie needs at least this many ratings to be called "polarizing"
spark = (SparkSession.builder.appName("analytics-queries")
         .config("spark.hadoop.dfs.replication", "2")
         .config("spark.sql.shuffle.partitions", "64")
         .getOrCreate())

# ---------- views
spark.read.parquet(f"{HD}/netflix/parquet/ratings").createOrReplaceTempView("ratings")
spark.read.parquet(f"{HD}/netflix/meta/matches").select("movie_id", "tmdb_id", "genres").createOrReplaceTempView("matches")
raw = (spark.read.option("encoding", "ISO-8859-1").option("sep", "\u0001").csv(f"{HD}/netflix/meta/movie_titles.csv"))
(raw.select(F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 1).cast("int").alias("movie_id"),
            F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 2).alias("year_s"),
            F.regexp_extract("_c0", r"^(\d+),([^,]*),(.*)$", 3).alias("title"))
    .withColumn("release_year", F.when(F.col("year_s") != "NULL", F.col("year_s").cast("int")))
    .drop("year_s").createOrReplaceTempView("titles"))

timings = []
def run(name, sql, rows_to_show=0):
    """Run one SQL query, save the result as Parquet, record the time."""
    t0 = time.time()
    df = spark.sql(sql)
    out = f"{HD}/netflix/results/{name}"
    df.coalesce(1).write.mode("overwrite").parquet(out)
    n = spark.read.parquet(out).count()
    dt = time.time() - t0
    timings.append((name, round(dt, 1), n))
    print(f"[{name}] {dt:.0f}s, {n} rows", flush=True)
    if rows_to_show:
        spark.read.parquet(out).show(rows_to_show, truncate=False)

# 1. Volume over time (reads partition folders only: year, month)
run("ratings_by_month", """
    SELECT year, month, COUNT(*) AS n_ratings, ROUND(AVG(rating), 4) AS avg_rating
    FROM ratings GROUP BY year, month ORDER BY year, month""")

# 2. Rating drift: do people rate higher or lower over the years?
run("drift_by_year", """
    SELECT year, COUNT(*) AS n_ratings, ROUND(AVG(rating), 4) AS avg_rating,
           ROUND(STDDEV(rating), 4) AS sd_rating,
           ROUND(AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END), 4) AS share_1star,
           ROUND(AVG(CASE WHEN rating = 5 THEN 1.0 ELSE 0.0 END), 4) AS share_5star
    FROM ratings GROUP BY year ORDER BY year""", 10)

# 3. Overall rating histogram
run("rating_histogram", """
    SELECT rating, COUNT(*) AS n FROM ratings GROUP BY rating ORDER BY rating""", 5)

# 4. Per-movie statistics for ALL movies (reused later: polarization, cold-start buckets)
run("movie_stats", """
    SELECT movie_id, COUNT(*) AS n_ratings, ROUND(AVG(rating), 4) AS avg_rating,
           ROUND(STDDEV(rating), 4) AS sd_rating,
           ROUND(AVG(CASE WHEN rating = 1 THEN 1.0 ELSE 0.0 END), 4) AS share_1star,
           ROUND(AVG(CASE WHEN rating = 5 THEN 1.0 ELSE 0.0 END), 4) AS share_5star,
           MIN(date) AS first_rated, MAX(date) AS last_rated
    FROM ratings GROUP BY movie_id""")
spark.read.parquet(f"{HD}/netflix/results/movie_stats").createOrReplaceTempView("movie_stats")

# 5. Most polarizing movies: high spread of ratings (1-star and 5-star both common)
run("polarizing_movies", f"""
    SELECT s.movie_id, t.title, t.release_year, s.n_ratings, s.avg_rating, s.sd_rating,
           s.share_1star, s.share_5star
    FROM movie_stats s JOIN titles t ON s.movie_id = t.movie_id
    WHERE s.n_ratings >= {MIN_N}
    ORDER BY s.sd_rating DESC LIMIT 50""", 10)

# 6. Most rated / best rated movies
run("top_movies", f"""
    SELECT s.movie_id, t.title, t.release_year, s.n_ratings, s.avg_rating
    FROM movie_stats s JOIN titles t ON s.movie_id = t.movie_id
    ORDER BY s.n_ratings DESC LIMIT 50""")

# 7. Per-user statistics for ALL users (480k rows, small)
run("user_stats", """
    SELECT user_id, COUNT(*) AS n_ratings, ROUND(AVG(rating), 4) AS avg_rating,
           MIN(date) AS first_rating, MAX(date) AS last_rating,
           APPROX_COUNT_DISTINCT(date) AS active_days
    FROM ratings GROUP BY user_id""")
spark.read.parquet(f"{HD}/netflix/results/user_stats").createOrReplaceTempView("user_stats")

# 8. Most active users
run("top_users", """
    SELECT * FROM user_stats ORDER BY n_ratings DESC LIMIT 50""", 5)

# 9. How lopsided is user activity? (this is the skew we handle in the optimisation lab)
q = spark.table("user_stats").stat.approxQuantile("n_ratings", [0.5, 0.9, 0.99], 0.001)
tot = spark.table("user_stats").agg(F.sum("n_ratings"), F.count("*")).first()
top1 = spark.table("user_stats").filter(F.col("n_ratings") >= q[2]).agg(F.sum("n_ratings")).first()[0]
skew = [("users", float(tot[1])), ("ratings", float(tot[0])), ("median_ratings_per_user", q[0]),
        ("p90_ratings_per_user", q[1]), ("p99_ratings_per_user", q[2]),
        ("share_of_ratings_from_top_1pct_users", round(top1 / tot[0], 4))]
spark.createDataFrame(skew, "metric string, value double").coalesce(1) \
     .write.mode("overwrite").parquet(f"{HD}/netflix/results/user_activity_skew")
print("user activity:", skew, flush=True)

# 10. Genre trends (only movies matched to TMDB; broadcast join with the small matches table)
run("genre_by_year", """
    SELECT /*+ BROADCAST(m) */ g AS genre, r.year, COUNT(*) AS n_ratings,
           ROUND(AVG(r.rating), 4) AS avg_rating
    FROM ratings r JOIN matches m ON r.movie_id = m.movie_id
    LATERAL VIEW EXPLODE(m.genres) x AS g
    WHERE g IS NOT NULL
    GROUP BY g, r.year ORDER BY g, r.year""", 10)

# 11. Genre summary
run("genre_summary", """
    SELECT genre, SUM(n_ratings) AS n_ratings,
           ROUND(SUM(n_ratings * avg_rating) / SUM(n_ratings), 4) AS avg_rating
    FROM parquet.`""" + f"{HD}/netflix/results/genre_by_year" + """`
    GROUP BY genre ORDER BY n_ratings DESC""", 20)

# 12. Cold-start: how many movies have few ratings? (motivates the adaptive hybrid)
run("movie_popularity_buckets", """
    SELECT CASE WHEN n_ratings < 10 THEN '1) <10'
                WHEN n_ratings < 100 THEN '2) 10-99'
                WHEN n_ratings < 1000 THEN '3) 100-999'
                WHEN n_ratings < 10000 THEN '4) 1k-9.9k'
                ELSE '5) 10k+' END AS bucket,
           COUNT(*) AS n_movies, SUM(n_ratings) AS n_ratings
    FROM movie_stats GROUP BY 1 ORDER BY 1""", 10)

spark.createDataFrame(timings, "query string, seconds double, rows_out long").coalesce(1) \
     .write.mode("overwrite").parquet(f"{HD}/netflix/results/query_timings")
print("\n========== QUERY TIMINGS ==========")
for n, s, r in timings:
    print(f"{n:28}{s:>8}s{r:>10} rows")
print("===================================")
spark.stop()
