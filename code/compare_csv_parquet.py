"""
CSV vs Parquet comparison, v2 (fairer than v1).
Changes from v1:
  * CSV is read with an explicit schema, so the slow schema-guessing scan is not hidden in the timing
  * results are not collected to the driver (write to the 'noop' sink instead)
  * every query runs 3 times; we report each run, plus the median
  * a tiny warm-up query runs first so Spark start-up cost is not counted
Run on the full data after parse_netflix.py.
"""
import time, statistics
from pyspark.sql import SparkSession, functions as F

HD = "hdfs://namenode:9000"
spark = SparkSession.builder.appName("csv-vs-parquet-v2").getOrCreate()

csv = (spark.read.option("header", True)
       .schema("movie_id INT, user_id INT, rating INT, date DATE, year INT, month INT")
       .csv(f"{HD}/netflix/csv/ratings"))
pq = spark.read.parquet(f"{HD}/netflix/parquet/ratings")

def timed(df):
    t = time.time(); df.write.format("noop").mode("overwrite").save(); return time.time() - t

queries = {
    "avg rating per movie": lambda d: d.groupBy("movie_id").agg(F.avg("rating")),
    "rating histogram":     lambda d: d.groupBy("rating").count(),
    "count where year=2005": lambda d: d.filter(F.col("year") == 2005).agg(F.count("*")),
}
timed(pq.limit(10)); timed(csv.limit(10))                      # warm-up

rows = []
for qname, q in queries.items():
    for fmt, d in (("CSV", csv), ("Parquet", pq)):
        ts = [timed(q(d)) for _ in range(3)]
        rows.append((qname, fmt, [round(x, 1) for x in ts], round(statistics.median(ts), 1)))
        print(f"{qname:24}{fmt:8} runs={rows[-1][2]}  median={rows[-1][3]}s", flush=True)

print("\n========== CSV vs PARQUET (median of 3) ==========")
for q in queries:
    c = next(r[3] for r in rows if r[0] == q and r[1] == "CSV")
    p = next(r[3] for r in rows if r[0] == q and r[1] == "Parquet")
    print(f"{q:24} CSV {c:>7}s   Parquet {p:>7}s   ratio {c/p:.1f}x")
print("==================================================")
pq.filter(F.col("year") == 2005).explain()
spark.createDataFrame([(a, b, float(d)) for a, b, c, d in rows], "query string, format string, median_seconds double") \
     .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/csv_vs_parquet")
spark.stop()
