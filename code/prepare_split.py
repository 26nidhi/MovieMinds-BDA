"""
Step 4A: Temporal train / validation / test split on a sample of USERS.

Why temporal: a recommender is used to predict the FUTURE, so we train on the past and test on
the later ratings. A random split would leak future taste into training and flatter the RMSE.

  train = oldest 80% of ratings by date, val = next 10% (used only to tune ALS),
  test  = newest 10%. Final models are trained on train+val and judged on test.

We keep all ratings of a random subset of users (hash of user_id), so each user's history stays
intact. user_pct = percentage of users to keep (100 = everything).

Usage: spark-submit ... prepare_split.py [user_pct]      default 10
"""
import sys
from pyspark.sql import SparkSession, functions as F

HD = "hdfs://namenode:9000"
user_pct = int(sys.argv[1]) if len(sys.argv) > 1 else 10
spark = (SparkSession.builder.appName("prepare-split")
         .config("spark.hadoop.dfs.replication", "2").config("spark.sql.shuffle.partitions", "64").getOrCreate())

r = spark.read.parquet(f"{HD}/netflix/parquet/ratings").select("user_id", "movie_id", "rating", "date")
if user_pct < 100:
    r = r.filter((F.abs(F.hash("user_id")) % 100) < user_pct)
r = r.withColumn("day", F.datediff("date", F.lit("1998-01-01"))).cache()
n = r.count()
q80, q90 = r.stat.approxQuantile("day", [0.8, 0.9], 0.001)

train = r.filter(F.col("day") < q80)
val = r.filter((F.col("day") >= q80) & (F.col("day") < q90))
test = r.filter(F.col("day") >= q90)
for name, df in (("train", train), ("val", val), ("test", test)):
    df.select("user_id", "movie_id", "rating", "date", "day").write.mode("overwrite").parquet(f"{HD}/netflix/model/{name}")

tr = spark.read.parquet(f"{HD}/netflix/model/train"); te = spark.read.parquet(f"{HD}/netflix/model/test")
va = spark.read.parquet(f"{HD}/netflix/model/val")
n_tr, n_va, n_te = tr.count(), va.count(), te.count()
seen_u = tr.select("user_id").distinct(); seen_m = tr.select("movie_id").distinct()
unseen_user = te.join(seen_u, "user_id", "left_anti").count()
unseen_movie = te.join(seen_m, "movie_id", "left_anti").count()
d = lambda x: spark.range(1).select(F.date_add(F.lit("1998-01-01"), int(x)).alias("d")).first()[0]
print("\n========== SPLIT ==========")
print(f"users kept: {user_pct}%   total ratings: {n:,}")
print(f"train {n_tr:,} ({100*n_tr/n:.0f}%)   val {n_va:,}   test {n_te:,}")
print(f"val starts {d(q80)}, test starts {d(q90)}")
print(f"test ratings whose user is NOT in train : {unseen_user:,} ({100*unseen_user/n_te:.1f}%)")
print(f"test ratings whose movie is NOT in train: {unseen_movie:,} ({100*unseen_movie/n_te:.1f}%)")
print("===========================")
spark.createDataFrame([(user_pct, n, n_tr, n_va, n_te, unseen_user, unseen_movie)],
    "user_pct int, total long, train long, val long, test long, test_unseen_user long, test_unseen_movie long") \
    .coalesce(1).write.mode("overwrite").parquet(f"{HD}/netflix/results/split_summary")
spark.stop()
