"""
Step 1: Parse raw Netflix Prize files with Spark and save as partitioned Parquet.

Raw format (combined_data_1.txt ... combined_data_4.txt):
    1:                      <- movie id line
    1488844,3,2005-09-06    <- user_id,rating,date
    822109,5,2005-05-13
    2:                      <- next movie

Method: read lines, mark movie-id lines (end with ":"), forward-fill the id
down to the rating lines using a window function. Fully distributed, no collect().
Parsing runs ONCE (Parquet write). The CSV copy and the row count read that Parquet.

Usage: parse_netflix.py [fraction]     e.g. 0.01 for a 1% test, no argument = full data
"""
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

spark = (SparkSession.builder.appName("netflix-parse")
         .config("spark.sql.shuffle.partitions", "200")
         .config("spark.hadoop.dfs.replication", "2")   # we only have 2 datanodes
         .getOrCreate())

# 1) Read lines and forward-fill the movie id (per file)
lines = spark.read.text("hdfs://namenode:9000/netflix/raw/combined_data_*.txt")
lines = (lines.withColumn("file", F.input_file_name())
              .withColumn("row_id", F.monotonically_increasing_id()))

is_movie = F.col("value").endswith(":")
lines = lines.withColumn("movie_id",
    F.when(is_movie, F.regexp_replace("value", ":", "").cast("int")))

w = (Window.partitionBy("file").orderBy("row_id")
     .rowsBetween(Window.unboundedPreceding, 0))
lines = lines.withColumn("movie_id", F.last("movie_id", ignorenulls=True).over(w))

ratings = (lines.filter(~is_movie)
    .withColumn("p", F.split("value", ","))
    .select("movie_id",
            F.col("p")[0].cast("int").alias("user_id"),
            F.col("p")[1].cast("int").alias("rating"),
            F.to_date(F.col("p")[2]).alias("date"))
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date")))

if FRAC < 1.0:
    ratings = ratings.sample(False, FRAC, seed=42)

# 2) Parse ONCE: write Parquet partitioned by year and month
(ratings.repartition("year", "month")
    .write.mode("overwrite")
    .partitionBy("year", "month")
    .parquet("hdfs://namenode:9000/netflix/parquet/ratings"))

# 3) Everything below reads the small Parquet output, not the raw files
pq = spark.read.parquet("hdfs://namenode:9000/netflix/parquet/ratings")
pq.write.mode("overwrite").option("header", True).csv("hdfs://namenode:9000/netflix/csv/ratings")
print("rows:", pq.count())
spark.stop()