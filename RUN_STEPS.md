# Step 1: How to run

## 0. Get the data
Download "Netflix Prize data" from Kaggle. Copy combined_data_1.txt ... combined_data_4.txt into ./data

## 1. Start the cluster
    docker compose up -d
Check: HDFS UI  http://localhost:9870  (2 live datanodes)
       Spark UI http://localhost:8080  (2 workers)

## 2. Load raw files into HDFS
    docker exec namenode hdfs dfs -mkdir -p /netflix/raw
    docker exec namenode hdfs dfs -put /data/combined_data_*.txt /netflix/raw/
    docker exec namenode hdfs dfs -ls -h /netflix/raw
    docker exec namenode hdfs fsck /netflix/raw -files -blocks     # shows blocks + replicas

## 3. Test on 1% first
    docker exec spark-master spark-submit --master spark://spark-master:7077 /code/parse_netflix.py 0.01

## 4. Full run
    docker exec spark-master spark-submit --master spark://spark-master:7077 /code/parse_netflix.py

## 5. Compare formats
    docker exec namenode hdfs dfs -du -s -h /netflix/csv /netflix/parquet
    docker exec spark-master spark-submit --master spark://spark-master:7077 /code/compare_csv_parquet.py

Save: screenshots of HDFS UI + Spark UI, the du sizes, and the timings. These go in your dashboard (Pipeline page).
