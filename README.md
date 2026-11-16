# Movie Recommendation System Using Big Data Analytics (Netflix Prize dataset)

CS404 Big Data Analytics course project. It builds a movie recommender on the Netflix Prize ratings
(about 100 million ratings, 480k users, 17,770 movies, roughly 2 GB of raw text) with a Hadoop HDFS +
Apache Spark pipeline, and a Streamlit dashboard that shows every stage.

> Everything was measured on one laptop running a small Docker cluster (2 HDFS datanodes, 2 Spark workers,
> 4 CPUs shared by all). Timings are noisy; read them as rough sizes. Model experiments use a 2% sample of users.

## Pipeline

Raw files -> HDFS -> Spark parse -> Parquet (partitioned by year/month) -> TMDB matching -> Spark SQL analytics ->
models (bias, ALS, content-based, blends) -> dashboard

| Step | What | Scripts (`code/`) |
|---|---|---|
| 1 | Ingest and parse: HDFS, window-function forward-fill of movie ids, Parquet, CSV-vs-Parquet | `parse_netflix.py`, `compare_csv_parquet.py` |
| 2 | Match Netflix titles to TMDB: exact title+year, then MinHash LSH | `fetch_tmdb.py`, `match_titles.py`, `benchmark_matching.py`, `score_labels.py` |
| 3 | Spark SQL analytics, optimisation lab, Spark vs Pandas | `analytics_queries.py`, `optimization_lab.py`, `spark_bench.py`, `pandas_bench.py` |
| 4 | Temporal split, baselines, ALS, content kNN, hybrid, ranking metrics, bootstrap CIs | `prepare_split.py`, `als_train.py`, `content_model.py`, `hybrid_eval.py`, `bootstrap_ci.py` |
| 5 | Additions: time decay, adaptive blend weight, polarization re-ranking | `adaptive_blend.py`, `rerank_polarization.py` |
| 6 | Dashboard and its data export | `dashboard/`, `export_recommendations.py`, `fetch_missing_posters.py` |

## Selected results (measured; see the dashboard for all of them)

- Ratings parsed: 100,480,507. Parquet is 474.5 MB against 3.1 GB of CSV (about 6.7x smaller).
- Netflix-to-TMDB matching: 11,419 of 17,770 movies matched (64.3%); precision from a hand-labelled sample of 100: **<fill in from score_labels.py>**.
- Spark finished all 100M ratings; Pandas capped at 2 GB finished 13M rows and was killed at 57M and 100M.
- Adding cores did not speed Spark up on this single laptop (the "nodes" share the same CPUs and disk).
- Test RMSE (newest 10% of ratings, 2% of users): bias baseline 0.9828, ALS 0.9830 (a tie), ALS + content 0.9662,
  + time decay 0.9626, + adaptive weight 0.9557 (-0.0270 vs baseline, 95% interval over users [-0.0304, -0.0240]).
- Polarization re-ranking did not improve accuracy; it works as a safe/risky control (risky picks lower precision).

## Limitations

- The data stops in 2005, so "current" movies are not in it.
- Model results are on a 2% sample of users; 45% of test ratings come from users with no earlier ratings.
- Content features exist only for movies matched to TMDB (TV seasons and box sets are not matched).
- Single runs on a shared laptop: timing differences smaller than about 2x should not be over-read.

## Running it

1. Start the cluster: `docker compose up -d` (needs Docker Desktop). Download the Netflix Prize files from Kaggle
   into `data/` (not included in this repository).
2. Run the Spark jobs in step order with `spark-submit`, as in the step guides. Each script has its usage in its header.
3. Copy the result tables out of HDFS into `data/dashboard_results`, then
   `python -m pip install -r dashboard/requirements.txt` and `python -m streamlit run dashboard/app.py`.

TMDB: this product uses the TMDB API but is not endorsed or certified by TMDB. Set your own key in the
`TMDB_API_KEY` environment variable; no key is stored in this repository.
