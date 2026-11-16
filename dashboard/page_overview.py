import pandas as pd
import plotly.express as px
import streamlit as st
from lib import table, FACTS, fmt_int

def render():
    st.title("Pipeline overview")
    st.caption("Netflix Prize ratings → HDFS → Spark → Parquet → Spark SQL / MLlib → this dashboard. "
               "Everything here was measured on one laptop running a Docker cluster (2 datanodes, 2 Spark workers, 4 CPUs shared by all).")
    s, m = FACTS["storage"], FACTS["matching"]
    by_month, skew, mstats = table("ratings_by_month"), table("user_activity_skew"), table("movie_stats")
    c = st.columns(4)
    if by_month is not None:
        c[0].metric("Ratings", fmt_int(by_month["n_ratings"].sum()))
    if skew is not None:
        c[1].metric("Users", fmt_int(skew.set_index("metric").loc["users", "value"]))
    if mstats is not None:
        c[2].metric("Movies", fmt_int(len(mstats)))
    c[3].metric("Movies matched to TMDB", f"{m['matched']:,} ({100*m['matched']/m['netflix_movies']:.1f}%)")

    st.subheader("Architecture")
    st.markdown(
        f"1. **Ingest:** 4 raw text files ({s['raw_gb']} GB) copied into HDFS: {s['datanodes']} datanodes, replication {s['replication']}, {s['block_mb']} MB blocks.\n"
        "2. **Parse:** Spark reads the lines and copies each movie id down to its rating lines with a window function (nothing is collected to the driver).\n"
        "3. **Store:** Parquet partitioned by year and month.\n"
        "4. **Match:** Netflix titles to TMDB by normalised title and year, then MinHash LSH for the leftovers.\n"
        "5. **Analyse and model:** Spark SQL, ALS, content-based kNN, blends with time decay and context-dependent weights.\n"
        "6. **Serve:** Spark writes small aggregate tables; this app only reads them.")

    left, right = st.columns(2)
    with left:
        st.subheader("Storage format")
        df = pd.DataFrame({"format": ["Raw text", "CSV", "Parquet"], "size_gb": [s["raw_gb"], s["csv_gb"], s["parquet_mb"] / 1024]})
        fig = px.bar(df, x="format", y="size_gb", text=df["size_gb"].map(lambda v: f"{v:.2f} GB"), labels={"size_gb": "GB (one copy)"})
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Parquet is about {s['csv_gb']*1024/s['parquet_mb']:.1f}x smaller than the CSV copy of the same ratings.")
    with right:
        st.subheader("Netflix ↔ TMDB matching")
        df = pd.DataFrame({"stage": ["Exact title + year (±1)", "MinHash LSH fuzzy", "Unmatched"],
                           "movies": [m["exact"], m["fuzzy"], m["netflix_movies"] - m["matched"]]})
        st.plotly_chart(px.bar(df, x="stage", y="movies", text="movies"), width="stretch")
        prec = m.get("precision")
        st.caption("Precision of the matches: " + (f"{100*prec:.1f}% on a hand-labelled sample of 100." if prec else
                   "not measured yet (the 100-row hand-labelling is still to do)."))

    bj = table("matching_benchmark", kind="json")
    if bj is not None:
        st.subheader("Matching speed: LSH vs year-blocked Levenshtein")
        b = bj[bj["method"].isin(["naive_levenshtein", "lsh_join_only"])]
        st.plotly_chart(px.line(b, x="movies", y="seconds", color="method", markers=True,
                                labels={"movies": "Netflix movies matched", "seconds": "seconds (join only)"}), width="stretch")
        st.caption("Each point is one run on a shared laptop. Both grow roughly in a straight line with the number of movies; "
                   "LSH is about 9x cheaper per movie. The one-time LSH index build over the TMDB titles is timed separately.")

    qt = table("query_timings")
    if qt is not None:
        st.subheader("Analytics job timings (Spark SQL over all ratings)")
        qt = qt.sort_values("seconds")
        st.plotly_chart(px.bar(qt, x="seconds", y="query", orientation="h", text="seconds"), width="stretch")

    sp = table("split_summary")
    if sp is not None:
        r = sp.iloc[0]
        st.subheader("Train / validation / test split (by date)")
        st.write(f"Sample: **{int(r['user_pct'])}% of users** ({fmt_int(r['total'])} ratings). Oldest 80% train, next 10% validation, newest 10% test.")
        a, b2, c2 = st.columns(3)
        a.metric("Train", fmt_int(r["train"])); b2.metric("Validation", fmt_int(r["val"])); c2.metric("Test", fmt_int(r["test"]))
        st.info(f"{100*r['test_unseen_user']/r['test']:.1f}% of test ratings come from users with no ratings in the training period, "
                f"and {100*r['test_unseen_movie']/r['test']:.1f}% are for movies not seen in training. No collaborative model can personalise for those users.")
