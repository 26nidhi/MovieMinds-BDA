import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import table, FACTS

def render():
    st.title("Performance lab")
    st.caption("Single laptop, Docker cluster sharing 4 CPUs. Timings are noisy: the same job varied by up to 2-3x between runs, "
               "so read them as rough sizes, not exact ratios.")

    cp = table("csv_vs_parquet")
    if cp is not None:
        st.subheader("CSV vs Parquet (median of 3 runs)")
        st.plotly_chart(px.bar(cp, x="query", y="median_seconds", color="format", barmode="group", text="median_seconds",
                               labels={"median_seconds": "seconds"}), width="stretch")
        st.caption("The year filter is fast on Parquet mainly because the data is partitioned by year (Spark skips the other folders). "
                   "A CSV written with the same partitions would also skip them, so that row mixes format and partitioning.")

    ol = table("optimization_lab")
    if ol is not None:
        st.subheader("Optimisation lab (median of 3 runs each)")
        ol = ol.assign(group=ol["experiment"])
        tabs = st.tabs(["Caching", "Join strategy", "AQE", "Bucketing"])
        for tab, key in zip(tabs, ["cache", "join", "aqe", "bucketing"]):
            with tab:
                d = ol[ol["experiment"].str.startswith(key)]
                if d.empty:
                    st.info("Not run (add the word 'bucketing' when running optimization_lab.py)." if key == "bucketing" else "No rows.")
                    continue
                st.plotly_chart(px.bar(d, x="variant", y="median_s", color="experiment", text="median_s", labels={"median_s": "median seconds"}),
                                width="stretch")
                st.dataframe(d[["experiment", "variant", "first_run_s", "median_s", "detail"]], width="stretch")
        note = {"cache": "Caching the full data did not pay off: building the cache cost far more than one plain scan.",
                "join": "Broadcast was fastest in our runs, but the spread inside one strategy is as large as the gaps between strategies.",
                "aqe": "AQE mattered most with many small shuffle partitions (2000), where it merged them.",
                "bucketing": "Bucketing removes shuffle steps from the plan but costs a one-time write; it pays off only if the same join is repeated."}
        with st.expander("What the experiments showed"):
            for k, v in note.items():
                st.markdown(f"- **{k}:** {v}")
        sc = table("skew_check")
        if sc is not None:
            st.subheader("Is there key skew?")
            st.dataframe(sc, width="stretch")
            st.caption("A key must be several times larger than an average shuffle partition before skew handling pays off. Neither movie nor user keys are, so no skew fix was needed.")

    st.subheader("Spark vs Pandas (Pandas capped at 2 GB)")
    sp = FACTS["spark_vs_pandas"]
    spark = pd.DataFrame(sp["spark_4cores"]).assign(system="Spark (4 cores)")
    pdf = pd.DataFrame(sp["pandas_2gb"])
    fig = px.line(spark, x="rows", y="seconds", markers=True, color="system", labels={"rows": "ratings processed", "seconds": "seconds (load + 3 queries)"})
    ok = pdf[pdf["status"] == "ok"]
    fig.add_trace(go.Scatter(x=ok["rows"], y=ok["seconds"], mode="lines+markers", name="Pandas (2 GB limit)"))
    dead = pdf[pdf["status"] != "ok"]
    if not dead.empty:
        fig.add_trace(go.Scatter(x=dead["rows"], y=[0] * len(dead), mode="markers", marker=dict(symbol="x", size=14, color="red"), name="Pandas killed (out of memory)"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Pandas read only 3 columns and loaded the data once; Spark re-read the files for each query. "
               "Pandas finished 13M rows (about 721 MB peak) and was killed at 57M and 100M. That shows the 2 GB limit, not that Pandas could never do it on a bigger machine.")

    st.subheader("Does adding cores help? (all 100M ratings, 3 queries)")
    rows = [(f"{c} core" + ("s" if c != "1" else ""), t) for c, ts in FACTS["scaling_full_data_runs"].items() for t in ts]
    sdf = pd.DataFrame(rows, columns=["cores", "seconds"])
    st.plotly_chart(px.strip(sdf, x="cores", y="seconds", labels={"seconds": "seconds per run"}), width="stretch")
    med = sdf.groupby("cores")["seconds"].median().round(1).to_dict()
    st.caption(f"Medians: {med}. Every run is plotted. The ranges overlap, so there is no measurable speed-up from 1 to 4 cores here: "
               "the 'nodes' share one laptop's CPUs and disk. Real horizontal scaling needs separate machines. The 117 s point is an outlier we could not explain.")

    bf = table("matching_benchmark_full", kind="json")
    if bf is not None:
        st.subheader("Matching all movies with LSH")
        st.dataframe(bf, width="stretch")
