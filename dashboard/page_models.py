import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lib import table, FACTS

def render():
    st.title("Model results")
    st.caption("Sample: 2% of users, split by date (train 80% / validation 10% / test newest 10%). Models are tuned on validation and judged once on test. "
               "Lower RMSE is better.")

    ab = table("ablation_rmse")
    if ab is not None:
        st.subheader("Ablation: adding one idea at a time (test RMSE)")
        d = ab.copy()
        fig = go.Figure()
        dd = d.dropna(subset=["delta_vs_prev"])
        fig.add_trace(go.Scatter(x=dd["delta_vs_prev"], y=dd["model"], mode="markers", marker=dict(size=11),
                                 error_x=dict(type="data", symmetric=False, array=dd["ci_high"] - dd["delta_vs_prev"],
                                              arrayminus=dd["delta_vs_prev"] - dd["ci_low"])))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(xaxis_title="change in RMSE vs the previous row (negative = better), 95% CI over users", yaxis=dict(autorange="reversed"), height=330)
        st.plotly_chart(fig, width="stretch")
        d["95% CI"] = [("" if pd.isna(a) else f"[{a:+.4f}, {b:+.4f}]") for a, b in zip(d["ci_low"], d["ci_high"])]
        d["excludes 0"] = [("" if pd.isna(a) else ("yes" if (a > 0 or b < 0) else "no")) for a, b in zip(d["ci_low"], d["ci_high"])]
        st.dataframe(d[["model", "rmse", "delta_vs_prev", "95% CI", "excludes 0"]].round(4), width="stretch", hide_index=True)
        st.caption("Intervals resample users (ratings from one user are correlated). ALS alone ties the baseline; the gain comes from combining it with content, "
                   "then from time decay and context-dependent weights. The decay length and weight rule were chosen on validation data only.")

    left, right = st.columns(2)
    with left:
        ma, gr = table("model_metrics_als"), table("als_grid")
        if ma is not None:
            st.subheader("Baselines and ALS")
            st.dataframe(ma.round(4), width="stretch", hide_index=True)
            st.caption("'predictable pairs' = test ratings whose user and movie were seen in training. The bias baseline on the same pairs is listed for a like-for-like comparison.")
        if gr is not None:
            st.markdown("**ALS grid (validation RMSE)**")
            st.dataframe(gr.round(4), width="stretch", hide_index=True)
    with right:
        mc, cb = table("model_metrics_content"), table("content_by_bucket")
        if mc is not None:
            st.subheader("Content-based model")
            st.dataframe(mc.round(4), width="stretch", hide_index=True)
        if cb is not None:
            m = cb.melt(id_vars=["bucket", "test_pairs", "content_covered"], value_vars=["rmse_bias", "rmse_content"], var_name="model", value_name="rmse")
            st.plotly_chart(px.bar(m, x="bucket", y="rmse", color="model", barmode="group", title="RMSE on the pairs the content model covers, by movie popularity"), width="stretch")
            st.caption("The content model only covers TMDB-matched movies whose neighbours the user has rated. The gain is small and similar in every bucket.")

    rk = table("ranking_metrics")
    if rk is not None:
        st.subheader("Top-10 recommendations (precision / NDCG on the test period)")
        n_all, n_pool = FACTS["ranking"]["all_candidates"], FACTS["ranking"]["pool"]
        rk = rk.assign(distinct_movies=[round(c * (n_pool if "pool" in m.lower() else n_all)) for m, c in zip(rk["method"], rk["coverage"])])
        m = rk.melt(id_vars="method", value_vars=["precision_at_10", "ndcg_at_10"], var_name="metric", value_name="value")
        st.plotly_chart(px.bar(m, x="method", y="value", color="metric", barmode="group"), width="stretch")
        st.dataframe(rk[["method", "precision_at_10", "recall_at_10", "ndcg_at_10", "distinct_movies", "users"]].round(4), width="stretch", hide_index=True)
        st.caption("A hit = the user rated the movie 4 or more in the test period. Values are small because users rate few movies in a 2.5-month window. "
                   "ALS over ALL movies fails (it ranks obscure movies first); restricted to the 1000 most-rated movies it is the best method. "
                   "'distinct_movies' = how many different movies were recommended across all users (coverage x candidate count).")

    rp = table("rerank_polarization")
    if rp is not None:
        st.subheader("Polarization-aware re-ranking (safe vs risky picks)")
        base = st.selectbox("Base ranker", sorted(rp["base"].unique()))
        d = rp[rp["base"] == base].sort_values("lam")
        a, b, c = st.columns(3)
        a.plotly_chart(px.line(d, x="lam", y="p10", markers=True, title="Precision@10"), width="stretch")
        b.plotly_chart(px.line(d, x="lam", y="distinct_movies", markers=True, title="Distinct movies recommended"), width="stretch")
        c.plotly_chart(px.line(d, x="lam", y="bad_rate", markers=True, title="Bad rate (rated <= 2 stars)"), width="stretch")
        d = d.assign(**{"change in P@10 vs lambda=0": [("" if pd.isna(x) else f"{x:+.4f} [{lo:+.4f}, {hi:+.4f}]") for x, lo, hi in zip(d["d_p10"], d["d_p10_lo"], d["d_p10_hi"])]})
        st.dataframe(d[["lam", "p10", "ndcg10", "distinct_movies", "mean_polarization", "bad_rate", "change in P@10 vs lambda=0"]].round(4), width="stretch", hide_index=True)
        st.caption("lambda > 0 = safe picks, lambda < 0 = risky picks. Pushing polarizing movies up (risky) clearly lowers precision. Pushing them down (safe) leaves precision "
                   "unchanged but narrows variety. Bad rate counts only recommended movies the user later rated, so it rests on few events and is noisy.")

    hy = table("hybrid_rmse")
    if hy is not None:
        with st.expander("Blend weight sensitivity (intervals treat pairs as independent, so they are narrower than the user-level ones above)"):
            st.dataframe(hy.round(5), width="stretch", hide_index=True)
