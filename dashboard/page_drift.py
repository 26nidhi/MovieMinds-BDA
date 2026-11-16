import pandas as pd
import plotly.express as px
import streamlit as st
from lib import table

USER_ORDER = ["0", "1-19", "20-99", "100-499", "500+"]
MOVIE_ORDER = ["0", "1-9", "10-99", "100-999", "1000+"]
CONF_ORDER = ["none", "<0.5", "0.5-1.5", "1.5-3", "3+"]

def render():
    st.title("Drift explorer")
    st.caption("Taste and rating behaviour change over time. This page shows the drift in the data and how the models use it.")

    dr = table("drift_by_year")
    if dr is not None:
        st.subheader("Average rating and spread by year")
        a, b = st.columns(2)
        a.plotly_chart(px.line(dr, x="year", y="avg_rating", markers=True, title="Average rating"), width="stretch")
        b.plotly_chart(px.line(dr, x="year", y="sd_rating", markers=True, title="Std dev of ratings"), width="stretch")

    gy = table("genre_by_year")
    if gy is not None:
        pick = st.multiselect("Genres", sorted(gy["genre"].unique()), default=sorted(gy["genre"].unique())[:3], key="drift_genres")
        if pick:
            st.plotly_chart(px.line(gy[gy["genre"].isin(pick)], x="year", y="avg_rating", color="genre", markers=True, title="Average rating by genre and year"), width="stretch")
            st.caption("Genre statistics cover only the movies matched to TMDB.")

    ds = table("decay_search")
    if ds is not None:
        st.subheader("Time decay: how fast should old ratings fade?")
        d = ds.copy()
        d["tau"] = ["no decay" if t == 0 else f"{int(t)} days" for t in d["tau_days"]]
        m = d.melt(id_vars=["tau", "tau_days"], value_vars=["val_rmse_bias", "val_rmse_content", "val_rmse_blend50"], var_name="model", value_name="validation RMSE")
        order = ["no decay"] + [f"{int(t)} days" for t in sorted(d["tau_days"][d["tau_days"] > 0], reverse=True)]
        st.plotly_chart(px.line(m, x="tau", y="validation RMSE", color="model", markers=True, category_orders={"tau": order}), width="stretch")
        best = d.loc[d["val_rmse_blend50"].idxmin()]
        pos = d[d["tau_days"] > 0]["tau_days"]
        edge = len(pos) > 0 and best["tau_days"] == pos.min()
        st.caption(f"Each rating is weighted exp(-age/tau). The 50/50 blend was best at **{best['tau']}** on validation data."
                   + (" That is the shortest decay tried, so an even shorter one might do better; this grid did not test it." if edge else ""))

    ar, at = table("alpha_rules"), table("alpha_table")
    if ar is not None:
        st.subheader("Adaptive blend weight")
        st.plotly_chart(px.bar(ar.sort_values("val_rmse"), x="val_rmse", y="rule", orientation="h", title="Validation RMSE of each weight rule"), width="stretch")
        st.caption("Each rule sets the weight on ALS (the rest goes to the content model) separately per bucket, using a closed-form least-squares fit on validation data. "
                   "Differences between rules are small; the winner was chosen on validation, not test.")
    if at is not None:
        keys = [c for c in at.columns if c not in ("alpha", "n")]
        st.markdown(f"**Fitted weight on ALS by {' x '.join(keys)}**")
        orders = {"b_user": USER_ORDER, "b_movie": MOVIE_ORDER, "b_conf": CONF_ORDER}
        if len(keys) == 2:
            p = at.pivot(index=keys[0], columns=keys[1], values="alpha")
            p = p.reindex(index=[x for x in orders.get(keys[0], list(p.index)) if x in p.index],
                          columns=[x for x in orders.get(keys[1], list(p.columns)) if x in p.columns])
            fig = px.imshow(p, text_auto=".2f", color_continuous_scale="RdBu", zmin=0, zmax=1, aspect="auto",
                            labels=dict(x=keys[1].replace("b_", "movie popularity" if keys[1] == "b_movie" else keys[1]),
                                        y="user history length" if keys[0] == "b_user" else keys[0], color="weight on ALS"))
            st.plotly_chart(fig, width="stretch")
        elif len(keys) == 1:
            d = at.copy(); d[keys[0]] = pd.Categorical(d[keys[0]], [x for x in orders.get(keys[0], list(d[keys[0]])) if x in set(d[keys[0]])], ordered=True)
            st.plotly_chart(px.bar(d.sort_values(keys[0]), x=keys[0], y="alpha", text="alpha"), width="stretch")
        else:
            st.dataframe(at, width="stretch")
        st.caption("Weight near 1 = trust the collaborative model (ALS); near 0 = trust the content model. Blank cells had no validation pairs.")
