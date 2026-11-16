import pandas as pd
import plotly.express as px
import streamlit as st
from lib import table, fmt_int

def render():
    st.title("Data analysis")
    st.caption("All numbers come from Spark SQL queries over the full 100M ratings, except genres, which cover only the movies matched to TMDB.")

    bm = table("ratings_by_month")
    if bm is not None:
        st.subheader("Ratings per month")
        bm = bm.assign(date=pd.to_datetime(dict(year=bm["year"], month=bm["month"], day=1)))
        st.plotly_chart(px.line(bm, x="date", y="n_ratings", labels={"n_ratings": "ratings"}), width="stretch")

    dr = table("drift_by_year")
    if dr is not None:
        st.subheader("Do people rate higher or lower over time?")
        a, b = st.columns(2)
        a.plotly_chart(px.line(dr, x="year", y="avg_rating", markers=True, title="Average rating by year"), width="stretch")
        s = dr.melt(id_vars="year", value_vars=["share_1star", "share_5star"], var_name="kind", value_name="share")
        b.plotly_chart(px.line(s, x="year", y="share", color="kind", markers=True, title="Share of 1-star and 5-star ratings"), width="stretch")

    hist = table("rating_histogram")
    pb = table("movie_popularity_buckets")
    a, b = st.columns(2)
    if hist is not None:
        a.subheader("Rating distribution"); a.plotly_chart(px.bar(hist, x="rating", y="n"), width="stretch")
    if pb is not None:
        b.subheader("How many ratings does a movie have?")
        b.plotly_chart(px.bar(pb, x="bucket", y="n_movies", text="n_movies"), width="stretch")
        b.caption("Many movies have few ratings: this is the cold-start problem.")

    gs, gy = table("genre_summary"), table("genre_by_year")
    if gs is not None and gy is not None:
        st.subheader("Genres (TMDB-matched movies only, about 64% of the catalogue)")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(gs.sort_values("n_ratings"), x="n_ratings", y="genre", orientation="h"), width="stretch")
        pick = b.multiselect("Genres to compare over time", sorted(gy["genre"].unique()), default=list(gs.sort_values("n_ratings", ascending=False)["genre"][:4]))
        if pick:
            b.plotly_chart(px.line(gy[gy["genre"].isin(pick)], x="year", y="n_ratings", color="genre", markers=True), width="stretch")

    sk, tu = table("user_activity_skew"), table("top_users")
    if sk is not None:
        st.subheader("How lopsided is user activity?")
        v = sk.set_index("metric")["value"]
        c = st.columns(4)
        c[0].metric("Median ratings per user", fmt_int(v["median_ratings_per_user"]))
        c[1].metric("90th percentile", fmt_int(v["p90_ratings_per_user"]))
        c[2].metric("99th percentile", fmt_int(v["p99_ratings_per_user"]))
        c[3].metric("Ratings from top 1% of users", f"{100*v['share_of_ratings_from_top_1pct_users']:.1f}%")
    if tu is not None:
        with st.expander("50 most active users"):
            st.dataframe(tu, width="stretch")

    pm = table("polarizing_movies")
    if pm is not None:
        st.subheader("Most polarizing movies (largest spread of ratings, at least 1000 ratings)")
        st.plotly_chart(px.scatter(pm, x="avg_rating", y="sd_rating", size="n_ratings", hover_name="title",
                                   labels={"avg_rating": "average rating", "sd_rating": "std dev of ratings"}), width="stretch")
        st.dataframe(pm[["title", "release_year", "n_ratings", "avg_rating", "sd_rating", "share_1star", "share_5star"]].head(20), width="stretch")
        st.caption("These statistics use all ratings, so they describe the data. The re-ranking experiment used polarization computed from training ratings only.")
