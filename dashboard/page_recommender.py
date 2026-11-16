import pandas as pd
import streamlit as st
from lib import table, extra_posters

IMG = "https://image.tmdb.org/t/p/w185"
MODES = {"Safe picks (avoid polarizing movies)": 0.2, "Balanced": 0.0, "Risky picks (favour polarizing movies)": -0.2}
RANKERS = {"ALS (collaborative filtering)": "als", "Adaptive hybrid (collaborative + content)": "adaptive"}

def stars(x):
    return "★" * int(round(x))

def card(r, rank, extra):
    searched = False
    poster = r["poster_path"] if isinstance(r["poster_path"], str) and r["poster_path"] else None
    if poster is None and int(r["movie_id"]) in extra:
        poster, searched = extra[int(r["movie_id"])], True
    if poster:
        st.image(IMG + poster, width=150)
        if searched:
            st.markdown("<span style='color:#888;font-size:11px'>poster found by title search</span>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='width:150px;height:100px;border:1px solid #888;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;color:#888'>no poster</div>", unsafe_allow_html=True)
    year = "" if pd.isna(r["year"]) else f" ({int(r['year'])})"
    st.markdown(f"**{rank}. {r['title']}{year}**")
    if isinstance(r["genres"], str) and r["genres"]:
        st.caption(r["genres"])
    st.write(f"Predicted rating: **{r['als']:.1f}**")
    has_content = isinstance(r["because_title"], str)
    if has_content:
        a = float(r["alpha"]) * 100
        st.markdown(f"<div style='display:flex;height:8px;border-radius:4px;overflow:hidden'><div style='width:{a:.0f}%;background:#4c78a8'></div>"
                    f"<div style='width:{100-a:.0f}%;background:#f58518'></div></div>"
                    f"<div style='font-size:12px'>{a:.0f}% collaborative / {100-a:.0f}% content</div>", unsafe_allow_html=True)
    z = r["z"]
    tag = "polarizing" if z > 0.75 else ("consensus pick" if z < -0.75 else "mixed opinions")
    st.caption(f"{tag} (rating spread {r['pol']:.2f})")
    if has_content:
        st.caption(f"Because you rated *{r['because_title']}* {stars(r['because_rating'])} (similar plot/genre)")
    else:
        st.caption("Collaborative filtering only: no plot-similar movie in this user's history "
                   "(TV seasons and titles not matched to TMDB have no content data).")
    if pd.notna(r["test_rating"]):
        st.markdown(f"{'✅' if r['test_rating'] >= 4 else '▫️'} Later rated {stars(r['test_rating'])}")
    else:
        st.markdown("<span style='color:#888;font-size:12px'>Not rated in the test period</span>", unsafe_allow_html=True)

def render():
    st.title("Recommender")
    st.caption("Offline demo on 1000 sampled users. Recommendations come from the 1000 most-rated movies, using models trained on ratings before the test period. "
               "A ✅ marks a movie the user actually rated 4 or more stars during the test period; no mark does NOT mean they disliked it, only that they did not rate it.")
    cand, usr = table("rec_candidates"), table("rec_users")
    if cand is None or usr is None:
        return
    usr = usr.sort_values("user_id")
    labels = {int(r.user_id): f"user {int(r.user_id)}  ·  {int(r.n_hist)} earlier ratings  ·  {int(r.n_liked_in_test)} movies liked later" for r in usr.itertuples()}
    uid = st.sidebar.selectbox("User", list(labels), format_func=lambda u: labels[u], key="rec_user")
    ranker = st.sidebar.radio("Ranker", list(RANKERS), key="rec_ranker")
    mode = st.sidebar.radio("Style", list(MODES), index=1, key="rec_mode")
    lam = MODES[mode]
    u = usr[usr["user_id"] == uid].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Ratings before the test period", int(u["n_hist"]))
    b.metric("Average rating given", f"{u['avg_hist_rating']:.2f}")
    c.metric("Movies rated 4+ in the test period", int(u["n_liked_in_test"]))
    if isinstance(u["liked_before"], str):
        st.markdown(f"**Highest-rated earlier:** {u['liked_before']}")

    d = cand[cand["user_id"] == uid].copy()
    d["score"] = d[RANKERS[ranker]] - lam * d["z"]
    top = d.sort_values(["score", "movie_id"], ascending=[False, True]).head(10)
    hits = int((top["test_rating"] >= 4).sum())
    st.subheader(f"Top 10 for this user: {hits} of 10 were rated 4+ later")
    extra = extra_posters()
    for start in (0, 5):
        cols = st.columns(5)
        for col, (i, (_, r)) in zip(cols, enumerate(top.iloc[start:start + 5].iterrows(), start=start + 1)):
            with col:
                card(r, i, extra)

    rp = table("rerank_polarization")
    if rp is not None:
        with st.expander("What does the Safe / Risky setting do? (measured on all 1000 users)"):
            base_name = "ALS, popular pool" if RANKERS[ranker] == "als" else None
            if base_name is None:
                st.write("The re-ranking experiment was run on the ALS ranker and on a fixed 50/50 hybrid, not on this adaptive hybrid, so no measured numbers are shown for it.")
            else:
                s = rp[(rp["base"] == base_name) & (rp["lam"].isin([-0.2, 0.0, 0.2]))].sort_values("lam")
                s = s.assign(style=s["lam"].map({-0.2: "Risky", 0.0: "Balanced", 0.2: "Safe"}))
                st.dataframe(s[["style", "p10", "distinct_movies", "mean_polarization", "bad_rate"]].round(4), width="stretch", hide_index=True)
                st.caption("Risky lowers precision and raises the share of disliked picks. Safe keeps precision but recommends a narrower set of movies.")
    with st.expander("How to read a recommendation card"):
        st.markdown("- **Predicted rating:** from the collaborative model (ALS).\n"
                    "- **Collaborative / content bar (only when the movie has content evidence for this user):** the weight the adaptive rule gives the two models. It was fitted for rating prediction "
                    "(on validation data), so it explains the blend, not the ranking order of the ALS ranker.\n"
                    "- **Because you rated:** the most similar movie in the user's history, by TF-IDF similarity of TMDB plots and genres.\n"
                    "- **Polarizing / consensus:** how much people disagree about the movie, from the spread of its training-period ratings.\n"
                    "- Posters load from TMDB, so they need an internet connection.")
