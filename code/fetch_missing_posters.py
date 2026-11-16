r"""
Display-only helper: find TMDB posters for recommended titles that have none.

Why they have none: the Netflix catalogue contains TV seasons and box sets (e.g. "The Sopranos: Season 2"),
and the TMDB list we matched against holds movies only. This script searches TMDB's TV and movie endpoints
by a cleaned-up title and saves the poster path. It changes NOTHING in the models or the matching statistics;
the dashboard only uses it to show a picture.

Run on your laptop (not in Docker), from the netflix-bda folder:
    $env:TMDB_API_KEY = "your_key"
    python code\fetch_missing_posters.py
Needs data\dashboard_results\rec_candidates (from export_recommendations.py + the HDFS copy step).
Safe to stop and re-run: finished titles are skipped. Output: data\extra_posters.csv
"""
import csv, os, re, sys, time
import pandas as pd
import requests

KEY = os.environ.get("TMDB_API_KEY")
SRC = "data/dashboard_results/rec_candidates"
OUT = "data/extra_posters.csv"
BASE = "https://api.themoviedb.org/3"

def clean_title(t: str) -> str:
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r":\s*(Season|Seasons|Series|Vol\.?|Volume|Disc|Part)\b.*$", "", t, flags=re.I)
    t = re.sub(r":\s*(The\s+)?(Complete|Collector'?s|Special|Extended|Director'?s|Ultimate|Platinum|Widescreen|Full Screen).*$", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" :-")
    return t

def looks_like_tv(t: str) -> bool:
    return bool(re.search(r"\b(Season|Seasons|Series|Complete Series)\b", t, flags=re.I))

def get(path, **params):
    params["api_key"] = KEY
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2)) + 1); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return {"results": []}

def search(kind, q, year=None):
    p = {"query": q, "include_adult": "false", "language": "en-US"}
    if kind == "movie" and year:
        p["year"] = year
    for res in get(f"/search/{kind}", **p).get("results", []):
        if res.get("poster_path"):
            return res["poster_path"], res.get("title") or res.get("name")
    return None, None

if __name__ == "__main__":
    if not KEY:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    df = pd.read_parquet(SRC)
    need = df[df["poster_path"].isna()][["movie_id", "title", "year"]].drop_duplicates("movie_id")
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["movie_id"])
    todo = need[~need["movie_id"].isin(done)]
    print(f"{len(need)} recommended titles have no poster; {len(todo)} still to look up", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["movie_id", "poster_path", "source", "found_title", "query"])
        for i, r in enumerate(todo.itertuples(), 1):
            q = clean_title(str(r.title))
            order = ["tv", "movie"] if looks_like_tv(str(r.title)) else ["movie", "tv"]
            poster = found = src = None
            for kind in order:
                poster, found = search(kind, q, None if pd.isna(r.year) else int(r.year))
                if poster:
                    src = kind; break
            w.writerow([int(r.movie_id), poster or "", src or "", found or "", q]); f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
            time.sleep(0.05)
    out = pd.read_csv(OUT)
    print(f"done: {out['poster_path'].notna().sum()} of {len(out)} titles got a poster -> {OUT}")
