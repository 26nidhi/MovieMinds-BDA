"""
Step 2a: Download a TMDB movie catalogue (title, year, genres, plot, poster) for
release years 1896-2005 and save it as one JSON-lines file: data/tmdb_movies.jsonl

Run on your laptop (not in Docker):
    pip install requests
    $env:TMDB_API_KEY = "your_key_here"          (PowerShell)
    python code/fetch_tmdb.py

Safe to stop and re-run: each year is saved to data/tmdb_raw/ and skipped next time.
Netflix movies are matched to this catalogue in match_titles.py.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

KEY = os.environ.get("TMDB_API_KEY")
if not KEY:
    sys.exit("Set the TMDB_API_KEY environment variable first.")

BASE = "https://api.themoviedb.org/3"
RAW_DIR = "data/tmdb_raw"
OUT = "data/tmdb_movies.jsonl"
os.makedirs(RAW_DIR, exist_ok=True)
sess = requests.Session()

def get(path, **params):
    params["api_key"] = KEY
    for attempt in range(6):
        try:
            r = sess.get(f"{BASE}{path}", params=params, timeout=20)
            if r.status_code == 429:                      # rate limited
                time.sleep(int(r.headers.get("Retry-After", 2)) + 1)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))                 # network hiccup: wait and retry
    raise RuntimeError(f"failed: {path} {params}")

genre_map = {g["id"]: g["name"] for g in get("/genre/movie/list")["genres"]}

def max_pages(year):
    # recent years hold most Netflix titles, so fetch deeper there
    return 150 if year >= 1990 else 50

def fetch_year(year):
    path = f"{RAW_DIR}/{year}.jsonl"
    if os.path.exists(path):
        return year, -1                                   # already downloaded
    rows, page, total = [], 1, 1
    while page <= min(total, max_pages(year), 500):
        d = get("/discover/movie", primary_release_year=year, page=page,
                sort_by="vote_count.desc", include_adult="false", language="en-US")
        total = d.get("total_pages", 1)
        for m in d["results"]:
            rows.append({
                "tmdb_id": m["id"], "title": m.get("title"),
                "original_title": m.get("original_title"),
                "release_date": m.get("release_date"),
                "overview": m.get("overview"),
                "genres": [genre_map.get(g) for g in m.get("genre_ids", []) if g in genre_map],
                "popularity": m.get("popularity"), "vote_count": m.get("vote_count"),
                "vote_average": m.get("vote_average"), "poster_path": m.get("poster_path"),
            })
        page += 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)                                 # year file appears only when complete
    return year, len(rows)

years = list(range(2005, 1895, -1))
with ThreadPoolExecutor(max_workers=6) as ex:
    for y, n in ex.map(fetch_year, years):
        print(f"{y}: " + ("already done" if n == -1 else f"{n} movies"), flush=True)

# combine all years into one file, dropping duplicate ids
seen, total = set(), 0
with open(OUT, "w", encoding="utf-8") as out:
    for y in years:
        p = f"{RAW_DIR}/{y}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            if rec["tmdb_id"] in seen:
                continue
            seen.add(rec["tmdb_id"]); out.write(line); total += 1
print("TOTAL movies saved:", total, "->", OUT)
