"""Shared helpers: load the small result tables that Spark saved, and the hand-copied facts."""
import glob, json, os
from pathlib import Path
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DASH_DATA", HERE.parent / "data" / "dashboard_results"))
FACTS = json.loads((HERE / "facts.json").read_text())

@st.cache_data(show_spinner=False)
def _read_parquet(path: str):
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def _read_json_dir(path: str):
    files = sorted(glob.glob(os.path.join(path, "part-*.json")))
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True) if files else None

def table(name: str, kind: str = "parquet"):
    """Return a DataFrame, or None (and a visible warning) if the table has not been exported yet."""
    p = DATA / name
    if p.exists():
        df = _read_parquet(str(p)) if kind == "parquet" else _read_json_dir(str(p))
        if df is not None:
            return df
    st.warning(f"Table '{name}' not found in {DATA}. Run the export step (see RUN_STEPS_6A.md) after the Spark job that creates it.")
    return None

def fmt_int(x):
    return f"{int(x):,}"


@st.cache_data(show_spinner=False)
def extra_posters():
    """Optional display-only posters found by title search (data/extra_posters.csv). Returns {movie_id: poster_path}."""
    p = DATA.parent / "extra_posters.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d["poster_path"].notna() & (d["poster_path"].astype(str) != "")]
    return {int(m): str(x) for m, x in zip(d["movie_id"], d["poster_path"])}
