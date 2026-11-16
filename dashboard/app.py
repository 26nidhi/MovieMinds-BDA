import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st

st.set_page_config(page_title="Netflix BDA project", layout="wide")
PAGES = ["1. Pipeline overview", "2. Data analysis", "3. Performance lab",
         "4. Model results", "5. Drift explorer", "6. Recommender"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("CS404 Big Data Analytics. Reads small result tables exported from HDFS; Spark is never run from here.")

if page == PAGES[0]:
    import page_overview; page_overview.render()
elif page == PAGES[1]:
    import page_analysis; page_analysis.render()
elif page == PAGES[2]:
    import page_performance; page_performance.render()
elif page == PAGES[3]:
    import page_models; page_models.render()
elif page == PAGES[4]:
    import page_drift; page_drift.render()
else:
    import page_recommender; page_recommender.render()
