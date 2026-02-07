
import streamlit as st
import pandas as pd
from datetime import datetime

from procrastination_analyzer.analysis.advanced_patterns import (
    ensure_ts, detect_perfectionism, detect_pattern, predict_risk, build_explainability
)
from procrastination_analyzer.rag.retrieve import retrieve_snippets
from procrastination_analyzer.report.generate import generate_report
from procrastination_analyzer.ui.visualize import plot_hour_distribution, plot_daily_activity, plot_bucket_pie

st.set_page_config(page_title="Procrastination Analyzer", layout="wide")

st.markdown('''
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.3px; }
.card { border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.03); }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.04); font-size:0.85rem; margin-right:6px; }
hr { border:none; border-top:1px solid rgba(255,255,255,0.12); margin: 1.2rem 0; }
</style>
''', unsafe_allow_html=True)

st.title("Procrastination Pattern Analyzer")
st.write("Upload a timestamp log (or type a few timestamps). This app summarizes your rhythm and suggests small adjustments.")

SNIPPETS_PATH = "procrastination_analyzer/rag/snippets.json"

def risk_label(r: float):
    if r >= 0.8:
        return ("High", "Tomorrow might slip unless you reduce friction tonight.")
    if r >= 0.5:
        return ("Medium", "You’ll be okay with a small plan + one early win.")
    return ("Low", "Rhythm looks decent — keep it simple and consistent.")

def pattern_summary_text(pat: str) -> str:
    if "Avoidance" in pat:
        return "Long gaps + late returns usually mean the task feels high-stakes or unclear."
    if "Fatigue" in pat:
        return "Evening-heavy work can be fine, but fatigue makes starting harder."
    if "Deadline" in pat:
        return "Bursts can work, but earlier mini-deadlines reduce stress."
    if "Consistent" in pat:
        return "Pretty steady. Your main goal is protecting this rhythm."
    return "Mixed pattern — usually fixed by clearer next steps + a predictable work slot."

def top_drivers(exp: dict):
    peak = exp.get("peak_hour", None)
    late = exp.get("late_night_ratio", 0)
    gaps24 = exp.get("long_gaps_24h", 0)
    burst = exp.get("burst_count_5in2h", 0)
    wknd = exp.get("weekend_ratio", 0)

    drivers = []
    if gaps24: drivers.append(f"{gaps24} long inactivity gap(s) (>24h).")
    if late >= 0.25: drivers.append(f"Late-night activity shows up often ({int(late*100)}%).")
    if burst: drivers.append(f"{burst} burst window(s): 5+ events within 2 hours.")
    if peak is not None: drivers.append(f"Peak activity hour: around {peak}:00.")
    if wknd >= 0.35: drivers.append(f"A lot of activity happens on weekends ({int(wknd*100)}%).")
    if not drivers: drivers.append("No single strong driver — looks mixed.")
    return drivers[:5]

# Sidebar
st.sidebar.header("Input")
mode = st.sidebar.radio("Choose data source", ["Upload CSV", "Manual input"], index=0)
st.sidebar.header("Suggestions")
top_k = st.sidebar.slider("How many suggestions", 2, 8, 4)
category_filter = st.sidebar.selectbox(
    "Category (optional)",
    ["(any)", "avoidance", "fatigue", "planning", "focus", "habits", "environment", "mindset", "time_management", "recovery"],
    index=0
)
st.sidebar.divider()
load_sample = st.sidebar.button("Use sample data")

df = None

@st.cache_data(show_spinner=False)
def parse_csv(uploaded):
    raw = pd.read_csv(uploaded)
    if "timestamp" not in raw.columns:
        raise ValueError("CSV must contain a `timestamp` column.")
    raw["ts"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    return raw.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def make_df_from_timestamps(timestamps):
    out = pd.DataFrame({"timestamp": timestamps})
    out["ts"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

if load_sample:
    df = pd.read_csv("procrastination_analyzer/data/sample_commits.csv")
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
elif mode == "Upload CSV":
    st.markdown("#### Upload a timestamp CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        df = parse_csv(uploaded)
else:
    st.markdown("#### Manual timestamps")
    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = [{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
    edited = st.data_editor(st.session_state.manual_rows, num_rows="dynamic", use_container_width=True)
    st.session_state.manual_rows = edited
    if st.button("Analyze"):
        timestamps = [row.get("timestamp") for row in edited if row.get("timestamp")]
        df = make_df_from_timestamps(timestamps)

if df is None:
    st.info("Add data from the sidebar to begin.")
    st.stop()

df = ensure_ts(df)

pattern = detect_pattern(df)
perfectionism = detect_perfectionism(df)
risk = predict_risk(df)
exp = build_explainability(df)

cat = None if category_filter == "(any)" else category_filter
snippets = retrieve_snippets(
    query=f"{pattern} procrastination advice next step",
    snippets_path=SNIPPETS_PATH,
    k=top_k,
    category=cat
)

risk_name, risk_msg = risk_label(risk)

st.markdown("### Overview")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pattern")
    st.write(pattern)
    st.caption(pattern_summary_text(pattern))
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Avoidance score")
    st.write(f"{perfectionism} / 1.0")
    st.caption("Heuristic score from gaps + bursts + late-night drift.")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Next-day risk")
    st.write(f"{risk_name} ({risk})")
    st.caption(risk_msg)
    st.markdown("</div>", unsafe_allow_html=True)

st.progress(min(1.0, float(risk)))
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("### Why the app thinks this")
for d in top_drivers(exp):
    st.markdown(f"- {d}")
with st.expander("See raw metrics"):
    st.json(exp)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Your rhythm")
left, right = st.columns([1, 1])
with left:
    st.pyplot(plot_hour_distribution(df))
    st.pyplot(plot_bucket_pie(df))
with right:
    st.pyplot(plot_daily_activity(df))

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Suggestions you can actually try")
for s in snippets:
    with st.expander(f"{s.get('title','Suggestion')} · relevance {s.get('score',0)}"):
        st.write(s["text"])
        tags = s.get("tags") or []
        if tags:
            st.markdown(" ".join([f"<span class='badge'>{t}</span>" for t in tags[:5]]), unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### Export")
if st.button("Generate report.md"):
    out_path = "procrastination_analyzer/output/report.md"
    data = {
        "advanced": {"perfectionism": perfectionism, "type": pattern, "risk": risk},
        "explainability": exp,
        "snippets": snippets,
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "events_used": len(df)}
    }
    generate_report(data, out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        st.download_button("Download report.md", f.read(), file_name="report.md", mime="text/markdown")
