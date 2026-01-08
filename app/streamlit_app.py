import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from typing import Optional, List, Dict, Any

# Optional YAML support
try:
    import yaml
    YAML_OK = True
except Exception:
    YAML_OK = False

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Boredom Is Bandwidth", layout="wide")

REPORT_CSV = Path("data/report/boredom_report_per_doc.csv")
NOVELTY_CURVES = Path("data/lsa/novelty_curves_per_doc.jsonl")
REDUNDANCY_CHUNKS = Path("data/redundancy/redundancy_per_chunk.jsonl")
COLLECTIONS_YAML = Path("collections/collections.yaml")

# -----------------------------
# Helpers
# -----------------------------
def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_collections():
    if not YAML_OK or not COLLECTIONS_YAML.exists():
        return {"collections": []}
    with COLLECTIONS_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"collections": []}

def bandwidth_color(score):
    if not np.isfinite(score):
        return "gray"
    if score < 0.25:
        return "#d73027"
    if score < 0.45:
        return "#fc8d59"
    if score < 0.65:
        return "#fee08b"
    return "#1a9850"

def saturation_bucket(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    if x >= 0.75:
        return "fast saturation"
    if x >= 0.45:
        return "moderate saturation"
    return "slow saturation"

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def safe_mean(series: pd.Series):
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.mean())

def resolve_collection_doc_ids(collection: dict, report_df: pd.DataFrame) -> list[str]:
    """
    Resolve doc_ids for a collection using either:
      - explicit doc_ids (if provided), else
      - match rules in collection['match'] (title_contains)
    """
    # 1) explicit doc_ids if present
    explicit = collection.get("doc_ids")
    if isinstance(explicit, list) and explicit:
        return [str(x) for x in explicit]

    match = collection.get("match", {}) or {}
    title_contains = match.get("title_contains", []) or []
    any_of = bool(match.get("any_of", True))  # default OR behavior

    subset = report_df.copy()

    # Optional: grade band filtering (lightweight; relies on title conventions)
    grade_band = collection.get("grade_band", "")
    # Not enforcing grade, but you could if your titles include it

    if title_contains:
        title_series = subset["title"].fillna("").astype(str).str.lower()

        needles = [str(x).lower() for x in title_contains if str(x).strip()]
        if any_of:
            mask = False
            for n in needles:
                mask = mask | title_series.str.contains(n, regex=False)
        else:
            mask = True
            for n in needles:
                mask = mask & title_series.str.contains(n, regex=False)

        subset = subset[mask]

    # Return resolved doc_ids
    return subset["doc_id"].dropna().astype(str).tolist()

    from typing import Optional, List, Dict, Any

def saturation_speed_from_curve(curve_rows, field="novelty_win", fraction=0.5):
    """
    0..1 score:
      0 = never drops below fraction of initial novelty
      1 = drops immediately
    """
    if not curve_rows:
        return None

    # Extract series (fallback to novelty_cum)
    vals = []
    for r in curve_rows:
        v = r.get(field)
        if v is None and field != "novelty_cum":
            v = r.get("novelty_cum")
        vals.append(v)

    valid = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
    if len(valid) < 2:
        return None

    start_i, start = valid[0]
    if start <= 0:
        return None
    thr = start * fraction

    t = None
    for i, v in valid[1:]:
        if v <= thr:
            t = i
            break

    span = valid[-1][0] - start_i
    if span <= 0:
        return None

    if t is None:
        return 0.0
    return 1.0 - (t - start_i) / span


# -----------------------------
# Load data
# -----------------------------
st.title("Boredom Is Bandwidth")

st.markdown(
"""
**This app analyzes instructional materials, not students.**  
Boredom here is treated as a *signal* that a text environment may be transmitting too little information over time.

Higher bandwidth ≠ “better.”  
It means **more semantic information per unit exposure**, which may or may not be desirable depending on the instructional goal.
"""
)

if not REPORT_CSV.exists():
    st.error("Missing data/report/boredom_report_per_doc.csv. Run notebook 06 first.")
    st.stop()

df = pd.read_csv(REPORT_CSV)

novelty_curves = read_jsonl(NOVELTY_CURVES)
redundancy_chunks = read_jsonl(REDUNDANCY_CHUNKS)

curve_by_doc = {r["doc_id"]: r for r in novelty_curves}
red_by_doc = {}
for r in redundancy_chunks:
    red_by_doc.setdefault(r["doc_id"], []).append(r)

collections_cfg = load_collections()
collections = collections_cfg.get("collections", []) or []

# Saturation speed per doc (how fast novelty collapses)
sat_speed_50 = {}
sat_speed_25 = {}

for doc_id, obj in curve_by_doc.items():
    curve = obj.get("curve", [])
    sat_speed_50[doc_id] = saturation_speed_from_curve(curve, fraction=0.5)
    sat_speed_25[doc_id] = saturation_speed_from_curve(curve, fraction=0.25)

df["saturation_speed_p50"] = df["doc_id"].map(sat_speed_50)
df["saturation_speed_p25"] = df["doc_id"].map(sat_speed_25)

# -----------------------------
# Navigation
# -----------------------------
tab = st.sidebar.radio("View", ["Browse", "Compare", "Methods"])

# -----------------------------
# Browse view
# -----------------------------
if tab == "Browse":
    st.header("Browse materials")

    titles = df["title"].fillna("Untitled").astype(str).tolist()
    title_to_row = {row["title"]: row for _, row in df.iterrows()}

    selected_title = st.sidebar.selectbox("Material", titles)
    row = title_to_row[selected_title]
    doc_id = row["doc_id"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Bandwidth score", fmt(row.get("bandwidth_score")),
                  help="Composite proxy from novelty, variety, redundancy.")
    with c2:
        st.markdown(
            f"""
            <div style="padding:0.5em;border-radius:6px;
                        background:{bandwidth_color(row.get('bandwidth_score', np.nan))};
                        color:white;font-weight:bold;text-align:center;">
            {str(row.get('bandwidth_bucket','unknown')).replace('_',' ').title()}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Materials-side bucket")
    with c3:
        st.metric("Mean semantic novelty", fmt(row.get("novelty_win_mean")),
                  help="Average new meaning per chunk relative to recent context.")
    with c4:
        st.metric("Surface redundancy", fmt(row.get("redundancy_gzip_mean")),
                  help="Higher = more compressible / repetitive surface structure.")
    with c5:
        val = row.get("saturation_speed_p50")
        st.metric(
            "Saturation speed (50%)",
            fmt(val),
            help="0=never drops below 50% of initial novelty; 1=drops immediately (faster saturation)."
        )
        st.caption(saturation_bucket(val).title())


    st.subheader("Semantic novelty over time")
    curve = curve_by_doc.get(doc_id)
    if curve:
        curve_df = pd.DataFrame(curve["curve"]).dropna(subset=["novelty_win"])
        chart = (
            alt.Chart(curve_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("chunk_index:Q", title="Chunk / Page"),
                y=alt.Y("novelty_win:Q", title="Semantic novelty (1 - cosine similarity)"),
                tooltip=["chunk_index", "novelty_win"]
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Flat regions indicate low semantic information rate; spikes indicate new meaning / topic shifts.")
    else:
        st.info("No novelty curve available for this document.")

    st.subheader("Most repetitive chunks (surface form)")
    red_chunks = red_by_doc.get(doc_id, [])
    if red_chunks:
        red_df = pd.DataFrame(red_chunks)
        top_red = red_df.sort_values("redundancy_gzip", ascending=False).head(6)
        for _, r in top_red.iterrows():
            st.markdown(
                f"""
                **Chunk {int(r['chunk_index'])}**  
                Redundancy (gzip): `{fmt(r.get('redundancy_gzip'))}`  
                Repeated sentence-opener fraction: `{fmt(r.get('top_opener_frac'))}`
                """
            )
    else:
        st.info("No redundancy chunk-level data available.")

    st.subheader("Contextual diversity snapshot")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric("Semantic variety", fmt(row.get("mean_pairwise_distance")),
                  help="Higher = broader semantic neighborhood coverage (LSA dispersion).")
    with cc2:
        st.metric("Semantic tightness", fmt(row.get("centroid_sim_mean")),
                  help="Higher = tighter clustering around one meaning (lower variety).")
    with cc3:
        st.metric("Surface entropy", fmt(row.get("char_ngram_entropy_norm_mean")),
                  help="Lower = more predictable character patterns.")

    st.subheader("How to read this report")
    st.markdown(
        """
- **Low bandwidth is not bad.** It can be intentional (e.g., stabilizing decoding).
- **Risk emerges when low bandwidth persists without recovery.** Flat novelty + high redundancy suggests informational collapse.
- **Good sequences manage constraints over time.** Constraint → release → new meaning → consolidation.
        """
    )

# -----------------------------
# Compare view
# -----------------------------
elif tab == "Compare":
    st.header("Compare collections (side-by-side)")

    if not collections:
        st.warning("No collections found. Create collections/collections.yaml (and install pyyaml).")
        st.stop()

    # Resolve doc_ids for each collection and show what matched
    resolved = []
    for c in collections:
        doc_ids = resolve_collection_doc_ids(c, df)
        resolved.append((c, doc_ids))
        
    # DEBUG: show matched titles (collapsed by default)
    with st.expander("Show matched documents (debug)", expanded=False):
        for c, doc_ids in resolved:
            st.markdown(f"**{c['display_name']}**")
            matched = df[df["doc_id"].isin(doc_ids)]
            if matched.empty:
                st.write("No documents matched.")
            else:
                for t in matched["title"].tolist():
                    st.write(f"- {t}")
    

    options = [c["display_name"] for c, _ in resolved]
    default_sel = options[:2] if len(options) >= 2 else options
    selected = st.multiselect("Choose collections", options, default=default_sel)

    if len(selected) < 2:
        st.info("Select at least 2 collections to compare.")
        st.stop()

    rows_out = []
    for c, doc_ids in resolved:
        if c["display_name"] not in selected:
            continue

        subset = df[df["doc_id"].isin(doc_ids)].copy() if doc_ids else df.copy()

        rows_out.append({
            "Collection": c["display_name"],
            "Sampling mode": c.get("sampling_mode", "Unspecified"),
            "Grade band": c.get("grade_band", ""),
            "Docs matched": int(len(subset)),
            "Matched titles": ", ".join(subset["title"].fillna("").astype(str).head(6).tolist()) + (" …" if len(subset) > 6 else ""),
            "Bandwidth score": safe_mean(subset.get("bandwidth_score")),
            "Mean novelty": safe_mean(subset.get("novelty_win_mean")),
            "Redundancy (gzip)": safe_mean(subset.get("redundancy_gzip_mean")),
            "Surface entropy": safe_mean(subset.get("char_ngram_entropy_norm_mean")),
            "Semantic variety": safe_mean(subset.get("mean_pairwise_distance")),
            "Semantic tightness": safe_mean(subset.get("centroid_sim_mean")),
            "Saturation speed (50%)": safe_mean(subset.get("saturation_speed_p50")),
        })

    compare_df = pd.DataFrame(rows_out)
    st.dataframe(compare_df, use_container_width=True)

    st.caption("Collection metrics are averages across matched documents. Use Browse to inspect individual novelty curves.")

    chart_df = compare_df.copy()
    chart_df["Bandwidth score"] = pd.to_numeric(chart_df["Bandwidth score"], errors="coerce")
    bar = (
        alt.Chart(chart_df.dropna(subset=["Bandwidth score"]))
        .mark_bar()
        .encode(
            x=alt.X("Collection:N", sort="-y"),
            y=alt.Y("Bandwidth score:Q"),
            tooltip=["Collection", "Sampling mode", "Docs matched", "Bandwidth score"]
        )
        .properties(height=260)
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("Sampling notes")
    for c, doc_ids in resolved:
        if c["display_name"] in selected:
            st.markdown(f"**{c['display_name']}**")
            st.write(f"Sampling mode: **{c.get('sampling_mode','Unspecified')}**")
            st.write(c.get("sampling_mode_explainer", ""))
            st.caption(c.get("notes", ""))
            if doc_ids:
                st.caption(f"Matched {len(doc_ids)} document(s).")
            else:
                st.caption("Matched 0 documents — adjust title keywords in collections.yaml.")
            st.markdown("---")

# -----------------------------
# Methods view
# -----------------------------
else:
    st.header("Methods")

    st.subheader("Entry sequence vs Transition window")
    st.markdown(
        """
**Entry sequence**  
A contiguous run of the *first* student-facing texts introduced at the start of a reading phase or unit.  
Use this to ask: *What is the information environment when this design begins?*

**Transition window**  
A contiguous run of texts surrounding a documented change in instructional constraints (e.g., release from strict decodability).  
Use this to ask: *Does semantic bandwidth actually increase when it is supposed to?*
        """
    )

    st.subheader("Sampling protocol (materials-side, reproducible)")
    st.markdown(
        """
**Guiding principles**
- Student-facing text only (no teacher prompts, no assessments)
- Contiguous sequences (not isolated excerpts)
- Comparable grade band and purpose
- Predefined inclusion rules (reduce cherry-picking)

**Inclusion criteria**
- Same grade band per comparison
- ≥ 10 consecutive passages in intended order
- Clear ordering (lesson number, page order, or publication sequence)
        """
    )

    st.subheader("Metric families")
    st.markdown(
        """
- **Semantic novelty** (LSA): new meaning relative to recent context  
- **Surface redundancy** (gzip, n-gram entropy, templates): repetition and predictability of form  
- **Contextual diversity** (semantic dispersion): breadth of semantic neighborhoods covered

These are **material properties**, not learner diagnoses.
        """
    )

    st.subheader("Semantic saturation speed")
    st.markdown(
        """
**What it measures**  
How quickly a text’s semantic novelty collapses over time, independent of its average richness.

**How it’s computed**  
From each document’s semantic novelty curve (LSA), we measure the normalized time required for novelty to drop below a fixed fraction (default: 50%) of its initial value.

**How to read it**  
- **Fast saturation**: rapid collapse of semantic uncertainty (tight scaffolding or over-constraint)  
- **Slow saturation**: sustained delivery of new semantic information

This metric distinguishes *productive repetition* from *dead semantic bandwidth* and should not be interpreted as a quality judgment.
        """
    )


    st.subheader("Neutral naming conventions")
    st.markdown(
        """
Collections are labeled by **design intent** rather than publisher brands, e.g.:
- Programmed knowledge-building sequence
- Authentic narrative set
- Hybrid decoding-to-meaning sequence
        """
    )
