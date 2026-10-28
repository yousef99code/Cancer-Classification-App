import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction | Youssef Khaled Madin",
    page_icon="🩺",
    layout="wide",
)

# ------------------------------------------------------------------
# Custom CSS (colors + animations)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Cairo:wght@400;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', 'Cairo', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    .hero {
        text-align: center;
        padding: 2.2rem 1rem;
        border-radius: 24px;
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #ec4899, #6366f1);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite, fadeDown 1s ease;
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.45);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-family: 'Cairo', sans-serif;
        font-weight: 900;
        font-size: 2.6rem;
        color: white;
        margin: 0;
        text-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .hero p { color: #f1f5f9; margin: 0.4rem 0 0 0; font-size: 1.05rem; letter-spacing: 1px; }
    .badge {
        display: inline-block; margin-top: 0.8rem; padding: 0.3rem 1rem;
        background: rgba(255,255,255,0.18); border-radius: 999px;
        font-size: 0.85rem; backdrop-filter: blur(6px);
    }

    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(10px);
        animation: fadeUp 0.9s ease;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(99,102,241,0.3); }

    .chart-title {
        text-align: center; font-weight: 600; color: #c4b5fd;
        margin: 0.8rem 0 0 0; font-size: 1.05rem; letter-spacing: 0.5px;
    }

    .result {
        text-align: center; padding: 1.8rem 1rem; border-radius: 22px;
        animation: popIn 0.7s cubic-bezier(.17,.89,.32,1.28);
    }
    .benign {
        background: linear-gradient(135deg, #059669, #10b981);
        box-shadow: 0 0 35px rgba(16,185,129,0.55);
    }
    .malignant {
        background: linear-gradient(135deg, #be123c, #f43f5e);
        box-shadow: 0 0 35px rgba(244,63,94,0.55);
        animation: popIn 0.7s cubic-bezier(.17,.89,.32,1.28), pulse 1.8s infinite 0.7s;
    }
    .result h2 { margin: 0; font-size: 2rem; font-weight: 700; color: white; }
    .result p  { margin: 0.4rem 0 0 0; color: #f8fafc; font-size: 1.1rem; }

    .stButton > button {
        width: 100%;
        border: none; border-radius: 14px; padding: 0.7rem 1rem;
        font-weight: 600; font-size: 1rem; color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 22px rgba(139,92,246,0.55);
        color: white;
    }

    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }

    .footer { text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2.5rem; }
    .footer b { color: #c4b5fd; }

    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes fadeDown { from{opacity:0; transform:translateY(-25px)} to{opacity:1; transform:translateY(0)} }
    @keyframes fadeUp   { from{opacity:0; transform:translateY(25px)}  to{opacity:1; transform:translateY(0)} }
    @keyframes popIn    { from{opacity:0; transform:scale(0.8)} to{opacity:1; transform:scale(1)} }
    @keyframes pulse    { 0%,100%{box-shadow:0 0 25px rgba(244,63,94,0.45)} 50%{box-shadow:0 0 50px rgba(244,63,94,0.85)} }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Animated charts (pure SVG + CSS + JS, no extra libraries needed)
# ------------------------------------------------------------------
CHART_BASE_CSS = """
<style>
  body { margin:0; background:transparent; color:#e2e8f0;
         font-family:'Poppins',sans-serif; text-align:center; }
</style>
<script>
function countUp(el, target, dec, suffix, ms) {
  var start = null;
  function step(ts) {
    if (!start) start = ts;
    var p = Math.min((ts - start) / ms, 1);
    var e = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * e).toFixed(dec) + suffix;
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
</script>
"""

GAUGE_TEMPLATE = (
    CHART_BASE_CSS
    + """
<style>
  .wrap { width:280px; margin:0 auto; animation:fade 0.8s ease; }
  svg { width:280px; height:165px; overflow:visible; }
  .needle { transform-origin:140px 140px; transform:rotate(-90deg);
            transition:transform 1.8s cubic-bezier(.34,1.45,.64,1); }
  .val { font-size:34px; font-weight:700; margin-top:-6px; color:__COLOR__; }
  .lbl { font-size:13px; color:#94a3b8; letter-spacing:1px; }
  .zones { display:flex; justify-content:space-between; width:240px;
           margin:6px auto 0; font-size:11px; color:#94a3b8; }
  @keyframes fade { from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }
  .hub { animation: glow 2s infinite; }
  @keyframes glow { 0%,100%{opacity:1} 50%{opacity:0.55} }
</style>
<div class="wrap">
  <svg viewBox="0 0 280 165">
    <path d="M40 140 A100 100 0 0 1 90 53.4" fill="none" stroke="#10b981" stroke-width="18" stroke-linecap="round"/>
    <path d="M96 49.9 A100 100 0 0 1 184 49.9" fill="none" stroke="#f59e0b" stroke-width="18" stroke-linecap="round"/>
    <path d="M190 53.4 A100 100 0 0 1 240 140" fill="none" stroke="#f43f5e" stroke-width="18" stroke-linecap="round"/>
    <g class="needle" id="needle">
      <line x1="140" y1="140" x2="140" y2="62" stroke="#f8fafc" stroke-width="4" stroke-linecap="round"/>
    </g>
    <circle class="hub" cx="140" cy="140" r="10" fill="#f8fafc"/>
    <circle cx="140" cy="140" r="4" fill="#0f172a"/>
  </svg>
  <div class="val" id="v">0%</div>
  <div class="lbl">__LABEL__</div>
  <div class="zones"><span>Low</span><span>Medium</span><span>High</span></div>
</div>
<script>
  var p = __P__;
  setTimeout(function(){
    document.getElementById('needle').style.transform = 'rotate(' + (-90 + p * 180) + 'deg)';
  }, 150);
  countUp(document.getElementById('v'), p * 100, 1, '%', 1800);
</script>
"""
)

DONUT_TEMPLATE = (
    CHART_BASE_CSS
    + """
<style>
  .wrap { width:280px; margin:0 auto; animation:fade 0.8s ease; }
  svg { width:190px; height:190px; }
  .seg { transition: stroke-dasharray 1.6s cubic-bezier(.22,.9,.3,1); }
  .center { position:relative; width:190px; height:190px; margin:0 auto; }
  .ctext { position:absolute; inset:0; display:flex; flex-direction:column;
           align-items:center; justify-content:center; }
  .big { font-size:28px; font-weight:700; color:#f8fafc; }
  .small { font-size:12px; color:#94a3b8; letter-spacing:1px; }
  .legend { display:flex; justify-content:center; gap:18px; margin-top:8px; font-size:13px; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  @keyframes fade { from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }
</style>
<div class="wrap">
  <div class="center">
    <svg viewBox="0 0 180 180">
      <circle cx="90" cy="90" r="65" fill="none" stroke="#1e293b" stroke-width="22"/>
      <circle class="seg" id="s1" cx="90" cy="90" r="65" fill="none" stroke="__C1__"
              stroke-width="22" stroke-dasharray="0 408.4" transform="rotate(-90 90 90)"/>
      <circle class="seg" id="s2" cx="90" cy="90" r="65" fill="none" stroke="__C2__"
              stroke-width="22" stroke-dasharray="0 408.4" stroke-dashoffset="-__OFF__"
              transform="rotate(-90 90 90)"/>
    </svg>
    <div class="ctext">
      <div class="big" id="cv">0</div>
      <div class="small">__CLABEL__</div>
    </div>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:__C1__"></span>__L1__ __V1__</span>
    <span><span class="dot" style="background:__C2__"></span>__L2__ __V2__</span>
  </div>
</div>
<script>
  var C = 408.4, p1 = __P1__, p2 = 1 - p1;
  setTimeout(function(){
    document.getElementById('s1').style.strokeDasharray = (p1 * C) + ' ' + C;
    document.getElementById('s2').style.strokeDasharray = (p2 * C) + ' ' + C;
  }, 150);
  countUp(document.getElementById('cv'), __CVAL__, __CDEC__, '__CSUF__', 1600);
</script>
"""
)

TILES_TEMPLATE = (
    CHART_BASE_CSS
    + """
<style>
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; padding:4px; }
  .tile { border-radius:18px; padding:16px 10px; border:1px solid rgba(255,255,255,0.12);
          background:rgba(255,255,255,0.06); animation:rise 0.8s ease both;
          transition:transform .3s, box-shadow .3s; }
  .tile:hover { transform:translateY(-5px); box-shadow:0 10px 26px rgba(139,92,246,0.35); }
  .n { font-size:32px; font-weight:700; }
  .t { font-size:12px; color:#94a3b8; letter-spacing:1px; margin-top:2px; }
  @keyframes rise { from{opacity:0; transform:translateY(20px)} to{opacity:1; transform:none} }
</style>
<div class="grid" id="g"></div>
<script>
  var items = __ITEMS__;
  var g = document.getElementById('g');
  items.forEach(function(it, i){
    var d = document.createElement('div');
    d.className = 'tile';
    d.style.animationDelay = (i * 0.12) + 's';
    d.innerHTML = '<div class="n" style="color:' + it.color + '" id="n' + i + '">0</div>' +
                  '<div class="t">' + it.label + '</div>';
    g.appendChild(d);
    setTimeout(function(){ countUp(document.getElementById('n' + i), it.value, 0, '', 1400); }, i * 120);
  });
</script>
"""
)


def gauge_html(p_malignant: float) -> str:
    color = "#10b981" if p_malignant < 0.33 else "#f59e0b" if p_malignant < 0.66 else "#f43f5e"
    return (
        GAUGE_TEMPLATE.replace("__P__", f"{p_malignant:.4f}")
        .replace("__COLOR__", color)
        .replace("__LABEL__", "MALIGNANCY RISK")
    )


def donut_html(p1, label1, color1, label2, color2, center_value, center_label,
               decimals=1, suffix="%"):
    return (
        DONUT_TEMPLATE.replace("__P1__", f"{p1:.4f}")
        .replace("__OFF__", f"{p1 * 408.4:.2f}")
        .replace("__C1__", color1)
        .replace("__C2__", color2)
        .replace("__L1__", label1)
        .replace("__L2__", label2)
        .replace("__V1__", f"{p1 * 100:.1f}%")
        .replace("__V2__", f"{(1 - p1) * 100:.1f}%")
        .replace("__CVAL__", f"{center_value:.4f}")
        .replace("__CDEC__", str(decimals))
        .replace("__CSUF__", suffix)
        .replace("__CLABEL__", center_label)
    )


def tiles_html(items):
    return TILES_TEMPLATE.replace("__ITEMS__", json.dumps(items))


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1> Yousef Khaled Madin </h1>
        <p>Breast Cancer Prediction using Artificial Neural Network</p>
        <span class="badge">TensorFlow / Keras &nbsp;•&nbsp; Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load data, scaler and model
# ------------------------------------------------------------------
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"  # optional


@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, list(data.feature_names)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH, compile=False)


@st.cache_resource
def get_scaler():
    """Use saved scaler.pkl if it exists, otherwise rebuild the exact same
    scaler from the training notebook (same split: 80/20, random_state=42,
    stratified, fitted on the training part only)."""
    if os.path.exists(SCALER_PATH):
        import joblib

        return joblib.load(SCALER_PATH)

    df_, _ = load_data()
    X = df_.drop(columns="target")
    y = df_["target"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


df, feature_names = load_data()
model = get_model()
scaler = get_scaler()

if model is None:
    st.error(
        "⚠️ ملف الموديل **model.h5** مش موجود. حطه في نفس فولدر app.py وأعد التشغيل."
    )
    st.stop()

# ------------------------------------------------------------------
# Slider state + sample buttons
# ------------------------------------------------------------------
stats = df.drop(columns="target").describe().T


def set_defaults():
    for f in feature_names:
        st.session_state[f] = float(stats.loc[f, "mean"])


def set_sample(label):
    row = df[df["target"] == label].drop(columns="target").median()
    for f in feature_names:
        st.session_state[f] = float(row[f])


if feature_names[0] not in st.session_state:
    set_defaults()

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------
tab_predict, tab_data, tab_about = st.tabs(["🔮 Prediction", "📊 Dashboard", "ℹ️ About"])

# ===================== Prediction tab =====================
with tab_predict:
    st.markdown("### Quick fill")
    b1, b2, b3 = st.columns(3)
    b1.button("🟢 Typical Benign case", on_click=set_sample, args=(1,))
    b2.button("🔴 Typical Malignant case", on_click=set_sample, args=(0,))
    b3.button("♻️ Reset to average", on_click=set_defaults)

    st.markdown("### Tumor measurements")
    groups = {
        "Mean values": [f for f in feature_names if f.startswith("mean")],
        "Error values": [f for f in feature_names if f.endswith("error")],
        "Worst values": [f for f in feature_names if f.startswith("worst")],
    }

    cols = st.columns(3)
    for col, (title, feats) in zip(cols, groups.items()):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b></div>', unsafe_allow_html=True)
            with st.expander(f"Adjust {title.lower()}", expanded=False):
                for f in feats:
                    lo, hi = float(stats.loc[f, "min"]), float(stats.loc[f, "max"])
                    st.slider(
                        f,
                        min_value=lo,
                        max_value=hi,
                        step=(hi - lo) / 200,
                        format="%.4f",
                        key=f,
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict", key="predict_btn"):
        values = np.array([[st.session_state[f] for f in feature_names]])
        values_scaled = scaler.transform(pd.DataFrame(values, columns=feature_names))
        prob_benign = float(model.predict(values_scaled, verbose=0)[0][0])
        prob_malignant = 1 - prob_benign

        if prob_benign > 0.5:
            st.markdown(
                f"""
                <div class="result benign">
                    <h2>✅ Benign (حميد)</h2>
                    <p>Confidence: {prob_benign * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result malignant">
                    <h2>⚠️ Malignant (خبيث)</h2>
                    <p>Confidence: {prob_malignant * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        g_col, d_col = st.columns(2)
        with g_col:
            st.markdown('<div class="chart-title">Risk gauge</div>', unsafe_allow_html=True)
            components.html(gauge_html(prob_malignant), height=250)
        with d_col:
            st.markdown('<div class="chart-title">Probability split</div>', unsafe_allow_html=True)
            components.html(
                donut_html(
                    prob_benign, "Benign", "#10b981", "Malignant", "#f43f5e",
                    max(prob_benign, prob_malignant) * 100, "CONFIDENCE",
                    decimals=1, suffix="%",
                ),
                height=270,
            )

        m1, m2 = st.columns(2)
        m1.metric("Probability Benign", f"{prob_benign * 100:.2f}%")
        m2.metric("Probability Malignant", f"{prob_malignant * 100:.2f}%")

        st.caption(
            "This app is for educational purposes only and is not a medical diagnosis."
        )

# ===================== Dashboard tab =====================
with tab_data:
    st.markdown("### Breast Cancer Wisconsin Dataset")

    n_benign = int((df["target"] == 1).sum())
    n_malignant = int((df["target"] == 0).sum())

    components.html(
        tiles_html(
            [
                {"label": "SAMPLES", "value": int(df.shape[0]), "color": "#c4b5fd"},
                {"label": "FEATURES", "value": int(df.shape[1] - 1), "color": "#67e8f9"},
                {"label": "BENIGN", "value": n_benign, "color": "#34d399"},
                {"label": "MALIGNANT", "value": n_malignant, "color": "#fb7185"},
            ]
        ),
        height=130,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-title">Class distribution</div>', unsafe_allow_html=True)
        components.html(
            donut_html(
                n_benign / len(df), "Benign", "#10b981", "Malignant", "#f43f5e",
                len(df), "SAMPLES", decimals=0, suffix="",
            ),
            height=270,
        )
    with c2:
        st.markdown('<div class="chart-title">Average tumor size by class</div>', unsafe_allow_html=True)
        size_df = (
            df.groupby(df["target"].map({0: "Malignant", 1: "Benign"}))["mean area"]
            .mean()
            .round(1)
        )
        st.bar_chart(size_df)

    st.markdown("#### First rows")
    st.dataframe(df.head(10), use_container_width=True)

# ===================== About tab =====================
with tab_about:
    st.markdown(
        """
        <div class="card">
        <h3>About this project</h3>
        <p>An Artificial Neural Network (ANN) built with TensorFlow/Keras to classify
        breast tumors as <b>malignant</b> or <b>benign</b> from 30 numerical features.</p>
        <ul>
            <li>Dataset: Breast Cancer Wisconsin (scikit-learn)</li>
            <li>Preprocessing: stratified 80/20 split + StandardScaler (no data leakage)</li>
            <li>Model: Dense layers with ReLU, Dropout, sigmoid output</li>
            <li>Interface: Streamlit with animated gauge and donut charts</li>
        </ul>
        <p>Built by <b>Yousef Khaled Madin</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Made with ❤️ by <b>Yousef Khaled Madin</b></div>',
    unsafe_allow_html=True,
)
