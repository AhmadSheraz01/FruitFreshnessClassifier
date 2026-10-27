import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# Fruit Freshness AI — polished Streamlit interface
# Keep this file beside the "outputs" folder.
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
IMG_SIZE = (160, 160)

st.set_page_config(
    page_title="Fruit Freshness AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --green: #169b70;
      --green-dark: #123d30;
      --green-mid: #236c50;
      --mint: #eaf8ef;
      --text: #1b332a;
      --muted: #75847c;
      --border: #e1ece5;
    }

    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp {
      background:
        radial-gradient(ellipse at 8% 0%, rgba(190, 239, 208, .42), transparent 32%),
        linear-gradient(180deg, #f5fbf7 0%, #ffffff 48%, #f7fbf8 100%);
      color: var(--text);
    }
    .block-container {
      max-width: 1180px;
      padding-top: 1.8rem;
      padding-bottom: 3rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
      position: relative;
      overflow: hidden;
      background: linear-gradient(118deg, #10372b 0%, #176448 60%, #239568 100%);
      border-radius: 28px;
      padding: 32px 36px;
      color: white;
      box-shadow: 0 18px 45px rgba(21, 83, 58, .15);
      margin-bottom: 24px;
    }
    .hero:after {
      content: "🍎";
      position: absolute;
      right: 35px;
      top: 8px;
      font-size: 100px;
      opacity: .13;
      transform: rotate(-14deg);
    }
    .eyebrow {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #a9ebc9;
      margin-bottom: 8px;
    }
    .hero h1 {
      font-family: 'Manrope', sans-serif;
      color: #fff !important;
      font-size: clamp(30px, 4vw, 46px);
      line-height: 1.1;
      font-weight: 800;
      letter-spacing: -1.4px;
      margin: 0 0 12px;
    }
    .hero p {
      color: #e0f5e8 !important;
      font-size: 15px;
      max-width: 680px;
      margin: 0;
      line-height: 1.7;
    }
    .chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
    .chip {
      display:inline-block;
      color:#f2fff7;
      border:1px solid rgba(255,255,255,.25);
      background:rgba(255,255,255,.10);
      padding:7px 12px;
      border-radius:30px;
      font-size:12px;
      font-weight:600;
    }
    .section-kicker {
      color: #16815b;
      font-size: 11px;
      letter-spacing: 1.7px;
      font-weight: 800;
      text-transform: uppercase;
      margin: 7px 0 9px;
    }
    .panel {
      background: rgba(255,255,255,.94);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(22, 62, 43, .045);
      margin-bottom: 14px;
    }
    .panel-green {
      background: linear-gradient(145deg, #effbf4, #ffffff 85%);
      border: 1.5px solid #a9dfbf;
    }
    .panel-title {
      font-family: 'Manrope', sans-serif;
      font-weight: 800;
      color: #193e30;
      font-size: 20px;
      margin-bottom: 3px;
    }
    .muted { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .mini-label {
      color: #78877f;
      font-size: 10px;
      letter-spacing: 1.2px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .result-box {
      background: #e9f8ef;
      border: 1px solid #c8ead4;
      color: #116a45;
      padding: 15px 17px;
      border-radius: 15px;
      font-size: 20px;
      font-weight: 800;
      margin: 12px 0;
    }
    .result-rotten {
      background: #fff1ee;
      border-color: #ffd6cc;
      color: #a63b2b;
    }
    .metric-card {
      background: #fff;
      border: 1px solid #e2eee6;
      border-radius: 17px;
      padding: 17px;
      min-height: 104px;
    }
    .metric-card strong {
      display:block;
      font-family:'Manrope',sans-serif;
      font-size:26px;
      font-weight:800;
      color:#16815b;
      line-height:1.3;
      overflow-wrap:anywhere;
    }
    .metric-card span {
      color:#77857d;
      font-size:10px;
      font-weight:800;
      letter-spacing:1px;
    }
    div[data-testid="stFileUploader"] {
      background: #f3fbf6;
      border: 1.5px dashed #77c69d;
      border-radius: 17px;
      padding: 13px;
    }
    div[data-testid="stFileUploader"] section {
      padding: 10px;
    }
    .stButton > button {
      border: 0;
      border-radius: 12px;
      min-height: 46px;
      font-weight: 800;
      color: #fff;
      background: linear-gradient(110deg, #15966a, #22ae7b);
      box-shadow: 0 7px 18px rgba(21,150,106,.18);
      transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
      color:#fff;
      border:0;
      transform:translateY(-1px);
      box-shadow:0 10px 22px rgba(21,150,106,.24);
    }
    div[data-testid="stAlert"] { border-radius: 14px; }
    div[data-testid="stProgress"] > div > div { background-color:#1ba875; }
    hr { border-color:#e5eee8; margin: 1.7rem 0; }
    .footer-note {
      text-align:center;
      color:#87948d;
      font-size:12px;
      line-height:1.7;
      padding:18px 0 0;
    }
    @media (max-width: 700px) {
      .block-container { padding: 1rem 1rem 2rem; }
      .hero { padding: 25px 22px; border-radius:22px; }
      .hero:after { right:10px; font-size:65px; }
      .panel { padding:17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer Vision · Deep Learning Project</div>
      <h1>🍎 Fruit Freshness AI</h1>
      <p>Upload a fruit photo and let your trained AI model estimate whether an apple, banana, or orange appears fresh or rotten.</p>
      <div class="chips">
        <span class="chip">🧠 MobileNetV2</span>
        <span class="chip">📷 Image Classification</span>
        <span class="chip">🍊 6 Fruit Classes</span>
        <span class="chip">⚡ Instant Prediction</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def load_labels():
    label_file = OUT / "class_names.json"
    if not label_file.exists():
        return ["freshapples", "freshbanana", "freshoranges",
                "rottenapples", "rottenbanana", "rottenoranges"]
    try:
        data = json.loads(label_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data[k] for k in sorted(data, key=lambda x: int(x))]
    except (OSError, ValueError, TypeError):
        pass
    return ["freshapples", "freshbanana", "freshoranges",
            "rottenapples", "rottenbanana", "rottenoranges"]


@st.cache_resource
def load_model():
    # The training script saves both names. Prefer the best checkpoint.
    for filename in ("best_model.keras", "fruit_freshness_model.keras"):
        model_path = OUT / filename
        if model_path.exists():
            try:
                return tf.keras.models.load_model(model_path), filename, None
            except Exception as exc:
                return None, filename, str(exc)
    return None, None, None


def nice_label(label):
    return str(label).replace("_", " ").replace("-", " ").title()


def prepare_image(image):
    # MobileNetV2 preprocessing is embedded in the saved training model.
    arr = np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    return np.expand_dims(arr, axis=0)


model, model_filename, model_error = load_model()
class_names = load_labels()

upload_col, info_col = st.columns([1.03, 0.97], gap="large")

with upload_col:
    st.markdown('<div class="section-kicker">01 / Upload your fruit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><div class="panel-title">Choose a photo</div>'
        '<div class="muted">For the clearest result, use a well-lit image with one fruit clearly visible.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop your fruit image here or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Your selected fruit", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please try another file.")
    else:
        st.markdown(
            '<div style="text-align:center;padding:26px 8px 14px">'
            '<div style="font-size:48px">📸</div>'
            '<div style="font-weight:800;color:#28513e;margin-top:8px">Your fruit photo goes here</div>'
            '<div class="muted">Upload an apple, banana, or orange</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="section-kicker">02 / AI model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel panel-green"><div class="panel-title">MobileNetV2</div>'
        '<div class="muted">Transfer learning with a six-category classification head.</div>',
        unsafe_allow_html=True,
    )
    if model is not None:
        st.success(f"Ready to analyze · {model_filename}")
    else:
        st.warning("The trained model was not found or could not be loaded.")
        st.caption("Check that outputs/best_model.keras or outputs/fruit_freshness_model.keras exists.")
        if model_error:
            st.code(model_error)
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px">
          <div style="background:#fff;border:1px solid #dceee2;border-radius:12px;padding:11px 13px;flex:1;min-width:105px">
            <div class="mini-label">Input size</div><b style="color:#1a6e4d">160 × 160</b>
          </div>
          <div style="background:#fff;border:1px solid #dceee2;border-radius:12px;padding:11px 13px;flex:1;min-width:105px">
            <div class="mini-label">Framework</div><b style="color:#1a6e4d">TensorFlow</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">What can it recognize?</div>
          <div class="muted">🍎 Fresh apples · 🍌 Fresh bananas · 🍊 Fresh oranges<br>
          🟤 Rotten apples · 🍌 Rotten bananas · 🍊 Rotten oranges</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if image is not None:
    st.markdown("---")
    st.markdown('<div class="section-kicker">03 / Prediction results</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Ready when you are</div><div class="muted">Run the model on your uploaded image to see the predicted category and its score.</div>', unsafe_allow_html=True)
    analyze = st.button("✨  Analyze Fruit", type="primary", use_container_width=True, disabled=(model is None))
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze and model is not None:
        with st.spinner("Analyzing your fruit image..."):
            try:
                probabilities = np.asarray(
                    model.predict(prepare_image(image), verbose=0)
                )[0].astype(float).ravel()

                if len(probabilities) != len(class_names):
                    st.error("The model output does not match the class labels in outputs/class_names.json.")
                    st.stop()

                order = np.argsort(probabilities)[::-1]
                best_index = int(order[0])
                prediction = str(class_names[best_index])
                confidence = float(probabilities[best_index]) * 100
                label_lower = prediction.lower()
                is_fresh = label_lower.startswith("fresh")
                is_rotten = label_lower.startswith("rotten")
                freshness = "Fresh" if is_fresh else "Rotten" if is_rotten else "Review"
                fruit_icon = "🍎" if "apple" in label_lower else "🍌" if "banana" in label_lower else "🍊" if "orange" in label_lower else "🍏"

                st.markdown(
                    f'<div class="panel panel-green"><div class="mini-label">AI prediction</div>'
                    f'<div style="font-size:34px;margin:3px 0 5px">{fruit_icon}</div>'
                    f'<div class="panel-title" style="font-size:27px">{nice_label(prediction)}</div>'
                    f'<div class="result-box {"result-rotten" if is_rotten else ""}">{freshness} · {confidence:.2f}% model confidence</div>'
                    f'<div class="muted">This is the model’s prediction, not a guarantee of food safety.</div></div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f'<div class="metric-card"><span>PREDICTED CLASS</span><strong style="font-size:20px">{nice_label(prediction)}</strong></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="metric-card"><span>MODEL CONFIDENCE</span><strong>{confidence:.2f}%</strong></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div class="metric-card"><span>FRESHNESS</span><strong>{freshness}</strong></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Class probability breakdown")
                st.caption("These are model output scores. High confidence does not guarantee a correct prediction.")
                for idx in order:
                    st.progress(
                        float(np.clip(probabilities[idx], 0.0, 1.0)),
                        text=f"{nice_label(class_names[idx])} — {probabilities[idx] * 100:.2f}%",
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.markdown('<div class="section-kicker">04 / Project overview</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><span>CLASS CATEGORIES</span><strong>06</strong><div class="muted">Fresh and rotten fruit</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><span>IMAGE INPUT</span><strong>160 × 160</strong><div class="muted">RGB image resizing</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><span>DEEP LEARNING</span><strong>MobileNetV2</strong><div class="muted">Transfer learning model</div></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-note">
      <b>Fruit Freshness AI</b> · Computer Vision & Deep Learning Project<br>
      Educational demonstration only. A model can make mistakes, and appearance alone cannot confirm food safety.
    </div>
    """,
    unsafe_allow_html=True,
)
