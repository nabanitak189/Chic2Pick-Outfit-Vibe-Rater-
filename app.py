import io
import os
import requests
import numpy as np
from PIL import Image
import streamlit as st
import random
from utils.analyzers import aggregate_scores
from utils.ui import score_chip, verdict, section_header
from utils.clip_utils import analyze_outfit, _load_pil   # use internal loader

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Outfit Vibe Rater", page_icon="👗", layout="wide")

# -------------------------
# Load CSS (if exists)
# -------------------------
if os.path.exists("styles.css"):
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("assets/logo.svg"):
        st.image("assets/logo.svg", width=72)
with col_title:
    st.title("✨ Outfit Vibe Rater")
    st.caption("Upload/capture an outfit photo → get vibe scores + size + product recommendations")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Upload")
    img_choice = st.radio("Input method", ["Upload", "Camera"], horizontal=True)

    st.divider()
    st.header("Options")
    show_debug = st.checkbox("Show CLIP raw dictionary (debug)")
    presentation_mode = st.checkbox("Presentation Mode", value=True)
    adjust_strength = st.slider("Adjustment strength", 0.0, 1.0, 0.6, 0.05) if presentation_mode else 0.0

# -------------------------
# Input widget
# -------------------------
st.markdown("## 📸 Upload or Capture Outfit")
img_file = (
    st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if img_choice == "Upload"
    else st.camera_input("Capture outfit photo")
)

if img_file is None:
    st.info("⬅️ Upload or capture an outfit photo to get started.")
    st.stop()

# -------------------------
# Load image safely with shared util
# -------------------------
image = _load_pil(img_file)   # handles both upload + camera
img_np = np.array(image)

# -------------------------
# Run CLIP analyzer
# -------------------------
clip_scores = analyze_outfit(img_file)

if show_debug:
    st.markdown("#### 🔍 Raw CLIP output dictionary")
    st.json(clip_scores)

# -------------------------
# 🎭 Show Vibe Probabilities
# -------------------------
st.subheader("🎭 Outfit Vibe Analysis")
for vibe in ["casual", "formal", "trendy", "sporty", "traditional", "party"]:
    if vibe in clip_scores:
        prob = clip_scores[vibe] * 100
        st.progress(int(prob), text=f"{vibe.capitalize()}: {prob:.1f}%")

if "_best_label" in clip_scores:
    st.success(
        f"✨ Best Match: **{clip_scores['_best_label'].capitalize()}** "
        f"({clip_scores['_best_prob']*100:.1f}%)"
    )

# -------------------------
# Classical analysis
# -------------------------
raw_scores = aggregate_scores(img_np)

# -------------------------
# Occasion classification (best label)
# -------------------------
if "_best_label" in clip_scores:
    occasion = clip_scores["_best_label"]
else:
    clean = {k: float(v) for k, v in clip_scores.items() if not k.startswith("_")}
    occasion = max(clean, key=clean.get)

# -------------------------
# Style score (randomized demo)
# -------------------------
style_score = random.randint(70, 95)

def get_score_color(score):
    if score < 60:
        return "#e74c3c"
    elif score < 75:
        return "#f1c40f"
    else:
        return "#2ecc71"

color = get_score_color(style_score)

st.markdown("### ✨ Style Score")
st.markdown(
    f"""
    <div style="border: 1px solid #ddd; border-radius: 10px; height: 28px; position: relative; background-color: #f0f0f0;">
        <div style="width: {style_score}%; 
                    background: {color}; 
                    height: 100%; 
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-weight: bold;">
            {style_score}/100
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Size suggestion
# -------------------------
fit_score = raw_scores.get("fit_tightness", 50)
if fit_score < 30:
    suggested_size = "XL"
elif fit_score < 50:
    suggested_size = "L"
elif fit_score < 70:
    suggested_size = "M"
else:
    suggested_size = "S"

st.info(f"📏 Suggested size: **{suggested_size}**")

# -------------------------
# Fetch product recommendations
# -------------------------
try:
    resp = requests.get("http://127.0.0.1:5001/products")
    products = resp.json()
except Exception:
    st.error("⚠️ Could not fetch products from API")
    products = []

recommendations = [
    p for p in products
    if p.get("occasion","").lower()==occasion.lower() and suggested_size in p.get("sizes",[])
]

st.markdown("## 🛍️ Recommended Outfits for You")
if not recommendations:
    st.warning("No matching items found for your occasion + size.")
else:
    for p in recommendations[:3]:
        st.markdown(f"""
        **{p.get('name','Item')}**  
        🏷️ {p.get('price','N/A')} | 📦 {p.get('vendor','Unknown')}  
        👉 [View Item]({p.get('url','#')})
        """)
        st.divider()

# -------------------------
# Show original image + classical scores
# -------------------------
st.image(image, caption="Uploaded Outfit", use_container_width=True)

st.markdown("### 🎨 Key Scores")
for key, label in {
    "vibe_maximalism":"Maximalist Vibe",
    "pattern_density":"Pattern Density",
    "fit_tightness":"Fit Tightness",
    "colorfulness":"Colorfulness",
    "complexity":"Visual Complexity",
    "boldness":"Boldness"
}.items():
    val = raw_scores.get(key,0)
    st.progress(int(val), text=f"{label}: {val:.1f}%")
