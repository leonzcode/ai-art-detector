import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# Model architecture (from Art_ML_Training.ipynb)
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "models" / "Cosmos_Art_Detector.pth"
IMG_SIZE = 128
PATCH_SIZE = 64
NUM_PATCHES = 10


class PatchCraftDetector(nn.Module):
    def __init__(self, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_out = self.features(torch.zeros(1, 3, patch_size, patch_size))
            in_features = dummy_out.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    model = PatchCraftDetector(patch_size=PATCH_SIZE)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    # Support both raw state_dict and {'model_state_dict': ...} bundle
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: nn.Module, image_bytes: bytes) -> tuple[bool, float]:
    """Returns (is_ai, ai_confidence_0_to_1) using patch voting."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    chw = np.transpose(arr, (2, 0, 1))                       # (3, H, W)
    img_t = torch.from_numpy(chw)

    rng = np.random.default_rng(seed=42)
    max_pos = IMG_SIZE - PATCH_SIZE
    patches = []
    for _ in range(NUM_PATCHES):
        x = int(rng.integers(0, max_pos + 1))
        y = int(rng.integers(0, max_pos + 1))
        patches.append(img_t[:, x:x + PATCH_SIZE, y:y + PATCH_SIZE])
    batch = torch.stack(patches)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
    ai_conf = float(probs.mean())
    return ai_conf > 0.5, ai_conf


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Cosmos AI Art Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main { padding: 2rem; }
    .title-container { text-align: center; margin-bottom: 2rem; }
    .result-container { padding: 1.5rem; border-radius: 10px; margin-top: 1.5rem; font-size: 1.1rem; }
    .ai-result    { background-color: #ffebee; border-left: 6px solid #f44336; }
    .human-result { background-color: #e8f5e9; border-left: 6px solid #4caf50; }
    .confidence-label { font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; }

    .leon-breadcrumb {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .leon-breadcrumb a { color: inherit; text-decoration: none; }
    .leon-breadcrumb a:hover { color: #111827; text-decoration: underline; }
    .leon-sep { color: #9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

# Breadcrumb back to the project's detail page on the main portfolio.
# Reorients visitors: they're still inside Leon's portfolio, in the AI section.
st.markdown(
    """
<nav class="leon-breadcrumb">
  <span>←</span>
  <a href="https://leonzhao.dev/">leonzhao.dev</a>
  <span class="leon-sep">/</span>
  <a href="https://leonzhao.dev/ai/">AI</a>
  <span class="leon-sep">/</span>
  <a href="https://leonzhao.dev/ai/ai-art/">AI Art Detection</a>
</nav>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="title-container">
    <h1>🎨 Cosmos AI Art Detector</h1>
    <p>Upload an image to find out if it's <b>AI-generated</b> or <b>human-made</b></p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("Patch-voting CNN trained on the Kaggle AI-vs-Human dataset (~80k images).")
    st.markdown(
        """
- Supports JPG, PNG, BMP, GIF, WebP
- Resizes input to 128×128, samples 10 random 64×64 patches
- Reports the averaged AI-class probability across patches
"""
    )
    st.markdown("---")
    st.markdown("**apps.leonzhao.dev/ai-art**")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Step 1: Upload Image(s)")
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "bmp", "gif", "webp"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
with col2:
    st.markdown("### Step 2: View the Results")

if not uploaded_files:
    col2.info("👆 Upload one or more images to get started!")
else:
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            image_bytes = uploaded_file.read()

            if len(image_bytes) > 50 * 1024 * 1024:
                col2.error(f"⚠️ {uploaded_file.name}: Image too large (max 50MB)")
                continue

            col1.markdown(f"#### Image {index}")
            col1.image(image_bytes, caption=uploaded_file.name, use_column_width=True)

            with st.spinner(f"🔄 Analyzing {uploaded_file.name}..."):
                is_ai, ai_conf = predict(model, image_bytes)
            human_conf = 1 - ai_conf

            verdict_color = "#f44336" if is_ai else "#4caf50"
            verdict_label = "AI-Generated" if is_ai else "Human-Made"
            verdict_class = "ai-result" if is_ai else "human-result"

            col2.markdown(
                f"""
                <div class="result-container {verdict_class}">
                    <h3>Image {index}: <span style="color: {verdict_color};">{verdict_label}</span></h3>
                    <p><strong>{uploaded_file.name}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col2.markdown(
                "<div class=\"confidence-label\">📊 Confidence Breakdown:</div>",
                unsafe_allow_html=True,
            )
            metric_col1, metric_col2 = col2.columns(2)
            with metric_col1:
                st.metric(f"🧑 Human {index}", f"{human_conf * 100:.1f}%")
            with metric_col2:
                st.metric(f"🤖 AI {index}", f"{ai_conf * 100:.1f}%")

            col2.markdown("**Confidence Meter:**")
            col2.progress(human_conf, text=f"Human: {human_conf * 100:.1f}%")
            col2.progress(ai_conf, text=f"AI: {ai_conf * 100:.1f}%")

            if index < len(uploaded_files):
                col2.markdown("---")

        except Exception as e:
            col2.error(f"⚠️ {uploaded_file.name}: {str(e)[:200]}")

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Cosmos AI Art Detector · PatchCraft CNN · Powered by PyTorch + Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
