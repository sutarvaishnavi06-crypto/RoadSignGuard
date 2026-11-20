import os
import streamlit as st
from PIL import Image
from predict import load_model, predict_image

st.set_page_config(
    page_title="RoadSignGuard | Perception Console",
    page_icon="◈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background:#0b0e12;
    color:#e8e9ed;
}
.block-container {
    max-width:1500px;
    padding:1.5rem 2rem;
}
header,[data-testid="stFooter"] {visibility:hidden;}
#MainMenu {visibility:hidden;}

section[data-testid="stSidebar"] {
    background:#11151b;
}

.topbar {
    background:#14181f;
    border:1px solid #292e37;
    border-radius:10px;
    padding:15px 22px;
    margin-bottom:18px;
}
.brand {
    font-size:27px;
    font-weight:900;
    letter-spacing:-1px;
}
.red {color:#e54848;}
.muted {color:#858d99;}
.eyebrow {
    font-family:monospace;
    color:#929aa8;
    letter-spacing:2px;
    font-size:10px;
}
.panel {
    background:#14181e;
    border:1px solid #2a3039;
    border-radius:10px;
    padding:18px;
    margin-bottom:12px;
}
.panel-head {
    color:#a0a8b4;
    font-family:monospace;
    font-size:11px;
    letter-spacing:1.5px;
    margin-bottom:12px;
}
.metric-card {
    background:#181c23;
    border:1px solid #303641;
    border-radius:9px;
    padding:16px;
    min-height:105px;
    margin-bottom:8px;
}
.metric-label {
    font-size:11px;
    color:#969eaa;
    letter-spacing:1px;
}
.metric-value {
    font-size:23px;
    font-weight:700;
    margin-top:10px;
    color:#f0f1f4;
    word-wrap:break-word;
}
.status-pill {
    display:inline-block;
    padding:5px 10px;
    border-radius:20px;
    background:#202a25;
    color:#70d49a;
    font-family:monospace;
    font-size:10px;
}
.alert-pill {
    background:#35251b;
    color:#f1ae54;
}
div.stButton>button {
    background:#d94145;
    color:white;
    border:0;
    border-radius:7px;
    height:46px;
    font-weight:700;
}
div.stButton>button:hover {
    background:#ee5052;
    color:white;
}
[data-testid="stFileUploader"] {
    background:#171b21;
    border:1px dashed #3a414b;
    border-radius:9px;
    padding:10px;
}
div[data-testid="stMetric"] {
    background:#181c23;
    border:1px solid #303641;
    padding:12px;
    border-radius:8px;
}
hr {border-color:#282d35;}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="topbar">
    <div class="eyebrow">AUTONOMOUS PERCEPTION SYSTEM</div>
    <div class="brand">
        ROAD SIGN<span class="red">GUARD</span>
    </div>
    <div class="muted" style="margin-top:5px;">
        Traffic Sign Classification & Occlusion Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# SYSTEM STATUS
a, b, c = st.columns(3)

a.markdown("""
<div class="panel">
<div class="eyebrow">SYSTEM MODE</div>
<b>Single-frame analysis</b>
</div>
""", unsafe_allow_html=True)

b.markdown("""
<div class="panel">
<div class="eyebrow">MODEL ARCHITECTURE</div>
<b>Dual-head CNN</b>
</div>
""", unsafe_allow_html=True)

c.markdown("""
<div class="panel">
<div class="eyebrow">SIGN CLASSES</div>
<b>43 categories · GTSRB</b>
</div>
""", unsafe_allow_html=True)

# WORKSPACE
left, right = st.columns([1.15, 1], gap="medium")

with left:
    st.markdown("""
    <div class="panel-head">01 / VISUAL INPUT</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Import road sign frame",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    image = None

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        st.image(image, use_container_width=True)

        st.caption(
            f"INPUT FRAME · {image.width} × {image.height}"
        )
    else:
        st.markdown("""
        <div class="panel" style="text-align:center;padding:65px 20px;">
            <div style="font-size:42px;color:#4a515c;">◈</div>
            <div style="font-size:15px;">Camera frame unavailable</div>
            <div class="muted">Upload an image to initialize analysis</div>
        </div>
        """, unsafe_allow_html=True)

    run = st.button(
        "EXECUTE PERCEPTION  →",
        use_container_width=True,
        disabled=image is None
    )

with right:
    st.markdown("""
    <div class="panel-head">02 / INTELLIGENCE OUTPUT</div>
    """, unsafe_allow_html=True)

    if run and image:
        if not os.path.exists("saved_models/roadsignguard.pth"):
            st.error("Checkpoint missing. Train the model first.")
        else:
            with st.spinner("Running neural inference..."):
                model = load_model()
                result = predict_image(model, image)

            st.markdown("### Classification")

            st.metric(
                "Predicted traffic sign",
                result["sign"]
            )

            st.metric(
                "Classification confidence",
                f"{result['sign_confidence']:.2f}%"
            )

            st.markdown("### Visibility analysis")

            st.metric(
                "Occlusion status",
                result["occlusion"]
            )

            st.metric(
                "Occlusion confidence",
                f"{result['occlusion_confidence']:.2f}%"
            )

            st.progress(
                min(max(result["occlusion_confidence"] / 100, 0), 1),
                text="Occlusion probability"
            )

            if result["occlusion"] == "Occluded":
                st.warning(
                    "Obstruction flagged — sign interpretation may be unreliable."
                )
            else:
                st.success("No occlusion flagged by the model.")

    else:
        st.markdown("""
        <div class="panel" style="padding:25px;">
            <div class="eyebrow">CLASSIFICATION ENGINE</div>
            <h3>Awaiting inference</h3>
            <p class="muted">Predicted class and confidence will appear here.</p>
            <hr>
            <div class="eyebrow">OCCLUSION ENGINE</div>
            <h3>Awaiting image</h3>
            <p class="muted">Visibility assessment has not been generated.</p>
        </div>
        """, unsafe_allow_html=True)

# ANALYSIS FOOTER
st.markdown("---")
st.markdown("""
<div class="eyebrow">
MODEL PIPELINE · GTSRB · 43 CLASSES · CNN · SYNTHETIC OCCLUSION
</div>
""", unsafe_allow_html=True)