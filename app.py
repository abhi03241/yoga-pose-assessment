"""Yoga Pose Checker - upload/take a photo and get posture feedback.
Run:  streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from pose_utils import AUTO, POSES, PoseEstimator, analyze, draw_overlay

st.set_page_config(page_title="Yoga Pose Checker", page_icon="🧘", layout="wide")
st.title("🧘 Yoga Pose Checker")
st.caption("Upload a photo of your pose. I'll draw your skeleton, measure your joint angles and tell you what to fix.")

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Which pose are you doing?", [AUTO] + list(POSES))
    front = [n for n, s in POSES.items() if s["view"] == "front"]
    side = [n for n, s in POSES.items() if s["view"] == "side"]
    with st.expander("📸 Photo tips", expanded=True):
        st.markdown(
            "- Whole body in the frame, head to feet\n"
            "- Phone ~2-3 m away, at hip height, held straight\n"
            "- Plain background, good light, fitted clothing\n"
            f"- **Face the camera** for: {', '.join(front)}\n"
            f"- **Shoot from the side** for: {', '.join(side)}"
        )
    with st.expander(f"Supported poses ({len(POSES)})"):
        for name, spec in POSES.items():
            st.markdown(f"**{name}** ({spec['view']} view) – {spec['tips']}")


def load_image(file) -> np.ndarray:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")  # fixes rotated phone photos
    img.thumbnail((1280, 1280))
    return np.array(img)


source = st.radio("Photo source", ["Upload a photo", "Take a photo now"], horizontal=True)
file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"]) \
    if source == "Upload a photo" else st.camera_input("Take a photo")

if file is None:
    st.info("👆 Add a photo to begin.")
    st.stop()

image = load_image(file)
with st.spinner("Analysing your pose..."):
    with PoseEstimator(static=True) as est:
        result = analyze(image, choice, est)

if result is None:
    st.error("No person detected. Make sure your whole body is visible and well lit, then try again.")
    st.image(image, use_container_width=True)
    st.stop()

if result["evaluation"] is None:
    st.warning("I found a body but too few joints are visible to judge the pose. "
               "Step back so your full body fits in the photo.")
    st.image(draw_overlay(image, result), use_container_width=True)
    st.stop()

ev = result["evaluation"]
left, right = st.columns([3, 2])

with left:
    st.image(draw_overlay(image, result), use_container_width=True,
             caption="Green = joint is within range, red = needs correction (numbers are angles in degrees)")

with right:
    st.subheader(result["pose"])
    st.metric("Pose accuracy", f"{ev['overall']:.0f}%")
    st.progress(int(ev["overall"]))
    if ev["n_visible"] < ev["n_total"]:
        st.caption(f"⚠️ Only {ev['n_visible']} of {ev['n_total']} joints were clearly visible; "
                   "the score may be less reliable.")

    if choice == AUTO and len(result["ranking"]) > 1:
        others = ", ".join(f"{n} ({e['overall']:.0f}%)" for n, e in result["ranking"][:3])
        st.caption(f"Closest matches: {others}")
        if not result["confident"]:
            st.info("Not sure this is the right pose (few joints visible or a close runner-up). "
                    "Select your pose manually in the sidebar for accurate feedback.")

    st.subheader("Feedback")
    if result["feedback"]:
        for line in result["feedback"]:
            st.warning(line)
    else:
        st.success("Great form! All measured joints are within the target range. 🎉")

    with st.expander("Joint angle details"):
        rows = [{
            "Joint": r["joint"].replace("_", " "),
            "Your angle": "n/a" if r["angle"] is None else f"{r['angle']:.0f}°",
            "Target": f"{r['target']}° ± {r['tol']}°",
            "Status": "not visible" if r["ok"] is None else ("✅" if r["ok"] else "❌"),
        } for r in ev["results"]]
        st.table(rows)

st.caption("Angles are measured in the 2D photo, so camera angle affects results. "
           "This is a guide, not professional instruction.")
