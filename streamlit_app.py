import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import pathlib
# Load .env from project root (parent of this file)
load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")
import pyvista as pv

# Curated set of 3D models
PRESET_MODELS = {
    "Stanford Bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/bunny.obj",
  "Teapot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj",
  "Cow": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/cow.obj",
  "Dragon": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/dragon.obj",
  "Fandisk": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/fandisk.obj",
  "Torus": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/torus.obj",
  "Sphere": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/sphere.obj",
  "Cube": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/cube.obj",
  "Armchair": "https://raw.githubusercontent.com/mikedh/trimesh/master/models/featuretype/armchair.obj",
  "Fuze Model": "https://raw.githubusercontent.com/mikedh/trimesh/master/models/fuze.obj"
}

def show_3d_model_from_url(model_url):
    local_path = "temp.obj"
    try:
        response = requests.get(model_url)
        st.write("Download status:", response.status_code)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            st.success("3D model downloaded!")
            st.download_button("Download 3D Model (.obj)", open(local_path, "rb"), file_name="model.obj")
            st.markdown(
                "👉 [Open model in 3dviewer.net](https://3dviewer.net/) for interactive viewing. "
                "Just upload the downloaded file there!"
            )
        else:
            st.error("Could not download the .obj file. Please check the URL.")
    except Exception as e:
        st.error(f"3D model error: {e}")

def _get_hf_token():
    """Support multiple env var names for Hugging Face API key."""
    return (
        os.getenv("Hugging_api")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )


def enhance_image_prompt(user_prompt: str) -> str:
    return f"""
clean educational illustration, high quality, polished composition,
clear subject, neat layout, detailed but not cluttered,
textbook-style visual, white or simple background,
sharp focus, professional rendering.

Topic: {user_prompt.strip()}
""".strip()


def generate_image(prompt):
    token = _get_hf_token()
    if not token or not token.strip():
        st.error("Hugging Face API key not set. Add Hugging_api, HUGGINGFACE_TOKEN, or HF_TOKEN to your .env file.")
        return None

    try:
        from huggingface_hub import InferenceClient, InferenceTimeoutError

        client = InferenceClient(
            token=token.strip(),
            timeout=90
        )

        improved_prompt = enhance_image_prompt(prompt)

        image = client.text_to_image(
            improved_prompt,
            model="black-forest-labs/FLUX.1-schnell",
            guidance_scale=7.0,
            num_inference_steps=30,
            negative_prompt=(
                "blurry, low quality, distorted, gibberish text, watermark, "
                "extra objects, deformed anatomy, messy composition, cropped"
            ),
        )

        if image is None:
            st.error("No image returned. Try a different prompt or check model access.")
            return None

        return image

    except InferenceTimeoutError:
        st.error("Image generation timed out. Try a shorter prompt or try again.")
        return None

    except Exception as e:
        err_msg = str(e).lower()

        if "401" in err_msg or "unauthorized" in err_msg:
            st.error("Invalid Hugging Face API key. Check your token at https://huggingface.co/settings/tokens")
        elif "403" in err_msg or "forbidden" in err_msg:
            st.error("Access denied. Enable Inference permissions for the token or try another model.")
        elif "503" in err_msg or "loading" in err_msg:
            st.error("Model is loading. Wait a minute and try again.")
        else:
            st.error(f"Image generation failed: {e}")

        return None

def plot_expression(expr, x_min=-10, x_max=10):
    x = np.linspace(x_min, x_max, 400)
    # WARNING: eval is dangerous if not sanitized; for demo only!
    y = eval(expr, {"x": x, "np": np})
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"y = {expr}")
    return fig

st.title("AI STEM Tutor")

st.write("Ask any question (STEM or general):")

# Use session_state to preserve input
if "question" not in st.session_state:
    st.session_state["question"] = ""

def submit():
    st.session_state["question"] = st.session_state["input_question"]

agent = st.selectbox(
    "Choose agent (or leave as Auto):",
    ["Auto (Detect)", "General", "Math", "Retrieve", "Visuals"]
)

if agent == "Visuals":
    visual_type = st.radio("What do you want to see?", ["Image/Diagram", "3D Model", "Graph/Chart"])
    if visual_type == "3D Model":
        preset_choice = st.selectbox("Or select a sample 3D model:", ["None"] + list(PRESET_MODELS.keys()))
        if preset_choice != "None":
            user_input = PRESET_MODELS[preset_choice]
        else:
            user_input = st.text_input("Paste a direct .obj file URL (must end with .obj)", value="", key="input_question", on_change=submit)
    else:
        if visual_type == "Image/Diagram":
            input_label = "Describe the diagram you want to see (e.g., 'photosynthesis diagram')"
        elif visual_type == "Graph/Chart":
            input_label = "Type a math expression to plot (e.g., 'x**2', 'np.sin(x)')"
        user_input = st.text_input(input_label, value=st.session_state["question"], key="input_question", on_change=submit)
    if st.button("Show Visual"):
        if visual_type == "Image/Diagram":
            with st.spinner("Generating image..."):
                img = generate_image(user_input)
                if img:
                    st.image(img, caption=f"Generated: {user_input}")
                # Else: generate_image() already shows a specific error (key missing, 401, 403, 503, etc.)
        elif visual_type == "3D Model":
            if user_input and user_input.startswith("http") and user_input.endswith(".obj"):
                show_3d_model_from_url(user_input)
            else:
                st.info("Paste a direct .obj file URL from Hugging Face or another source (must end with .obj), or select a sample above.")
        elif visual_type == "Graph/Chart":
            expr = user_input.replace("plot", "").replace("y=", "").strip()
            try:
                fig = plot_expression(expr)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not plot the expression. Error: {e}")
else:
    input_label = "Your question"
    user_input = st.text_input(input_label, value=st.session_state["question"], key="input_question", on_change=submit)
    if st.button("Ask"):
        endpoint = "http://localhost:8001/ask"
        payload = {"user_id": "streamlit_user", "question": user_input}
        if agent == "Math":
            endpoint = "http://localhost:8001/solve"
            payload = {"user_id": "streamlit_user", "problem": user_input}
        elif agent == "Retrieve":
            endpoint = "http://localhost:8001/retrieve"
            payload = {"user_id": "streamlit_user", "query": user_input, "top_k": 3}
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            st.subheader("Answer:")
            st.write(data.get("answer") or data.get("message"))
            if data.get("steps"):
                st.markdown("**Steps:**")
                for step in data["steps"]:
                    st.write("-", step)
            if data.get("sources"):
                st.markdown("**Sources:**")
                for src in data["sources"]:
                    st.write(src)
        except Exception as e:
            st.error(f"Error: {e}")

