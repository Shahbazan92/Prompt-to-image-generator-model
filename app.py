"""
AI Text-to-Image Studio
-----------------------
Type a prompt -> get an image, powered by the Hugging Face Inference API.
Deployable on Streamlit Community Cloud.
"""

import io
import random
import time
import zipfile
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Text-to-Image Studio",
    page_icon="🎨",
    layout="wide",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODELS = {
    "FLUX.1 Schnell (fast, high quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 3.5 Large": "stabilityai/stable-diffusion-3.5-large",
    "FLUX.1 Dev": "black-forest-labs/FLUX.1-dev",
    "Custom model ID...": "custom",
}

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": "photorealistic, ultra detailed, 8k, sharp focus, natural lighting",
    "Digital Art": "digital art, vibrant colors, highly detailed, artstation trending",
    "Anime": "anime style, studio ghibli, soft colors, detailed background",
    "Oil Painting": "oil painting, textured brush strokes, classical art, rich colors",
    "3D Render": "3d render, octane render, unreal engine, cinematic lighting",
    "Cyberpunk": "cyberpunk, neon lights, futuristic city, dark atmosphere",
    "Watercolor": "watercolor painting, soft gradients, delicate, pastel tones",
    "Pencil Sketch": "pencil sketch, hand drawn, monochrome, fine linework",
}

SIZE_PRESETS = {
    "Square 1024x1024": (1024, 1024),
    "Square 768x768": (768, 768),
    "Square 512x512": (512, 512),
    "Portrait 768x1024": (768, 1024),
    "Landscape 1024x768": (1024, 768),
    "Wide 1280x720": (1280, 720),
}

PROVIDERS = ["auto", "hf-inference", "fal-ai", "together", "replicate"]

DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, extra fingers, watermark, text"

EXAMPLE_PROMPTS = [
    "A astronaut riding a horse on Mars, cinematic lighting",
    "A cozy cabin in a snowy forest at night, warm glowing windows",
    "A futuristic city floating in the clouds at sunset",
    "A cute robot reading a book in a library, soft light",
]

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: image bytes, prompt, meta
if "prompt" not in st.session_state:
    st.session_state.prompt = ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_token() -> str:
    """Token priority: Streamlit secrets -> sidebar input."""
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return st.session_state.get("manual_token", "")


def image_to_bytes(image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def generate_image(token, provider, model_id, prompt, negative, width, height,
                   steps, guidance, seed):
    """Call the HF Inference API and return a PIL image."""
    client = InferenceClient(provider=provider, api_key=token)

    kwargs = {"model": model_id, "width": width, "height": height}
    if negative.strip():
        kwargs["negative_prompt"] = negative.strip()
    if steps:
        kwargs["num_inference_steps"] = steps
    if guidance is not None:
        kwargs["guidance_scale"] = guidance
    if seed is not None:
        kwargs["seed"] = seed

    try:
        return client.text_to_image(prompt, **kwargs)
    except TypeError:
        # Some providers don't accept every parameter - retry with basics
        return client.text_to_image(prompt, model=model_id)


def friendly_error(err: Exception) -> str:
    msg = str(err)
    low = msg.lower()
    if "401" in low or "unauthorized" in low or "invalid" in low and "token" in low:
        return "Invalid or missing Hugging Face token. Check your token in the sidebar or secrets."
    if "402" in low or "credit" in low or "quota" in low or "rate limit" in low or "429" in low:
        return "Free quota / rate limit reached. Wait a bit, try a different model/provider, or use a smaller image."
    if "503" in low or "loading" in low or "currently" in low:
        return "The model is loading or busy. Please try again in 20-30 seconds."
    if "404" in low or "not found" in low or "not supported" in low:
        return "This model isn't available on the selected provider. Try another model or set provider to 'auto'."
    return f"Generation failed: {msg}"


# --------------------------------------------------------------------------
# Sidebar - settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    # Token
    has_secret = False
    try:
        has_secret = "HF_TOKEN" in st.secrets
    except Exception:
        pass

    if has_secret:
        st.success("HF token loaded from secrets ✅")
    else:
        st.text_input(
            "Hugging Face Token",
            type="password",
            key="manual_token",
            help="Free token from huggingface.co/settings/tokens (role: read / inference).",
        )
        st.caption("[Get a free token](https://huggingface.co/settings/tokens)")

    st.divider()

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model_id = MODELS[model_label]
    if model_id == "custom":
        model_id = st.text_input("Custom model ID", placeholder="author/model-name")

    provider = st.selectbox("Provider", PROVIDERS, help="'auto' lets Hugging Face choose.")

    style = st.selectbox("Style preset", list(STYLE_PRESETS.keys()))
    size_label = st.selectbox("Image size", list(SIZE_PRESETS.keys()))
    width, height = SIZE_PRESETS[size_label]

    num_images = st.slider("Number of images", 1, 4, 1)

    with st.expander("Advanced"):
        steps = st.slider("Inference steps", 1, 50, 25,
                          help="FLUX Schnell works great with 4 steps.")
        guidance = st.slider("Guidance scale", 0.0, 15.0, 7.5, 0.5)
        use_random_seed = st.checkbox("Random seed", value=True)
        seed_value = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=use_random_seed)
        negative = st.text_area("Negative prompt", value=DEFAULT_NEGATIVE, height=80)

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🎨 AI Text-to-Image Studio")
st.caption("Describe anything you can imagine and let AI paint it. Powered by Hugging Face.")

# Example prompt buttons
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLE_PROMPTS))
for col, example in zip(cols, EXAMPLE_PROMPTS):
    if col.button(example[:28] + "...", use_container_width=True, key=f"ex_{example}"):
        st.session_state.prompt = example

prompt = st.text_area(
    "Your prompt",
    key="prompt",
    height=100,
    placeholder="e.g. A majestic lion wearing a golden crown, digital art",
)

generate = st.button("✨ Generate", type="primary", use_container_width=True)

if generate:
    token = get_token()
    if not token:
        st.error("Please enter your Hugging Face token in the sidebar.")
    elif not prompt.strip():
        st.warning("Please write a prompt first.")
    elif not model_id:
        st.warning("Please enter a model ID.")
    else:
        full_prompt = prompt.strip()
        if STYLE_PRESETS[style]:
            full_prompt += ", " + STYLE_PRESETS[style]

        results = []
        progress = st.progress(0, text="Starting...")
        start = time.time()

        for i in range(num_images):
            progress.progress(i / num_images, text=f"Generating image {i + 1} of {num_images}...")
            seed = random.randint(0, 2**31 - 1) if use_random_seed else int(seed_value) + i
            try:
                img = generate_image(
                    token, provider, model_id, full_prompt, negative,
                    width, height, steps, guidance, seed,
                )
                item = {
                    "bytes": image_to_bytes(img),
                    "prompt": prompt.strip(),
                    "model": model_id,
                    "style": style,
                    "size": f"{width}x{height}",
                    "seed": seed,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                results.append(item)
                st.session_state.history.insert(0, item)
            except Exception as e:
                st.error(friendly_error(e))
                break

        progress.progress(1.0, text="Done!")
        elapsed = time.time() - start

        if results:
            st.success(f"Generated {len(results)} image(s) in {elapsed:.1f}s")
            out_cols = st.columns(min(len(results), 2))
            for idx, item in enumerate(results):
                with out_cols[idx % len(out_cols)]:
                    st.image(item["bytes"], caption=f"Seed: {item['seed']}", use_container_width=True)
                    st.download_button(
                        "⬇️ Download PNG",
                        data=item["bytes"],
                        file_name=f"ai_image_{item['seed']}.png",
                        mime="image/png",
                        key=f"dl_new_{idx}_{item['seed']}",
                        use_container_width=True,
                    )

# --------------------------------------------------------------------------
# History gallery
# --------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    head_col, zip_col = st.columns([3, 1])
    head_col.subheader("🖼️ History")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for n, item in enumerate(st.session_state.history):
            zf.writestr(f"image_{n + 1}_{item['seed']}.png", item["bytes"])
    zip_col.download_button(
        "📦 Download all (ZIP)",
        data=zip_buf.getvalue(),
        file_name="ai_images.zip",
        mime="application/zip",
        use_container_width=True,
    )

    gal = st.columns(3)
    for n, item in enumerate(st.session_state.history[:12]):
        with gal[n % 3]:
            st.image(item["bytes"], use_container_width=True)
            st.caption(f"**{item['prompt'][:60]}**  \n{item['model'].split('/')[-1]} · {item['size']} · seed {item['seed']} · {item['time']}")
            st.button(
                "♻️ Reuse prompt",
                key=f"reuse_{n}",
                on_click=lambda p=item["prompt"]: st.session_state.update(prompt=p),
            )

st.divider()
st.caption("Tip: detailed prompts (subject, style, lighting, mood) give much better results.")
