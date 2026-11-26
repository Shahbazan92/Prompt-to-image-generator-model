# AI Text-to-Image Studio

Streamlit app that turns a text prompt into an image using the Hugging Face Inference API.

## Features
- Multiple models (FLUX.1 Schnell, SDXL, SD 3.5, custom model ID)
- Style presets, negative prompt, size presets
- Steps, guidance scale, seed control
- Up to 4 images per run, history gallery, PNG / ZIP download
- Friendly error messages

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste your free Hugging Face token in the sidebar (or put it in `.streamlit/secrets.toml`).

## Get a token
https://huggingface.co/settings/tokens -> New token -> type "Read" (or fine-grained with Inference permission).

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (do NOT commit secrets.toml).
2. Go to share.streamlit.io -> New app -> select repo, branch, `app.py`.
3. Advanced settings -> Secrets -> add: `HF_TOKEN = "hf_your_token"`
4. Deploy.
