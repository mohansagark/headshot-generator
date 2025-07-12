# Deployment Guide for Professional Headshot Generator

## Quick Deploy Options

### 1. Streamlit Cloud (Free & Recommended)

- URL: https://share.streamlit.io
- Repository: mohansagark/headshot-generator
- Branch: main
- Main file: app.py
- Requirements: requirements-deploy.txt

### 2. Hugging Face Spaces

```bash
# Clone your repo to HF Spaces
git clone https://github.com/mohansagark/headshot-generator.git
cd headshot-generator
pip install huggingface_hub
# Upload to HF Spaces following their guide
```

### 3. Railway

- URL: https://railway.app
- Connect GitHub repo: mohansagark/headshot-generator
- Auto-deploy from main branch

### 4. Render

- URL: https://render.com
- Connect GitHub repo
- Build command: pip install -r requirements-deploy.txt
- Start command: streamlit run app.py --server.port=$PORT

## Environment Variables (if needed)

```
TORCH_HOME=/tmp/.torch
TRANSFORMERS_CACHE=/tmp/.transformers
DIFFUSERS_CACHE=/tmp/.diffusers
```

## Performance Notes

- First startup: 5-10 minutes (downloading models)
- Subsequent runs: 30-60 seconds
- Memory usage: ~2-4GB (recommend paid tier for better performance)
