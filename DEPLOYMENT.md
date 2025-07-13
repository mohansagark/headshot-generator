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

## 🔧 Troubleshooting Deployment Issues

### Common Deployment Problems

#### 1. "Unable to locate package" Error
**Problem**: System packages not found in `packages.txt`
**Solution**: 
- Ensure `packages.txt` contains system dependencies:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  libsm6
  libxext6
  ```

#### 2. Memory/Resource Errors
**Problem**: AI models too large for free tier
**Solutions**:
- Use `requirements-minimal.txt` instead of `requirements-deploy.txt`
- Upgrade to paid tier for more memory
- Try alternative platforms (Railway, Render)

#### 3. Import/Dependency Errors
**Problem**: Package version conflicts
**Solutions**:
- Check specific error in deployment logs
- Try `requirements-minimal.txt` for basic functionality
- Restart deployment after code changes

### Alternative Requirements Files

#### Standard Deployment (2GB RAM)
File: `requirements-deploy.txt`
- ✅ AI headshot generation
- ✅ Background removal
- ✅ Image enhancement
- ❌ Advanced face detection

#### Minimal Deployment (500MB RAM)
File: `requirements-minimal.txt`
- ❌ AI generation (too resource intensive)
- ✅ Background removal
- ✅ Basic image processing
- ✅ File upload/download

#### Full Local Development (4GB+ RAM)
File: `requirements.txt`
- ✅ All features
- ✅ Face detection
- ✅ Advanced AI models
- ✅ Batch processing

### Platform-Specific Instructions

#### Streamlit Cloud
1. In advanced settings, try different requirements files:
   - First: `requirements-deploy.txt`
   - If failed: `requirements-minimal.txt`
2. Python version: `3.11`
3. Check logs for specific errors

#### Heroku
1. Add `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Use `requirements-deploy.txt`
3. Enable hobby dyno for more memory

#### Railway
1. Auto-detects Python app
2. Set start command: `streamlit run app.py --server.port=$PORT`
3. Use `requirements-deploy.txt`

### Deployment Status Indicators

When successfully deployed, you should see:
- 🌐 "Running on [Platform]" message
- ✅ "Headshot generator initialized successfully!"
- ⚠️ Optional compatibility warnings (normal for cloud deployment)

## 📞 Getting Help

If deployment still fails:
1. Check deployment logs for specific errors
2. Try the minimal requirements file
3. Open an issue on GitHub with logs
4. Join our Discord for real-time help

---
