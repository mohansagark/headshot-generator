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

#### 3. dlib Compilation Errors (CMAKE)

**Problem**: "CMake is not installed on your system!" during deployment
**Root Cause**: dlib package requires compilation with CMake and build tools

**Solutions (in order of preference)**:

1. **Use Basic Compatible Version**:

   - Change main file to: `app-basic.py`
   - Change requirements to: `requirements-streamlit.txt`
   - ✅ Guaranteed to work, no compilation needed

2. **Add Build Tools** (current attempt):

   - Ensure `packages.txt` contains:
     ```
     libgl1-mesa-glx
     libglib2.0-0
     cmake
     build-essential
     ```
   - ⚠️ May still fail due to cloud environment limitations

3. **Use Minimal Requirements**:

   - Change requirements to: `requirements-minimal.txt`
   - Removes dlib dependency entirely
   - ✅ Most features preserved without compilation

4. **Switch to Alternative Platform**:
   - Railway, Render, or Heroku may have better build tool support
   - ✅ Full feature compatibility possible

### Multiple Deployment Strategies

#### Strategy 1: Full Featured (Local/Powerful Servers)

**Files**: `app.py` + `requirements.txt`
**Memory**: 4GB+ RAM required
**Features**:

- ✅ All AI models and face detection
- ✅ Advanced image processing
- ✅ Batch processing capabilities
- ✅ Professional quality results

#### Strategy 2: Cloud Optimized (Standard Cloud Deployment)

**Files**: `app.py` + `requirements-deploy.txt`
**Memory**: 2GB RAM recommended
**Features**:

- ✅ AI headshot generation
- ✅ Background removal
- ✅ Image enhancement
- ❌ Advanced face detection (may cause compilation issues)

#### Strategy 3: Minimal AI (Lightweight Cloud)

**Files**: `app.py` + `requirements-minimal.txt`
**Memory**: 1GB RAM
**Features**:

- ❌ AI generation (too resource intensive)
- ✅ Background removal
- ✅ Basic image processing
- ✅ File upload/download

#### Strategy 4: Basic Compatible (Guaranteed Cloud Success)

**Files**: `app-basic.py` + `requirements-streamlit.txt`
**Memory**: 512MB RAM
**Features**:

- ❌ No AI models (maximum compatibility)
- ✅ Basic image enhancement
- ✅ Professional image processing
- ✅ 100% Streamlit Cloud compatible
- ✅ No compilation dependencies

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
