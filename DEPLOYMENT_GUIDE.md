# Free & Instant Deployment Guide
## Deploying the Indian Sign Language (ISL) Interpreter Online

This guide walks you through deploying the complete project—**Frontend, Backend, and AI Models**—for **100% FREE** with step-by-step instructions.

---

## 🌟 Why Deployment is Fast & Lightweight
1. **Unified Full-Stack Architecture**: The FastAPI backend (`backend/main.py`) serves both the REST API endpoints and mounts the static frontend (`static/`). You only need to deploy **one single service**.
2. **Pre-Trained Lightweight Model**: The trained ensemble model (`isl_landmark_model.joblib`) has been compressed to just **13.7 MB**, and the MediaPipe bundle (`hand_landmarker.task`) is **7.8 MB**. 
3. **Optimized Repository**: The large raw image folders (`dataset_ISL/`, `ISL_Dataset/`) are excluded via `.gitignore`. The complete deployable codebase is **under 25 MB**, meaning it uploads and builds in under 2 minutes.
4. **Automatic HTTPS / SSL**: Browser webcam security (`getUserMedia`) **strictly requires an HTTPS connection**. All the deployment options below provide free, automatic SSL certificates out-of-the-box.

---

## Method 1: Hugging Face Spaces (Recommended for AI / ML Demos)
* **Cost**: 100% Free Forever
* **Specs**: 2 vCPUs, 16 GB RAM (CPU Basic tier)
* **Time to Deploy**: ~3 minutes
* **Webcam Support**: Full HTTPS enabled

### Step-by-Step Instructions:

1. **Create an Account**:
   - Go to [huggingface.co](https://huggingface.co/) and sign up / log in.

2. **Create a New Space**:
   - Click your profile icon in the top right $\to$ select **New Space** (or go to `https://huggingface.co/new-space`).
   - **Space name**: `isl-interpreter` (or any name you like).
   - **License**: `mit` or `apache-2.0`.
   - **Select the Space SDK**: Choose **Docker** $\to$ select **Blank** template.
   - **Space hardware**: `CPU Basic · 2 vCPU · 16GB RAM · Free`.
   - Set visibility to **Public** $\to$ click **Create Space**.

3. **Upload Your Files**:
   You can either push using Git or upload directly through the browser:
   
   **Option A: Uploading via Web Browser (Easiest)**:
   - On your newly created Space page, click the **Files** tab.
   - Click **Add file** $\to$ **Upload files**.
   - Drag and drop the following files & folders from `D:\Sign Language Interpreter`:
     - `Dockerfile`
     - `requirements.txt`
     - `hand_landmarker.task`
     - `isl_landmark_model.joblib`
     - `backend/` (folder containing `__init__.py`, `main.py`, `model_service.py`)
     - `static/` (folder containing `index.html`, `style.css`, `app.js`, `placeholder_hand.png`)
   - Scroll down and click **Commit changes to main**.

   **Option B: Uploading via Git Terminal**:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/isl-interpreter
   git add Dockerfile requirements.txt hand_landmarker.task isl_landmark_model.joblib backend/ static/
   git commit -m "Deploy ISL Interpreter to Hugging Face"
   git push -u origin main
   ```

4. **Launch & Share**:
   - Hugging Face will automatically detect the `Dockerfile`, build the container, and launch the app.
   - Within 1–2 minutes, the status will show **Running**.
   - Your live public URL will be:
     ```
     https://huggingface.co/spaces/YOUR_USERNAME/isl-interpreter
     ```
   - Anyone globally can open this link on their laptop or mobile browser and use the interpreter live!

---

## Method 2: Render.com (Free Cloud Web Service from GitHub)
* **Cost**: 100% Free
* **Specs**: 512 MB RAM, free HTTPS
* **Time to Deploy**: ~4 minutes

### Step-by-Step Instructions:

1. **Push Code to GitHub**:
   - Create a new free repository on [github.com](https://github.com) named `sign-language-interpreter`.
   - Open PowerShell in `D:\Sign Language Interpreter` and run:
     ```powershell
     git init
     git add .
     git commit -m "Initial commit for deployment"
     git branch -M main
     git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sign-language-interpreter.git
     git push -u origin main
     ```
   *(Note: Thanks to `.gitignore`, only the necessary ~25 MB of code and models are uploaded!)*

2. **Connect to Render**:
   - Go to [render.com](https://render.com/) and sign up with your GitHub account.
   - Click **New +** $\to$ select **Web Service**.
   - Choose **Build and deploy from a Git repository** $\to$ select your `sign-language-interpreter` repository.

3. **Configure Settings**:
   - **Name**: `isl-interpreter`
   - **Region**: Choose closest to you (e.g., Singapore, Frankfurt, or Ohio).
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan Type**: Select **Free**.
   - Click **Deploy Web Service**.

4. **Access Your Live App**:
   - Render will build the environment and display your live URL (e.g., `https://isl-interpreter.onrender.com`).
   - Open the URL in any browser to verify the camera and models.

---

## Method 3: Instant Live URL in 30 Seconds (Ideal for Stage Presentation)
If you are at the international forum and want to run the model on your laptop while giving judges and audience members an immediate public link to try on their smartphones:

### Option A: Using Localtunnel (Zero Account Needed, Free)
1. Make sure your local app is running (via `run.bat` or `python run_interpreter.py`).
2. Open a new terminal and run:
   ```powershell
   npx localtunnel --port 8000
   ```
3. It will output a public HTTPS link:
   ```
   your url is: https://fair-pandas-sing.loca.lt
   ```
4. Share this link or generate a free QR code on screen for the audience. Anyone who opens the link reaches your live interpreter running on your laptop!

### Option B: Using Cloudflare Tunnel (Free, Fast, Secure)
1. Download `cloudflared.exe` from [Cloudflare Releases](https://github.com/cloudflare/cloudflared/releases).
2. Run:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```
3. Cloudflare will output an instant secure URL:
   ```
   https://random-name-trycloudflare.com
   ```

### Option C: Using Ngrok (Free)
1. Download [ngrok](https://ngrok.com/) and run:
   ```powershell
   ngrok http 8000
   ```
2. Copy the `https://xxxx.ngrok-free.app` URL and open it on any device.

---

## Method 4: Koyeb (Alternative Free Container Hosting)
* **Cost**: 100% Free (Nano instance)
* **Deploy**: 
  1. Go to [koyeb.com](https://www.koyeb.com/) and sign up.
  2. Click **Create App** $\to$ select **GitHub**.
  3. Select your repository. Koyeb detects the `Dockerfile` automatically.
  4. Click **Deploy**. Koyeb will generate a public `https://...koyeb.app` URL.

---

## 🔒 Crucial Checklist for Camera Permissions Online

When accessing the web app over the internet:
1. **HTTPS is Mandatory**: Browsers automatically block webcam access over unencrypted HTTP (except on `localhost`). Ensure you always visit your deployment using `https://`.
2. **Browser Prompt**: When the page loads, click **Allow** on the camera permission pop-up.
3. **Mobile Browser Testing**: On iOS (Safari) and Android (Chrome), ensure the browser has permission in OS Settings to access the camera.

---

## 🎯 Summary Comparison

| Platform | Cost | RAM / CPU | Build Time | Best For |
|---|---|---|---|---|
| **Hugging Face Spaces** | Free | 16 GB RAM / 2 vCPU | ~2 min | **Best overall AI/ML showcase & permanent link** |
| **Render.com** | Free | 512 MB RAM / 0.5 vCPU | ~3 min | **Clean web service from GitHub** |
| **Localtunnel / Cloudflare** | Free | Uses your laptop hardware | **30 seconds** | **Live stage presentations & instant judge demos** |
| **Koyeb** | Free | 512 MB RAM / 0.1 vCPU | ~2 min | **Reliable Docker container hosting** |

