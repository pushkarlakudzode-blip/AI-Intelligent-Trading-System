# Deployment Guide — AI Intelligent Trading System

This guide provides complete instructions for deploying the **AI-Based Intelligent Trading System** to **Streamlit Community Cloud** (recommended, 100% free cloud hosting with zero infrastructure management) as well as self-hosted **Docker**.

---

## 🚀 Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the official, free hosting platform for Streamlit applications. It connects directly to your GitHub repository and automatically deploys changes whenever you push new commits.

### Step 1: Push Project to GitHub

1. Ensure the local Git repository is initialized and committed (this has already been prepared):
   ```bash
   git status
   ```

2. Create a **New Repository** on [GitHub](https://github.com/new):
   - **Repository name**: `AI-Intelligent-Trading-System` (or your preferred name)
   - **Visibility**: Public (recommended for free Streamlit Cloud deployment) or Private
   - **Initialize with README/gitignore**: Leave unchecked (already initialized locally)

3. Link your local project to your GitHub remote and push:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/AI-Intelligent-Trading-System.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Deploy on Streamlit Community Cloud

1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
2. Click **"New app"** (or "Create app").
3. Configure the deployment settings:
   - **Repository**: `<YOUR_GITHUB_USERNAME>/AI-Intelligent-Trading-System`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL**: Choose a custom subdomain (e.g., `ai-trading-terminal.streamlit.app`)
4. *(Optional)* Configure API Secrets:
   - Click **"Advanced settings..."** -> **"Secrets"**.
   - Paste any API keys you wish to use (formatted as TOML):
     ```toml
     ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"
     ALPACA_API_KEY_ID = "your_alpaca_key_id"
     ALPACA_API_SECRET_KEY = "your_alpaca_secret_key"
     ```
   - These are automatically bridged to `os.environ` inside `dashboard/app.py`.
5. Click **"Deploy!"**.

Streamlit Cloud will provision the container, install packages from `requirements.txt`, load the pre-computed machine learning pipeline datasets from `data/processed/`, apply the dark terminal theme from `.streamlit/config.toml`, and make your live terminal available worldwide!

---

## 🐳 Option 2: Docker Container Deployment

If you want to host on a VPS, AWS EC2, DigitalOcean Droplet, Google Cloud Run, or local server:

### Run with Docker Compose:
```bash
docker compose up -d --build
```

### Or with Docker CLI:
```bash
docker build -t ai-trading-terminal .
docker run -d -p 8501:8501 --name ai-trading-terminal ai-trading-terminal
```

Access the terminal at `http://localhost:8501` or `http://<server-ip>:8501`.

---

## 📁 Key Deployment Files

| File | Purpose |
|---|---|
| [`dashboard/app.py`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/dashboard/app.py) | Main Streamlit application entry point |
| [`requirements.txt`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/requirements.txt) | Python dependencies installed automatically during build |
| [`.streamlit/config.toml`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/.streamlit/config.toml) | Dark terminal theme styling and headless server settings |
| [`.gitignore`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/.gitignore) | Keeps repo lightweight (<25 MB) by excluding raw staging dumps |
| [`Dockerfile`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/Dockerfile) | Production container definition for custom server hosting |
| [`docker-compose.yml`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/docker-compose.yml) | Compose orchestrator for container lifecycle |
| [`data/processed/`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/data/processed) | Essential processed analytics, PCA, regression, clustering, RL, and decision fusion CSVs |
