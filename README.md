# ChessEngine AI

A web application where you can play chess against custom trained deep learning models.

## Features

- **3 Different AI Models**
  - Aggressive — Offensive playstyle
  - Cautious — Defense-oriented
  - Wise — Balanced strategy

- **Game Modes**
  - Human vs Bot (Play as White or Black)
  - Bot vs Bot (Watch two models play, speed control)

## Architecture

```text
frontend/ (GitHub Pages)  →  backend/ (Render.com)
     HTML/CSS/JS               Flask + PyTorch
```

## Local Installation

```bash
# Clone the repository and navigate to the directory
cd ChessEngine

# Install backend dependencies
pip install -r backend/requirements.txt

# Start the Flask server
python backend/app.py
```

After starting the server, go to `http://localhost:5001` in your browser.

## Production Deployment

### Backend → Render.com
1. Create an account on [render.com](https://render.com)
2. "New Web Service" → Connect your GitHub repository
3. The `render.yaml` file will be automatically detected

### Frontend → GitHub Pages
1. Repository Settings → Pages → Source: GitHub Actions
2. Every `git push` automatically deploys the frontend

## Model Information

| Model | File | Description |
|-------|------|-------------|
| Aggressive | `training/checkpoints/cvt_model_epoch_10.pt` | Phase 1 training |
| Cautious | `training/checkpoints_finetune/cvt_finetune_epoch_5.pt` | Fine-tune |
| Wise | `training/checkpoints_ultra/cvt_ultra_epoch_3.pt` | Ultra fine-tune |

Architecture: CNN + Transformer (CvT), 256d, 8 head, 6 layer, 4096 policy output