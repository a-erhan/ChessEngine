# ChessEngine AI ♟

Kendi eğittiğim derin öğrenme modelleriyle satranç oynayabileceğiniz web uygulaması.

## 🎮 Özellikler

- **3 Farklı AI Modeli**
  - 🔥 Agresif — Saldırgan oynama stili
  - 🛡️ Tedbirli — Savunma odaklı
  - 🧠 Akıl — Dengeli strateji

- **Oyun Modları**
  - İnsan vs Bot (Beyaz veya Siyah oyna)
  - Bot vs Bot (iki modeli izle, hız kontrolü)

## 🏗️ Mimari

```
frontend/ (GitHub Pages)  →  backend/ (Render.com)
     HTML/CSS/JS               Flask + PyTorch
```

## 🚀 Lokal Kurulum

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
# Doğrudan tarayıcıda aç
open frontend/index.html
```

## 🌐 Production Deploy

### Backend → Render.com
1. [render.com](https://render.com) hesabı oluşturun
2. "New Web Service" → GitHub reponuzu bağlayın
3. `render.yaml` otomatik algılanır

### Frontend → GitHub Pages
1. Repo Ayarları → Pages → Source: GitHub Actions
2. Her `git push` otomatik deploy eder

### ⚠️ Render URL Güncelleme
`frontend/game.js` içindeki `API_BASE` değişkenini Render URL'iniz ile güncelleyin:
```javascript
const API_BASE = 'https://chess-engine-api.onrender.com';
```

## 🧠 Model Bilgisi

| Model | Dosya | Açıklama |
|-------|-------|----------|
| Agresif | `training/checkpoints/cvt_model_epoch_10.pt` | Phase 1 eğitimi |
| Tedbirli | `training/checkpoints_finetune/cvt_finetune_epoch_5.pt` | Fine-tune |
| Akıl | `training/checkpoints_ultra/cvt_ultra_epoch_3.pt` | Ultra fine-tune |

Mimari: CNN + Transformer (CvT), 256d, 8 head, 6 layer, 4096 policy output