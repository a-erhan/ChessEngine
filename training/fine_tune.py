import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessTransformer
import time
import os

# --- AYARLAR ---
BASE_MODEL_PATH = "training/checkpoints/cvt_model_epoch_10.pt"
FINETUNE_PGN = "training/games_finetune.pgn"  # YENİ PGN DOSYASININ ADI
CHECKPOINT_DIR = "training/checkpoints_finetune"
BATCH_SIZE = 256
EPOCHS = 5
NEW_LR = 3e-5  # Cerrahi İnce Ayar (Eskisi 3e-4 idi)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def run_fine_tuning():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🚀 FAZ 1.5: FINE-TUNING BAŞLIYOR (Cihaz: {device})")
    print("="*60)
    
    # 1. Yeni Veri Setini Yükle
    print(f"📂 Yeni PGN okunuyor: {FINETUNE_PGN}")
    # Faz 1'deki orijinal dataset parametrelerini koruyoruz
    full_dataset = ChessDataset(FINETUNE_PGN, max_games=30000, min_elo=2000, min_moves=25)
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"✅ Dataloader hazır. Bir turda (Epoch) {len(dataloader)} batch işlenecek.\n")
    
    # 2. Şampiyon Modeli Ayağa Kaldır
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print(f"🧠 Faz 1 Şampiyonu ({BASE_MODEL_PATH}) başarıyla yüklendi!")
    
    # 3. Yeni, Düşük Learning Rate ile Optimizer Ayarla
    optimizer = optim.AdamW(model.parameters(), lr=NEW_LR, weight_decay=1e-4)
    
    model.train()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        # 🚨 HATA BURADAYDI: Artık orijinal koddaki gibi 'batch' sözlüğü üzerinden ilerliyoruz
        for batch_idx, batch in enumerate(dataloader):
            boards = batch["board"].to(device)
            move_targets = batch["move_target"].to(device)
            value_targets = batch["value_target"].to(device)
            
            optimizer.zero_grad()
            
            # Modelden Çıktıları Al
            policy_logits, values = model(boards)
            
            # Orijinal Kayıp Hesaplaması
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            value_loss = F.mse_loss(values.view(-1), value_targets)
            loss = policy_loss + 0.5 * value_loss
            
            # Geri Yayılım
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Fine-Tune Epoch {epoch+1:02d} | Batch {batch_idx+1}/{len(dataloader)} | Anlık Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print(f"🏁 Fine-Tune Epoch {epoch+1:02d} Tamamlandı! | Ortalama Loss: {avg_loss:.4f} | Süre: {elapsed:.1f} sn")
        
        # Checkpoint Kaydet
        save_path = os.path.join(CHECKPOINT_DIR, f"cvt_finetune_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint kaydedildi: {save_path}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    run_fine_tuning()