print("🚨 DOSYA BAŞARIYLA OKUNDU! EĞER BUNU GÖRÜYORSAN SORUN İMPORTLARDA VEYA KÜTÜPHANEDE.")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessTransformer
import time
import os

def run_phase1_training():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🚀 FAZ 1: AĞIR SİKLET CvT EĞİTİMİ BAŞLIYOR (Cihaz: {device})")
    print("="*60)

    # Modelin C++ motoruna veya sonraya aktarılması için kayıt klasörü
    os.makedirs("training/checkpoints", exist_ok=True)

    # 1. Modeli Başlat (CvT Mimarisi)
    model = ChessTransformer(
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)

    # 2. Veriyi Çek (Artık Subset yok, tam kapasite yükleniyoruz)
    test_pgn = "training/games.pgn" 
    
    # dataset.py'deki 'tqdm' radarı burada devreye girecek
    print(f"📂 Veri yükleniyor... (Lütfen bekleyin, bu işlem biraz sürebilir)")
    full_dataset = ChessDataset(test_pgn, max_games=30000, min_elo=2000, min_moves=25)
    
    # Mac M4 Max için ideal batch size (GPU'yu tam doldurur ama taşırmaz)
    dataloader = DataLoader(full_dataset, batch_size=256, shuffle=True)
    
    print(f"✅ Dataloader hazır. Bir turda (Epoch) {len(dataloader)} batch işlenecek.\n")

    # 3. Zırhlı Optimizer (Karpathy Sabiti)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    epochs = 10 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            boards = batch["board"].to(device)
            move_targets = batch["move_target"].to(device)
            value_targets = batch["value_target"].to(device) # Gerçek maç sonucu!
            
            # 🚨 Modelden Hamle ve Değerlendirme Çek
            policy_logits, values = model(boards)
            
            # Kayıpları Hesapla
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            value_loss = F.mse_loss(values.view(-1), value_targets)
            
            # Toplam Kayıp
            loss = policy_loss + 0.5 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradyan Zırhı
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Her 100 batch'te bir durum raporu ver
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:02d} | Batch {batch_idx+1}/{len(dataloader)} | Anlık Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        
        print("-" * 60)
        print(f"🏁 Epoch {epoch+1:02d} Tamamlandı! | Ortalama Loss: {avg_loss:.4f} | Süre: {epoch_time:.1f} saniye")
        
        # Her epoch sonunda modeli yedekle
        checkpoint_path = f"training/checkpoints/cvt_model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint kaydedildi: {checkpoint_path}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    run_phase1_training()