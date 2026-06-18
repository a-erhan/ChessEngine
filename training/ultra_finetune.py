import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessTransformer
import time
import os

# --- MİKROSKOBİK AYARLAR ---
BASE_MODEL_PATH = "training/checkpoints_finetune/cvt_finetune_epoch_5.pt"
ENDGAME_PGN = "training/pure_endgames.pgn"
CHECKPOINT_DIR = "training/checkpoints_ultra"
BATCH_SIZE = 256
EPOCHS = 3
ULTRA_LR = 1e-6  # Milimetrik Alpha: 1.0 x 10^-6 (Sadece lokal minimumu cilalar)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def run_ultra_tuning():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🔬 FAZ 1.75: MİKROSKOBİK ENDGAME CİLASI BAŞLIYOR (Cihaz: {device})")
    print("="*60)
    
    dataset = ChessDataset(ENDGAME_PGN, max_games=50000, min_moves=0) # Zaten kırpılmış veri
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print(f"🧠 Faz 1.5 Şampiyonu başarıyla yatırıldı. (Alpha: {ULTRA_LR})")
    
    optimizer = optim.AdamW(model.parameters(), lr=ULTRA_LR, weight_decay=1e-5)
    model.train()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            boards = batch["board"].to(device)
            move_targets = batch["move_target"].to(device)
            value_targets = batch["value_target"].to(device)
            
            optimizer.zero_grad()
            policy_logits, values = model(boards)
            
            loss = F.cross_entropy(policy_logits, move_targets) + 0.5 * F.mse_loss(values.view(-1), value_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Degrade patlamasını önlemek için daha sıkı zırh
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Ultra Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Anlık Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"🏁 Ultra Epoch {epoch+1} Tamamlandı! | Ortalama Loss: {avg_loss:.4f} | Süre: {time.time()-start_time:.1f} sn")
        
        save_path = os.path.join(CHECKPOINT_DIR, f"cvt_ultra_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    run_ultra_tuning()