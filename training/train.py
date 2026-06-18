import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import ChessDataset
from model import ChessTransformer
from tqdm import tqdm

def train_model():
    # --- Yeni Dengeli Fine-Tuning Parametreleri ---
    BATCH_SIZE = 512       
    LEARNING_RATE = 2e-4   # Hassas ince ayar hızı
    EPOCHS = 6             # İstediğin gibi temiz bir 6 epoch
    MIN_ELO = 1800         
    
    # Klasördeki dosyalarının adıyla birebir eşleştiğinden emin ol:
    PGN_2014 = "training/games.pgn"
    PGN_2013 = "training/games_2013.pgn" 
    MODEL_SAVE_PATH = "training/chess_transformer_mvp.pt"
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # --- Dengeli Veri Yükleme (5k ve 7k) ---
    datasets = []
    
    # 2014 Dosyasından tam 5.000 oyun
    if os.path.exists(PGN_2014):
        print("Loading 2014 dataset (Target: 5000 games)...")
        datasets.append(ChessDataset(PGN_2014, max_games=5000, min_elo=MIN_ELO))
    else:
        print(f"ERROR: '{PGN_2014}' bulunamadı! Lütfen dosya adını kontrol et.")
        return
        
    # 2013 Dosyasından tam 7.000 oyun
    if os.path.exists(PGN_2013):
        print("Loading 2013 dataset (Target: 7000 games)...")
        datasets.append(ChessDataset(PGN_2013, max_games=7000, min_elo=MIN_ELO))
    else:
        print(f"ERROR: '{PGN_2013}' bulunamadı! Klasördeki dosyanın adının tam olarak 'games_2013.pgn' olduğundan emin ol.")
        return
        
    # İki farklı kaynaktan gelen veriyi harmanla
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dengeli veri seti başarıyla harmanlandı!")
    print(f"Total training samples (positions): {len(combined_dataset)}")
    
    # ... Kodun geri kalan model yükleme (load_state_dict) ve training loop kısmı tamamen aynı kalıyor.
    # --- Initialize Model ---
    model = ChessTransformer().to(device)
    
    # --- CRITICAL: Load Existing Checkpoint for Fine-Tuning ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🚀 Found existing checkpoint! Loading weights from: {MODEL_SAVE_PATH}")
        # map_location ensures it safely maps to the current device (MPS)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("No existing checkpoint found. Starting training from scratch.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # --- Training Loop ---
    print(f"Starting training loop for {EPOCHS} fine-tuning epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            boards = batch["board"].to(device)
            targets_from = batch["move_from"].to(device)
            targets_to = batch["move_to"].to(device)
            
            # Forward pass
            outputs_from, outputs_to = model(boards)
            
            # Compute dual losses
            loss_from = criterion(outputs_from, targets_from)
            loss_to = criterion(outputs_to, targets_to)
            loss = loss_from + loss_to
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save updated weights after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Checkpoint saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()