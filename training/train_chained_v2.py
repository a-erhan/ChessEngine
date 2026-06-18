import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessDataset
from model import ChessTransformer
from tqdm import tqdm

def train_stage(stage_name, pgn_file, max_games, epochs, learning_rate, model, device, save_path):
    """
    Handles a single training stage (dataset loading, training loop, and RAM cleanup).
    """
    print(f"\n{'='*50}")
    print(f"🚀 STAGE: {stage_name}")
    print(f"📁 Dataset: {pgn_file} | Target Games: {max_games} | LR: {learning_rate}")
    print(f"{'='*50}")
    
    if not os.path.exists(pgn_file):
        print(f"⚠️ WARNING: {pgn_file} not found! Skipping this stage...")
        return

    # Load high-quality, long games
    dataset = ChessDataset(pgn_file, max_games=max_games, min_elo=2000, min_moves=25)
    
    if len(dataset) == 0:
        print("⚠️ WARNING: No valid positions extracted. Skipping...")
        return
        
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(f"✅ Loaded {len(dataset)} positions into RAM for {stage_name}.\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"{stage_name} - Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            boards = batch["board"].to(device)
            targets = batch["move_target"].to(device)
            
            # Forward pass
            outputs = model(boards)
            loss = criterion(outputs, targets)
            
            # Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"🏁 {stage_name} - Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint securely after each epoch
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved to: {save_path}")
        
    # --- CRITICAL MEMORY CLEANUP ---
    print(f"🧹 Stage '{stage_name}' finished. Releasing RAM...")
    del dataloader
    del dataset
    gc.collect()  # Force Python garbage collector
    if device.type == 'mps':
        torch.mps.empty_cache()  # Clear Apple Silicon GPU cache
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    print("✅ RAM cleared. Ready for the next stage!\n")

def run_chained_pipeline():
    """
    Orchestrates the sequential training pipeline across different datasets.
    """
    MODEL_SAVE_PATH = "training/chess_transformer_v2_core.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🤖 CHAINED TRAINING PROTOCOL INITIATED (Device: {device})")
    
    # Initialize V2 Architecture
    model = ChessTransformer().to(device)
    
    # Resume from existing V2 checkpoint if available
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Found existing V2 weights! Loading from: {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("🌱 Starting fresh V2 model training from scratch.")

    # ---------------------------------------------------------
    # 📝 TRAINING PROTOCOL (STAGES)
    # ---------------------------------------------------------
    
    # STAGE 1: Heavy Lifting (2016 Games - Large Volume, Higher LR)
    train_stage(
        stage_name="PHASE 1 - HEAVY LIFTING (2016)",
        pgn_file="training/games_2016.pgn",
        max_games=50000,      
        epochs=8,             
        learning_rate=1e-3,   
        model=model,
        device=device,
        save_path=MODEL_SAVE_PATH
    )
    
    # STAGE 2: Refinement (2014 Games - Medium Volume, Medium LR)
    train_stage(
        stage_name="PHASE 2 - REFINEMENT (2014)",
        pgn_file="training/games.pgn",
        max_games=30000,      
        epochs=5,
        learning_rate=5e-4,   
        model=model,
        device=device,
        save_path=MODEL_SAVE_PATH
    )
    
    # STAGE 3: Fine-Tuning (2013 Games - Small Volume, Low LR)
    train_stage(
        stage_name="PHASE 3 - FINE-TUNING (2013)",
        pgn_file="training/games_2013.pgn",
        max_games=15000,      
        epochs=3,
        learning_rate=1e-4,   
        model=model,
        device=device,
        save_path=MODEL_SAVE_PATH
    )
    
    print("\n👑 ALL STAGES COMPLETED SUCCESSFULLY! THE ENGINE IS READY.")

if __name__ == "__main__":
    run_chained_pipeline()