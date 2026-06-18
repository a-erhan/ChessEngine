import torch
import chess
import chess.pgn
from model import ChessTransformer
from dataset import ChessDataset
import time

# --- GLADYATÖR KÖŞELERİ ---
MODEL_FAZ1 = "training/checkpoints/cvt_model_epoch_10.pt"
MODEL_FAZ1_5 = "training/checkpoints_finetune/cvt_finetune_epoch_5.pt"
MODEL_FAZ1_75 = "training/checkpoints_ultra/cvt_ultra_epoch_3.pt"

# London Sistemi'nin kilit noktası (8. hamle sonrası)
LONDON_FEN = "r1bq1rk1/pp3ppp/2nbpn2/2pp4/3P4/2PBPNB1/PP1N1PPP/R2QK2R w KQ - 0 9"

def load_gladiator(checkpoint_path, device):
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def get_model_move(model, board, device, dummy_dataset):
    tensor = dummy_dataset.board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(tensor)
    
    best_move = None
    best_logit = -float('inf')
    for move in board.legal_moves:
        m_id = move.from_square * 64 + move.to_square
        if logits[0, m_id].item() > best_logit:
            best_logit = logits[0, m_id].item()
            best_move = move
    return best_move

def play_match(white_model, white_name, black_model, black_name, device, match_id, start_fen):
    print(f"\n{'='*60}")
    print(f"🏆 MAÇ {match_id} (LONDON SİSTEMİ): ⚪ {white_name} vs ⚫ {black_name}")
    print(f"{'='*60}")
    
    # Tahtayı London FEN'inden başlatıyoruz
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = f"Yapay Zeka London Arenası - Maç {match_id}"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    # PGN'e başlangıç pozisyonunu ekle
    game.setup(board)
    
    node = game
    dummy_dataset = ChessDataset.__new__(ChessDataset)
    move_count = 9 # FEN 9. hamleden başlıyor

    while not board.is_game_over():
        current_model = white_model if board.turn == chess.WHITE else black_model
        move = get_model_move(current_model, board, device, dummy_dataset)
        
        if board.turn == chess.WHITE:
            print(f"{move_count}. {move.uci()} ", end="", flush=True)
        else:
            print(f"{move.uci()}")
            move_count += 1
            
        board.push(move)
        node = node.add_variation(move)
        time.sleep(0.1) 

    print("\n" + "-"*60)
    print(f"🏁 MAÇ {match_id} SONUCU: {board.result()}")
    print(game)

def start_finale():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("🏟️ LONDON SİSTEMİ ARENAYA KURULUYOR...")
    
    fighter_1 = load_gladiator(MODEL_FAZ1, device)
    fighter_1_5 = load_gladiator(MODEL_FAZ1_5, device)
    fighter_1_75 = load_gladiator(MODEL_FAZ1_75, device)
    
    print("✅ Modeller hazır. Stratejik savaş başlasın!\n")
    
    # MAÇ 1: Endgame Uzmanı (Beyaz) vs Orijinal (Siyah)
    play_match(fighter_1_75, "CvT_Faz1.75_Endgame", fighter_1, "CvT_Faz1_Orijinal", device, 1, LONDON_FEN)
    
    # MAÇ 2: Endgame Uzmanı (Beyaz) vs Defansif (Siyah)
    play_match(fighter_1_75, "CvT_Faz1.75_Endgame", fighter_1_5, "CvT_Faz1.5_Defansif", device, 2, LONDON_FEN)

if __name__ == "__main__":
    start_finale()