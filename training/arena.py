import torch
import chess
import chess.pgn
from model import ChessTransformer
from dataset import ChessDataset
import time

# --- AYARLAR VE KÖŞELER ---
MODEL_1_PATH = "training/checkpoints/cvt_model_epoch_10.pt"       # Mavi Köşe: Gözü Kara Savaşçı
MODEL_2_PATH = "training/checkpoints_finetune/cvt_finetune_epoch_5.pt" # Kırmızı Köşe: Temkinli Defansçı

def load_gladiator(checkpoint_path, device):
    """Modeli diskten arenaya çağırır."""
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def get_model_move(model, board, device, dummy_dataset):
    """Verilen tahta için MCTS olmadan saf sezgisel hamleyi çeker."""
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

def play_match(white_model, white_name, black_model, black_name, device, match_id):
    print(f"\n{'='*60}")
    print(f"🏆 MAÇ {match_id}: ⚪ {white_name} vs ⚫ {black_name}")
    print(f"{'='*60}")
    
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"Yapay Zeka Gladyatör Arenası - Maç {match_id}"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    node = game
    
    dummy_dataset = ChessDataset.__new__(ChessDataset)
    move_count = 1

    # Dövüş Döngüsü
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
        
        # M1/M4 işlemciler çok hızlıdır, gözle takip edebilmek için yarım saniye es
        time.sleep(0.1) 

    # Maç Sonu Analizi
    print("\n" + "-"*60)
    print(f"🏁 MAÇ {match_id} SONUCU: {board.result()}")
    
    termination_reasons = {
        chess.Termination.CHECKMATE: "ŞAH MAT!",
        chess.Termination.STALEMATE: "PAT! (Beraberlik)",
        chess.Termination.INSUFFICIENT_MATERIAL: "Yetersiz Taş! (Beraberlik)",
        chess.Termination.SEVENTYFIVE_MOVES: "75 Hamle Kuralı (Beraberlik)",
        chess.Termination.FIVEFOLD_REPETITION: "5 Konum Tekrarı (Beraberlik)",
        chess.Termination.FIFTY_MOVES: "50 Hamle Kuralı (Beraberlik)",
        chess.Termination.THREEFOLD_REPETITION: "3 Konum Tekrarı (Beraberlik)"
    }
    print(f"Bitiş Sebebi: {termination_reasons.get(board.outcome().termination, 'Bilinmiyor')}")
    print("-" * 60)
    print(game) # PGN Çıktısı

def start_tournament():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("🏟️ GLADYATÖRLER ISINIYOR... (Modeller RAM'e yükleniyor)")
    
    fighter_1 = load_gladiator(MODEL_1_PATH, device)
    fighter_2 = load_gladiator(MODEL_2_PATH, device)
    
    print("✅ Modeller arenaya indi. Dövüş başlıyor!\n")
    
    # MAÇ 1: Orijinal (Beyaz) vs Fine-Tune (Siyah)
    play_match(fighter_1, "CvT_Faz1_Orijinal", fighter_2, "CvT_Faz1.5_Defansif", device, 1)
    
    # MAÇ 2: Fine-Tune (Beyaz) vs Orijinal (Siyah)
    play_match(fighter_2, "CvT_Faz1.5_Defansif", fighter_1, "CvT_Faz1_Orijinal", device, 2)

if __name__ == "__main__":
    start_tournament()