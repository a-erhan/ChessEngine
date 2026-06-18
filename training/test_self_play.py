import torch
import chess
import chess.pgn
from model import ChessTransformer
from dataset import ChessDataset
import time

def deathmatch_self_play(checkpoint_path="training/checkpoints/cvt_model_epoch_10.pt"):
    # 1. MOTORU ISIT
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Ölümüne Gölge Boksu (Sınırsız)"
    game.headers["White"] = "CvT_Sezgi_Beyaz"
    game.headers["Black"] = "CvT_Sezgi_Siyah"

    node = game
    dummy_dataset = ChessDataset.__new__(ChessDataset)

    print("🥊 ÖLÜMÜNE GÖLGE BOKSU BAŞLADI!")
    print("Kurallar: Biri mat edene, pat olana veya kurallar gereği berabere kalana kadar devam!\n")
    print("-" * 50)

    move_count = 1
    
    # 2. SINIRSIZ DÖVÜŞ DÖNGÜSÜ
    while not board.is_game_over():
        tensor = dummy_dataset.board_to_tensor(board).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(tensor)

        best_move = None
        best_logit = -float('inf')
        
        # Legal hamleler arasından en yüksek ihtimalli olanı seç
        for move in board.legal_moves:
            m_id = move.from_square * 64 + move.to_square
            if logits[0, m_id].item() > best_logit:
                best_logit = logits[0, m_id].item()
                best_move = move

        # Hamleyi tahtaya ve PGN ağacına işle
        if board.turn == chess.WHITE:
            print(f"{move_count}. {best_move.uci()} ", end="", flush=True)
        else:
            print(f"{best_move.uci()}")
            move_count += 1
            
        board.push(best_move)
        node = node.add_variation(best_move)

    # 3. MAÇ SONUCU VE HAKEM KARARI
    print("\n" + "-" * 50)
    outcome = board.outcome()
    
    print("🏁 MAÇ SONA ERDİ!")
    print(f"Skor: {board.result()}")
    
    # Bitiş sebebini detaylandır
    termination_reasons = {
        chess.Termination.CHECKMATE: "ŞAH MAT! Kusursuz bir infaz.",
        chess.Termination.STALEMATE: "PAT! Tahtada hamle kalmadı (Beraberlik).",
        chess.Termination.INSUFFICIENT_MATERIAL: "Yetersiz Taş! İki taraf da mat edemez (Beraberlik).",
        chess.Termination.SEVENTYFIVE_MOVES: "75 Hamle Kuralı (Beraberlik).",
        chess.Termination.FIVEFOLD_REPETITION: "Aynı pozisyon 5 kez tekrarlandı (Beraberlik).",
        chess.Termination.FIFTY_MOVES: "50 Hamle Kuralı! Piyon sürülmeden 50 hamle geçti (Beraberlik).",
        chess.Termination.THREEFOLD_REPETITION: "Aynı pozisyon 3 kez tekrarlandı (Beraberlik)."
    }
    
    reason = termination_reasons.get(outcome.termination, "Bilinmeyen Sebep")
    print(f"Bitiş Sebebi: {reason}\n")

    # 4. PGN ÇIKTISI (Lichess'te analiz edebilmen için)
    print("📝 TAM PGN ÇIKTISI (Bunu kopyalayıp Lichess Analysis'e yapıştırabilirsin):")
    print("=" * 60)
    print(game)
    print("=" * 60)

if __name__ == "__main__":
    deathmatch_self_play()