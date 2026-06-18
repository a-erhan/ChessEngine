import torch
import chess
from model import ChessTransformer
from dataset import ChessDataset

def run_puzzle_benchmark(checkpoint_path="training/checkpoints/cvt_model_epoch_10.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Klasik taktik bulmacalar (FEN ve Beklenen Hamle)
    puzzles = [
        {"name": "Tek Hamlede Mat (Vezir Fadası Sonrası)", "fen": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "expected": "e1e8"},
        {"name": "Açmazdan Taş Kazanma", "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5", "expected": "f3e5"}, # İtalyan açılışı merkezi feda/taktik
        {"name": "Çatal (Fork) Fırsatı", "fen": "8/8/8/2k5/4N3/8/8/4K3 w - - 0 1", "expected": "e4d6"} # At çatalı senaryosu (Temsili)
    ]

    dummy_dataset = ChessDataset.__new__(ChessDataset)
    correct = 0

    print("🧠 TAKTİK BULMACA TESTİ BAŞLIYOR...\n" + "="*40)
    for p in puzzles:
        board = chess.Board(p["fen"])
        tensor = dummy_dataset.board_to_tensor(board).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(tensor)

        best_move = None
        best_logit = -float('inf')
        for move in board.legal_moves:
            m_id = move.from_square * 64 + move.to_square
            if logits[0, m_id].item() > best_logit:
                best_logit = logits[0, m_id].item()
                best_move = move.uci()

        status = "✅ BAŞARILI" if best_move == p["expected"] else f"❌ HATALI (Beklenen: {p['expected']})"
        if best_move == p["expected"]: correct += 1
        
        print(f"Test: {p['name']}")
        print(f"  -> Modelin Hamlesi: {best_move} | Sonuç: {status}\n")

    print(f"📊 BAŞARI ORANI: {correct}/{len(puzzles)} (%{correct/len(puzzles)*100:.0f})")

if __name__ == "__main__":
    run_puzzle_benchmark()