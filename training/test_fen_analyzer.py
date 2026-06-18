import torch
import torch.nn.functional as F
import chess
from model import ChessTransformer
from dataset import ChessDataset

def analyze_fen(fen, checkpoint_path="training/checkpoints_finetune/cvt_finetune_epoch_5.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    board = chess.Board(fen)
    print(f"\n🔍 İNCELENEN POZİSYON:\n{board}\n")

    dummy_dataset = ChessDataset.__new__(ChessDataset)
    tensor = dummy_dataset.board_to_tensor(board).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(tensor)
        # Logitleri yüzdelik ihtimale çevir
        probabilities = F.softmax(policy_logits[0], dim=0)

    legal_moves = list(board.legal_moves)
    move_probs = []

    for move in legal_moves:
        move_id = move.from_square * 64 + move.to_square
        prob = probabilities[move_id].item() * 100
        move_probs.append((move.uci(), prob))

    # İhtimale göre sırala ve ilk 3'ü al
    move_probs.sort(key=lambda x: x[1], reverse=True)

    print("📊 MODELİN ADAY HAMLELERİ (POLICY):")
    for i, (move, prob) in enumerate(move_probs[:3]):
        print(f"  {i+1}. Seçenek: {move} (Olasılık: %{prob:.2f})")

    # Value Head Çıktısı (-1.0 Siyah üstün, 1.0 Beyaz üstün)
    val = value.item()
    advantage = "BEYAZ" if val > 0 else "SİYAH"
    print(f"\n⚖️ POZİSYON DEĞERLENDİRMESİ (VALUE):")
    print(f"  Skor: {val:.4f} ({advantage} Üstünlüğü Hissediliyor)\n")

if __name__ == "__main__":
    # Gürültü (Noise) eklenmiş, dikkat dağıtıcı taşlarla dolu En Passant testi
    test_fen = "r3N3/pp6/7k/6pP/7K/1N6/P7/R1B5 w - g6 0 1"
    analyze_fen(test_fen)