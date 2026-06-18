import torch
import chess
from model import ChessTransformer
from dataset import board_to_array

def square_to_algebraic(square_idx):
    return chess.square_name(square_idx)

def test_diverse_scenarios():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 1. Modeli Yükle
    model = ChessTransformer().to(device)
    model.load_state_dict(torch.load("training/chess_transformer_mvp.pt", map_location=device))
    model.eval()
    
    # 2. Oyun Sonu, Mat ve Stratejik Test Konumları
    scenarios = {
        "1. Vezir Oyun Sonu (Vezir g7'den korumalı mat etmeli - Sıra Beyazda)": {
            "fen": "k7/6Q1/8/8/8/8/8/6K1 w - - 0 1",
            "expected_move": "g7g8" # veya g7b7 matları
        },
        "2. Kale Oyun Sonu Matı (Kale a8'e inip şahı kilitlemeli - Sıra Beyazda)": {
            "fen": "k7/8/R7/8/8/8/8/6K1 w - - 0 1",
            "expected_move": "a6a8"
        },
        "3. Şah Kanadı Saldırısı (Fil h7 fedası veya Vezir h5 atağı - Sıra Beyazda)": {
            "fen": "r1bqk2r/pppp1ppp/2n1pn2/8/1b1PP3/2N2N2/PPPB1PPP/R2QKB1R w KQkq - 0 1",
            "expected_move": "e4e5" # Merkez piyonunu sürerek atı kaçırmaya zorlama
        },
        "4. Taktiksel Taş Kazancı (Vezir açmazdaki fili vurmalı - Sıra Beyazda)": {
            "fen": "rnbqk1nr/ppp2ppp/3b4/3p4/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
            "expected_move": "f1d3" # Gelişim ve kontrol hamlesi
        }
    }
    
    print("--- 🧠 GELİŞMİŞ MODEL SEZGİ TESTİ BAŞLADI ---")
    
    with torch.no_grad():
        for name, info in scenarios.items():
            print(f"\n📍 Senaryo: {name}")
            board = chess.Board(info["fen"])
            print(board)
            
            # Tensor dönüşümü
            board_arr = board_to_array(board)
            board_tensor = torch.tensor(board_arr, dtype=torch.long).unsqueeze(0).to(device)
            
            # İleri besleme
            outputs_from, outputs_to = model(board_tensor)
            
            top_from = torch.topk(outputs_from[0], 5).indices.cpu().numpy() # Bu sefer ilk 5'e bakalım
            top_to = torch.topk(outputs_to[0], 5).indices.cpu().numpy()
            
            print("\n🔮 Modelin En Yüksek Olasılıklı 5 Sezgisi:")
            
            for i in range(5):
                from_sq = top_from[i]
                to_sq = top_to[i]
                from_name = square_to_algebraic(from_sq)
                to_name = square_to_algebraic(to_sq)
                move_uci = f"{from_name}{to_name}"
                
                try:
                    is_legal = "LEGAL" if chess.Move.from_uci(move_uci) in board.legal_moves else "İLLEGAL"
                except:
                    is_legal = "GEÇERSİZ HAMLE GEOMETRİSİ"
                    
                print(f"  {i+1}. Tercih: {from_name} -> {to_name} | {is_legal}")
            print("-" * 50)

if __name__ == "__main__":
    test_diverse_scenarios()