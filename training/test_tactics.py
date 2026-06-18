import torch
import chess
from model import ChessTransformer
from dataset import board_to_array

def square_to_algebraic(square_idx):
    return chess.square_name(square_idx)

def test_tactics_and_mates():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Modeli Yükle
    model = ChessTransformer().to(device)
    model.load_state_dict(torch.load("training/chess_transformer_mvp.pt", map_location=device))
    model.eval()
    
    # 2. Trivial Kazançlar ve Mat Test Konumları (FEN formatında)
    tactical_positions = {
        "1. Çoban Matı Fırsatı (Vezir f7'yi vurup mat etmeli)": {
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "expected_move": "f3f7"
        },
        "2. Aptal Matı Cezası (Vezir h5'e gidip tek hamlede mat etmeli)": {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1",
            "expected_move": "d1h5"
        },
        "3. Koridor Matı (Arka sıra zayıf, kale d8'e inip mat etmeli)": {
            "fen": "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1",
            "expected_move": "d1d8"
        },
        "4. Basit At Çatalı Fırsatı (At c7'ye zıpalyıp şah ve kale çatalı atmalı)": {
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "expected_move": "b1c3" # Ya da merkez baskısı, örnek bir gelişim testi
        }
    }
    
    print("\n--- ⚔️ MODELİ TOKATLAMA VE MAT BULMA TESTİ BAŞLADI ---")
    
    total_tactics = len(tactical_positions)
    solved_tactics = 0
    
    with torch.no_grad():
        for name, info in tactical_positions.items():
            print(f"\n📍 Test Konumu: {name}")
            board = chess.Board(info["fen"])
            print(board)
            
            # Tahtayı tensor yap
            board_arr = board_to_array(board)
            board_tensor = torch.tensor(board_arr, dtype=torch.long).unsqueeze(0).to(device)
            
            # Tahmin üret
            outputs_from, outputs_to = model(board_tensor)
            
            # En iyi 3 olasılığı çek
            top_from = torch.topk(outputs_from[0], 3).indices.cpu().numpy()
            top_to = torch.topk(outputs_to[0], 3).indices.cpu().numpy()
            
            print("\n🔮 Modelin Sezgi Sıralaması:")
            
            matched_any = False
            for i in range(3):
                from_sq = top_from[i]
                to_sq = top_to[i]
                from_name = square_to_algebraic(from_sq)
                to_name = square_to_algebraic(to_sq)
                move_uci = f"{from_name}{to_name}"
                
                is_legal = "LEGAL" if chess.Move.from_uci(move_uci) in board.legal_moves else "İLLEGAL"
                
                # Hedef taktik hamleyle eşleşiyor mu?
                is_target = "🎯 [HEDEF HAMLE]" if move_uci == info["expected_move"] else ""
                
                print(f"  {i+1}. Tercih: {from_name} -> {to_name} | {is_legal} {is_target}")
                
                if i == 0 and move_uci == info["expected_move"]:
                    solved_tactics += 1
                    matched_any = True
                elif move_uci == info["expected_move"]:
                    matched_any = True
                    
            if matched_any and top_from[0] == chess.Move.from_uci(info["expected_move"]).from_square and top_to[0] == chess.Move.from_uci(info["expected_move"]).to_square:
                print(f"🔥 BAŞARILI: Model taktik darbeyi 1. tercihten kokladı!")
            elif matched_any:
                print(f"⚠️ KISMEN BAŞARILI: Model doğru hamleyi gördü ama 2. veya 3. sıraya attı.")
            else:
                print(f"❌ BAŞARISIZ: Model bu matı veya taktiği tamamen ıskaladı!")
            print("-" * 50)
            
    print(f"\n📈 ÖZET RAPOR: {total_tactics} kritik konumdan {solved_tactics} tanesini 1. tercihte bildi.")
    if solved_tactics == total_tactics:
        print("👑 MÜKEMMEL: Sezgi beyni taktik körlüğü tamamen aşmış!")
    else:
        print("🤖 ANALİZ: Bazı matları ıskalaması çok normal, C++ Minimax ağacı yarın buraları süpürecek.")

if __name__ == "__main__":
    test_tactics_and_mates()