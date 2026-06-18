import torch
import chess
from model import ChessTransformer
from dataset import board_to_array

def square_to_algebraic(square_idx):
    return chess.square_name(square_idx)

def test_raw_engine():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # 1. Modeli Yükle
    model = ChessTransformer().to(device)
    model.load_state_dict(torch.load("training/chess_transformer_mvp.pt", map_location=device))
    model.eval()
    
    # 2. Test Pozisyonları Oluştur (Standart Açılış ve Kritik Bir Orta Oyun)
    test_positions = {
        "Standart Başlangıç Konumu": chess.Board(),
        "E4 Sonrası Siyahın Yanıtı (e4 oynandı)": chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        "Açık Bir Taktik Konum (Beyaz Hamlede)": chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    }
    
    print("\n--- 🧠 MODEL TEST PROTOKOLÜ BAŞLADI ---")
    
    with torch.no_grad():
        for name, board in test_positions.items():
            print(f"\n📍 Pozisyon: {name}")
            print(board)
            
            # Tahtayı modelin anlayacağı formata çevir
            board_arr = board_to_array(board)
            board_tensor = torch.tensor(board_arr, dtype=torch.long).unsqueeze(0).to(device)
            
            # Modelden hamle tahmini al
            outputs_from, outputs_to = model(board_tensor)
            
            # En yüksek olasılıklı ilk 3 'Nereden' ve 'Nereye' karelerini çek
            top_from = torch.topk(outputs_from[0], 3).indices.cpu().numpy()
            top_to = torch.topk(outputs_to[0], 3).indices.cpu().numpy()
            
            print("\n🔮 Modelin En Yüksek Olasılıklı Hamle Sezgileri:")
            
            found_legal = False
            # Olasılık kombinasyonlarını ekrana bas ve ilk legal olanı yakala
            for i in range(3):
                from_sq = top_from[i]
                to_sq = top_to[i]
                from_name = square_to_algebraic(from_sq)
                to_name = square_to_algebraic(to_sq)
                
                # UCI formatında hamle stringi oluştur (örn: e2e4)
                move_uci = f"{from_name}{to_name}"
                is_legal = "LEGAL (KURALA UYGUN)" if chess.Move.from_uci(move_uci) in board.legal_moves else "İLLEGAL"
                
                print(f"  {i+1}. Tercih: {from_name} -> {to_name} | Durum: {is_legal}")
                
                if is_legal == "LEGAL (KURALA UYGUN)" and not found_legal:
                    best_move = move_uci
                    found_legal = True
                    
            if found_legal:
                print(f"✅ Seçilen En Mantıklı Hamle: {best_move}")
            else:
                print("❌ UYARI: İlk 3 tahminde tamamen legal hamle üretilemedi! (C++ arama ağacı burada devreye girmeli)")
            print("-" * 40)

if __name__ == "__main__":
    test_raw_engine()