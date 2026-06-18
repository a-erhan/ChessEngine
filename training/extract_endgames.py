import chess.pgn
import os

def create_endgame_dataset(input_pgn="training/games_3.pgn", output_pgn="training/pure_endgames.pgn", max_pieces=10, min_elo=2000):
    print(f"🔍 {input_pgn} taranıyor...")
    print(f"⚙️ FİLTRELER: Max Taş = {max_pieces} | Min ELO = {min_elo}+")
    
    in_file = open(input_pgn, "r")
    out_file = open(output_pgn, "w")
    
    games_saved = 0
    games_scanned = 0
    
    while True:
        game = chess.pgn.read_game(in_file)
        if game is None:
            break
            
        games_scanned += 1
        
        # --- 1. KALİTE KONTROLÜ (ELO FİLTRESİ) ---
        try:
            # PGN'de Elo bilgisi yoksa veya '?' şeklindeyse ValueError verir
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            continue # Bozuk veya kalibresi belirsiz maçları direkt atla
            
        if white_elo < min_elo or black_elo < min_elo:
            continue # 300'lük sokak dövüşçülerini arenaya almıyoruz!
            
        # --- 2. ENTROPİ KONTROLÜ (TAŞ SAYISI) ---
        board = game.board()
        moves = list(game.mainline_moves())
        endgame_start_idx = -1
        
        for i, move in enumerate(moves):
            board.push(move)
            if len(board.piece_map()) <= max_pieces:
                endgame_start_idx = i
                break
        
        if endgame_start_idx != -1 and (len(moves) - endgame_start_idx) >= 5:
            new_game = chess.pgn.Game()
            new_game.setup(board)
            
            # Yeni PGN başlıklarına kalite damgasını vuruyoruz
            new_game.headers["Event"] = f"Elite Endgame (Pieces <= {max_pieces}, Elo {min_elo}+)"
            new_game.headers["Result"] = game.headers.get("Result", "*")
            new_game.headers["WhiteElo"] = str(white_elo)
            new_game.headers["BlackElo"] = str(black_elo)
            
            new_node = new_game
            for move in moves[endgame_start_idx + 1:]:
                new_node = new_node.add_variation(move)
            
            print(new_game, file=out_file, end="\n\n")
            games_saved += 1
            
            if games_saved % 100 == 0:
                print(f"  DAMITILAN ELİT OYUN SONU: {games_saved} (Taranan: {games_scanned})")

    in_file.close()
    out_file.close()
    print("-" * 60)
    print(f"✅ İŞLEM TAMAM! Toplam {games_scanned} oyun tarandı.")
    print(f"🎯 Sadece {games_saved} adet kusursuz, elit oyun sonu {output_pgn} dosyasına yazıldı.")

if __name__ == "__main__":
    create_endgame_dataset()