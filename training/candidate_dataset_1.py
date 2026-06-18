import torch
from torch.utils.data import Dataset
import chess
import chess.pgn

class ChessDataset(Dataset):
    def __init__(self, pgn_file, max_games=None, min_elo=2000, min_moves=25):
        self.positions = [] 
        self.targets = []   
        
        print(f"Reading the PGN: {pgn_file} (2D Multi-Channel Mode)")
        
        with open(pgn_file, 'r', encoding='utf-8', errors='replace') as f:
            games_parsed = 0
            elite_games = 0
            
            while True:
                if max_games and elite_games >= max_games:
                    break
                    
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                    
                games_parsed += 1
                if games_parsed % 20000 == 0:
                    print(f"  [Heartbeat] Total games scanned: {games_parsed} | Elite games found: {elite_games}")
                
                # Elo Filtresi
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue
                    
                if white_elo < min_elo or black_elo < min_elo:
                    continue
                    
                # Maçın uzunluk filtresi (ply = hamle * 2)
                if game.end().ply() < min_moves * 2:
                    continue
                
                elite_games += 1
                board = game.board()
                
                for move in game.mainline_moves():
                    # O anki tahta durumunu FEN olarak kaydet
                    self.positions.append(board.fen())
                    
                    # 4096'lık hedef hamleyi indeksle
                    move_id = move.from_square * 64 + move.to_square
                    self.targets.append(move_id)
                    
                    # Tahtayı bir sonraki hamleye ilerlet
                    board.push(move)
                    
        print(f"[Completed] {elite_games} adet derin oyun yüklendi.")
        print(f"[Completed] Toplam {len(self.positions)} pozisyon 2D Tensör üretimi için belleğe alındı.")

    def __len__(self):
        return len(self.positions)

    def board_to_tensor(self, board):
        # (Kanallar, Yükseklik, Genişlik) -> (14, 8, 8)
        # PyTorch Conv2d katmanlarının doğrudan yutacağı ve geometrik olarak en anlamlı format.
        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
        
        # Taşları kanallara haritalandır (0-5: Beyaz, 6-11: Siyah)
        piece_map = board.piece_map()
        
        for sq, piece in piece_map.items():
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            
            # chess kütüphanesinde taşlar 1'den başlar (P=1, N=2, B=3, R=4, Q=5, K=6)
            # İndekse çevirmek için 1 çıkarıyoruz (0-5)
            channel = piece.piece_type - 1 
            
            # Eğer taş siyahsa, kanal indeksine 6 ekleyip üst katmanlara (6-11) atıyoruz
            if piece.color == chess.BLACK:
                channel += 6
                
            tensor[channel, rank, file] = 1.0
            
        # Kanal 12: Sıra Kimde? (Beyaz = 1.0, Siyah = 0.0)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        # Kanal 13: Rok Hakları ve En Passant Kareleri
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, 0, 6] = 1.0 # g1 karesi
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[13, 0, 2] = 1.0 # c1 karesi
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[13, 7, 6] = 1.0 # g8 karesi
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[13, 7, 2] = 1.0 # c8 karesi
            
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            tensor[13, ep_rank, ep_file] = 1.0
            
        return tensor

    def __getitem__(self, idx):
        fen = self.positions[idx]
        target = self.targets[idx]
        
        board = chess.Board(fen)
        board_tensor = self.board_to_tensor(board)
        
        return {
            "board": board_tensor,
            "move_target": torch.tensor(target, dtype=torch.long)
        }