import torch
from torch.utils.data import Dataset
import chess
import chess.pgn

class ChessDataset(Dataset):
    def __init__(self, pgn_file, max_games=None, min_elo=2000, min_moves=25, seq_len=5):
        self.samples = [] 
        self.seq_len = seq_len
        
        print(f"📂 PGN okunuyor: {pgn_file} (Zaman Serisi: {seq_len} Hamlelik Pencereler)")
        
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
                
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    continue
                    
                if white_elo < min_elo or black_elo < min_elo:
                    continue
                    
                if game.end().ply() < min_moves * 2:
                    continue
                
                elite_games += 1
                board = game.board()
                fen_history = []
                
                for move in game.mainline_moves():
                    fen_history.append(board.fen())
                    move_id = move.from_square * 64 + move.to_square
                    
                    # Son 'seq_len' kadar tahtayı al, eğer yeterli hamle yoksa ilk tahtayı kopyalayarak doldur (Padding)
                    window = []
                    pad_len = self.seq_len - len(fen_history)
                    if pad_len > 0:
                        window.extend([fen_history[0]] * pad_len)
                        window.extend(fen_history)
                    else:
                        window.extend(fen_history[-self.seq_len:])
                    
                    self.samples.append((window, move_id))
                    board.push(move)
                    
        print(f"✅ {elite_games} oyun yüklendi. {len(self.samples)} adet {seq_len}'li zaman serisi oluşturuldu.")

    def __len__(self):
        return len(self.samples)

    def board_to_tensor(self, fen):
        board = chess.Board(fen)
        tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
        piece_map = board.piece_map()
        
        for sq, piece in piece_map.items():
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            channel = piece.piece_type - 1 
            if piece.color == chess.BLACK:
                channel += 6
            tensor[channel, rank, file] = 1.0
            
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        if board.has_kingside_castling_rights(chess.WHITE): tensor[13, 0, 6] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 0, 2] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 6] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 2] = 1.0
            
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            tensor[13, ep_rank, ep_file] = 1.0
            
        return tensor

    def __getitem__(self, idx):
        fen_window, target = self.samples[idx]
        
        # 5 adet FEN'i tensöre çevirip üst üste yığıyoruz (Seq, Channels, H, W) -> (5, 14, 8, 8)
        tensors = [self.board_to_tensor(fen) for fen in fen_window]
        sequence_tensor = torch.stack(tensors)
        
        return {
            "board_sequence": sequence_tensor,
            "move_target": torch.tensor(target, dtype=torch.long)
        }