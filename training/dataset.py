import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

# Mapping chess pieces to unique integer IDs (Empty square = 0)
# White pieces: 1-6, Black pieces: 7-12
PIECE_TO_ID = {
    None: 0,
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
    7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
}

def get_piece_id(piece):
    if piece is None:
        return 0
    idx = piece.piece_type
    if piece.color == chess.BLACK:
        idx += 6
    return idx

def board_to_array(board):
    """
    Converts a 8x8 chess board into a flat 64-element 1D numpy array.
    """
    arr = np.zeros(64, dtype=np.int64)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        arr[square] = get_piece_id(piece)
    return arr

class ChessDataset(Dataset):
    def __init__(self, pgn_filepath, max_games=10000, min_elo=2000):
        self.data = []
        
        with open(pgn_filepath, "r", encoding="utf-8") as pgn:
            game_count = 0
            while game_count < max_games:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                # Filter by player ratings to ensure high-quality data
                white_elo = int(game.headers.get("WhiteElo", 0) or 0)
                black_elo = int(game.headers.get("BlackElo", 0) or 0)
                if white_elo < min_elo or black_elo < min_elo:
                    continue
                
                board = game.board()
                for move in game.mainline_moves():
                    # Get current board state before the move
                    board_state = board_to_array(board)
                    
                    # Targets: origin square and destination square (0 to 63)
                    move_from = move.from_square
                    move_to = move.to_square
                    
                    self.data.append({
                        "board": board_state,
                        "move_from": move_from,
                        "move_to": move_to
                    })
                    
                    board.push(move)
                
                game_count += 1
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "board": torch.tensor(item["board"], dtype=torch.long),
            "move_from": torch.tensor(item["move_from"], dtype=torch.long),
            "move_to": torch.tensor(item["move_to"], dtype=torch.long)
        }

if __name__ == "__main__":
    print("ChessDataset module initialized successfully in clean MVP mode.")