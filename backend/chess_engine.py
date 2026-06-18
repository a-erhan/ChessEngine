"""
Chess AI Engine - Eğitilmiş modelleri kullanarak satranç hamlesi üretir.
"""
import os
import torch
import torch.nn as nn
import chess

# ─── MODEL MİMARİSİ (training/model.py ile aynı) ────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=65):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


class ChessTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        self.cnn = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU()
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=65)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x)
        features = features.view(batch_size, self.d_model, -1).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = self.pos_encoder(torch.cat((cls_tokens, features), dim=1))
        encoded = self.encoder(sequence)
        cls_out = encoded[:, 0, :]
        return self.policy_head(cls_out), self.value_head(cls_out)


# ─── MODEL TANIMI ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_CONFIGS = {
    "agresif": {
        "path": os.path.join(PROJECT_ROOT, "training", "checkpoints", "cvt_model_epoch_10.pt"),
        "name": "Agresif",
        "emoji": "🔥",
        "description": "Saldırgan oynama stili, risk almaktan çekinmez"
    },
    "tedbirli": {
        "path": os.path.join(PROJECT_ROOT, "training", "checkpoints_finetune", "cvt_finetune_epoch_5.pt"),
        "name": "Tedbirli",
        "emoji": "🛡️",
        "description": "Savunma odaklı, pozisyonu sağlam tutar"
    },
    "akil": {
        "path": os.path.join(PROJECT_ROOT, "training", "checkpoints_ultra", "cvt_ultra_epoch_3.pt"),
        "name": "Akıl",
        "emoji": "🧠",
        "description": "Dengeli ve hesaplı, uzun vadeli düşünür"
    }
}


# ─── BOARD ENCODING ───────────────────────────────────────────────────────────

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Chess board'u 14-kanallı tensor'e dönüştürür."""
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

    if board.has_kingside_castling_rights(chess.WHITE):   tensor[13, 0, 6] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):  tensor[13, 0, 2] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):   tensor[13, 7, 6] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):  tensor[13, 7, 2] = 1.0

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[13, ep_rank, ep_file] = 1.0

    return tensor


# ─── AI ENGINE ───────────────────────────────────────────────────────────────

class ChessAI:
    """Lazy-loading chess AI — sadece gerektiğinde model yükler."""

    def __init__(self):
        self.device = torch.device("cpu")  # Production'da CPU
        self._loaded: dict[str, ChessTransformer] = {}

    def _load_model(self, model_key: str) -> ChessTransformer:
        """Model henüz yüklenmemişse belleğe alır."""
        if model_key not in self._loaded:
            config = MODEL_CONFIGS[model_key]
            path = config["path"]

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model dosyası bulunamadı: {path}\n"
                    f"Lütfen model dosyalarının doğru konumda olduğunu kontrol edin."
                )

            print(f"[AI] {config['emoji']} {config['name']} modeli yükleniyor: {path}")
            model = ChessTransformer(
                d_model=256, nhead=8, num_layers=6,
                dim_feedforward=1024, dropout=0.0
            ).to(self.device)

            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._loaded[model_key] = model
            print(f"[AI] ✅ {config['name']} hazır!")

        return self._loaded[model_key]

    def get_best_move(self, fen: str, model_key: str) -> dict:
        """
        Verilen FEN pozisyonundan en iyi hamleyi hesaplar.

        Returns:
            dict: {
                "uci": "e2e4",
                "from": "e2",
                "to": "e4",
                "promotion": null | "q",
                "value": 0.35
            }
        """
        board = chess.Board(fen)

        if board.is_game_over():
            return {"error": "Oyun bitti", "game_over": True, "result": board.result()}

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {"error": "Yasal hamle yok"}

        model = self._load_model(model_key)
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = model(tensor)

        # En yüksek logit'e sahip yasal hamleyi seç
        best_move = None
        best_logit = -float("inf")

        for move in legal_moves:
            move_id = move.from_square * 64 + move.to_square
            logit = policy_logits[0, move_id].item()
            if logit > best_logit:
                best_logit = logit
                best_move = move

        if best_move is None:
            best_move = legal_moves[0]  # Fallback

        uci = best_move.uci()
        promotion = uci[4] if len(uci) == 5 else None

        return {
            "uci": uci,
            "from": uci[:2],
            "to": uci[2:4],
            "promotion": promotion,
            "value": float(value[0, 0].item()),
            "model": MODEL_CONFIGS[model_key]["name"]
        }

    def get_game_status(self, fen: str) -> dict:
        """Oyun durumunu döner."""
        board = chess.Board(fen)
        return {
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
            "is_check": board.is_check(),
            "legal_moves": [m.uci() for m in board.legal_moves],
            "move_count": board.fullmove_number,
        }


# Singleton instance
chess_ai = ChessAI()
