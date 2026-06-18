import torch
import chess
import berserk
from model import ChessTransformer

# --- 1. MODEL VE SEZGİ AYARLARI ---
class ChessBotAgent:
    def __init__(self, checkpoint_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Faz 1'de eğittiğimiz 6 katmanlı CvT şasisi
        self.model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def select_best_move(self, board):
        from dataset import ChessDataset
        dummy_dataset = ChessDataset.__new__(ChessDataset)
        tensor = dummy_dataset.board_to_tensor(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(tensor)
            
        legal_moves = list(board.legal_moves)
        best_move = None
        best_logit = -float('inf')
        
        for move in legal_moves:
            move_id = move.from_square * 64 + move.to_square
            logit = policy_logits[0, move_id].item()
            
            if logit > best_logit:
                best_logit = logit
                best_move = move
                
        return best_move

# --- 2. LICHESS API VE OYUN AKIŞ MOTORU ---
def start_lichess_bot():
    # 🚨 BURAYA LICHESS'TEN ALDIĞIN TOKEN'I YAPIŞTIR
    LICHESS_TOKEN = "lip_YOUR_API_TOKEN_HERE"
    
    # Faz 1'de pişen 10. epoch modelini bağlıyoruz
    agent = ChessBotAgent("training/checkpoints/cvt_model_epoch_10.pt")
    
    session = berserk.TokenSession(LICHESS_TOKEN)
    client = berserk.Client(session)
    
    print("🤖 Lichess Botu Aktif! Meydan okumalar bekleniyor...")
    
    # Gelen etkinlikleri (Meydan okumaları ve maç davetlerini) canlı dinle
    for event in client.bots.stream_incoming_events():
        
        # A: Yeni Bir Meydan Okuma Geldiyse (Otomatik Kabul Et)
        if event['type'] == 'challenge':
            challenge_id = event['challenge']['id']
            # Klasik zaman kontrolü dışındaki absürt modları elemek istersen filtre koyabilirsin
            client.bots.accept_challenge(challenge_id)
            print(f"⚔️ Gelen meydan okuma kabul edildi! Maç ID: {challenge_id}")
            
        # B: Kabul Edilen Maç Resmen Başladıysa
        elif event['type'] == 'gameStart':
            game_id = event['game']['id']
            print(f"🚀 Maç başladı! Savaş Arenası: https://lichess.org/{game_id}")
            
            board = chess.Board()
            
            # Maçın içindeki hamle akışını (Stream) canlı dinle
            for game_event in client.bots.stream_game_state(game_id):
                if game_event['type'] == 'gameFull':
                    # İlk başta tahtanın durumunu ve rengimizi senkronize et
                    my_id = client.account.get_profile()['id']
                    white_id = game_event['white'].get('id')
                    bot_color = chess.WHITE if my_id == white_id else chess.BLACK
                    
                    # Eğer maç ortadan başladıysa hamle geçmişini yükle
                    moves = game_event['state']['moves'].split()
                    for move in moves:
                        board.push_uci(move)
                        
                elif game_event['type'] == 'gameState':
                    # Sadece hamleler güncellendiyse son hamleyi tahtaya işle
                    moves = game_event['moves'].split()
                    if len(moves) > 0:
                        # Sanal tahtayı sıfırla ve baştan diz (senkronizasyon hatasını engeller)
                        board = chess.Board()
                        for move in moves:
                            board.push_uci(move)
                
                # Sıra bota mı geldi kontrolü
                if board.turn == bot_color and not board.is_game_over():
                    print("🧠 Model tahtaya bakıyor ve hamle seçiyor...")
                    best_move = agent.select_best_move(board)
                    
                    if best_move:
                        client.bots.make_move(game_id, best_move.uci())
                        print(f"🚀 Hamle Gönderildi: {best_move.uci()}")

if __name__ == "__main__":
    start_lichess_bot()