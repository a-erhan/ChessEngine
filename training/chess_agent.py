import time
import random
import torch
import chess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from model import ChessTransformer

def human_delay(base_min=0.8, base_max=2.0):
    """Lichess'in radarlarına takılmamak için insansı düşünme süresi."""
    time.sleep(random.uniform(base_min, base_max))

class ChessAgent:
    def __init__(self, checkpoint_path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ChessTransformer(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.0).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.board = chess.Board()
        
    def get_best_move(self):
        from dataset import ChessDataset
        dummy_dataset = ChessDataset.__new__(ChessDataset)
        tensor = dummy_dataset.board_to_tensor(self.board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(tensor)
            
        legal_moves = list(self.board.legal_moves)
        best_move = None
        best_logit = -float('inf')
        
        for move in legal_moves:
            move_id = move.from_square * 64 + move.to_square
            logit = policy_logits[0, move_id].item()
            
            if logit > best_logit:
                best_logit = logit
                best_move = move
                
        return best_move

class LichessAnonymousBridge:
    def __init__(self, agent):
        self.agent = agent
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })

    def get_square_element(self, square_str):
        """Lichess tahtasındaki 'e2', 'g1' gibi kare elementlerini bulur."""
        try:
            # Lichess CG-Board yapısındaki kare koordinatını yakalar
            return self.driver.find_element(By.CSS_SELECTOR, f"cg-board square.{square_str}")
        except:
            # Alternatif Lichess DOM yapısı için fallback
            try:
                return self.driver.find_element(By.CSS_SELECTOR, f".cg-wrap square[class*='{square_str}']")
            except:
                return None

    def play_move_on_lichess(self, uci_move):
        """'e2e4' formatındaki hamleyi Lichess tahtasında fiziksel olarak oynar."""
        from_sq = uci_move[0:2]
        to_sq = uci_move[2:4]
        
        from_el = self.get_square_element(from_sq)
        to_el = self.get_square_element(to_sq)
        
        if from_el and to_el:
            action = ActionChains(self.driver)
            action.move_to_element(from_el).click().perform()
            time.sleep(random.uniform(0.1, 0.25))
            action.move_to_element(to_el).click().perform()
            return True
        return False

    def watch_and_play(self, my_color=chess.WHITE):
        # Doğrudan anonim (Anonymous) oyuncuların havuzuna ışınlanıyoruz
        self.driver.get("https://lichess.org/play")
        print("\n🌐 Lichess Arena Fırlatıldı!")
        print("📌 Lütfen açılan ekranda 'Play Anonymous' seçeneğiyle hızlı bir maça girin.")
        print(f"🎯 Sizin Renginiz: {'BEYAZ' if my_color == chess.WHITE else 'SİYAH'}\n")
        
        last_move_str = ""
        
        while True:
            time.sleep(0.1)
            
            if self.agent.board.is_game_over():
                print("🏁 Maç bitti! Teori başarıyla uygulandı.")
                break
                
            if self.agent.board.turn != my_color:
                # 🔍 RAKİBİN HAMLESİNİ LICHESS HIGHLIGHT SINIFLARINDAN ÖĞREN
                try:
                    # Lichess son yapılan hamlenin karelerine 'last-move' sınıfı ekler
                    last_move_elements = self.driver.find_elements(By.CSS_SELECTOR, "cg-board square.last-move")
                    
                    if len(last_move_elements) == 2:
                        squares = []
                        for el in last_move_elements:
                            # Class listesinden koordinatı cımbızla çek (Örn: "square", "last-move", "e4")
                            classes = el.get_attribute("class").split()
                            for c in classes:
                                if len(c) == 2 and c[0] in 'abcdefgh' and c[1] in '12345678':
                                    squares.append(c)
                        
                        if len(squares) == 2:
                            # Lichess'te hangisinin 'from' hangisinin 'to' olduğunu anlamak için legal hamle kontrolü
                            cand_1 = squares[0] + squares[1]
                            cand_2 = squares[1] + squares[0]
                            
                            detected_move = None
                            for m in self.agent.board.legal_moves:
                                if m.uci() == cand_1 or m.uci() == cand_1 + "q": detected_move = m
                                elif m.uci() == cand_2 or m.uci() == cand_2 + "q": detected_move = m
                            
                            if detected_move and detected_move.uci() != last_move_str:
                                print(f"👤 Anonim Rakip Oynadı: {detected_move.uci()}")
                                self.agent.board.push(detected_move)
                                last_move_str = detected_move.uci()
                except:
                    continue
            else:
                # 🧠 SIRA BİZDE: MODELİN RADARINI TETİKLE
                print("🧠 Model tahtaya bakıyor (Saf Sezgi Modu)...")
                best_move = self.agent.get_best_move()
                
                if best_move:
                    human_delay() # Filtrelere takılmamak için sinsi bekleme
                    uci_str = best_move.uci()
                    
                    success = self.play_move_on_lichess(uci_str[:4])
                    if success:
                        print(f"🚀 Model Hamle Yaptı: {uci_str}")
                        self.agent.board.push(best_move)
                        last_move_str = uci_str
                        
                        # Piyon terfisi olduysa Lichess otomatik vezir seçsin diye kısa bir es
                        if len(uci_str) == 5:
                            time.sleep(0.5)

if __name__ == "__main__":
    # Faz 1'in kralı olan 10. Epoch beynini vidalıyoruz
    agent = ChessAgent("training/checkpoints/cvt_model_epoch_10.pt")
    bridge = LichessAnonymousBridge(agent)
    
    # 🚨 MAÇ BAŞLARKEN RENGİNİ BURADAN GÜNCELLE: chess.WHITE veya chess.BLACK
    bridge.watch_and_play(my_color=chess.WHITE)