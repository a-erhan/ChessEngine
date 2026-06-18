import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=65):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class ChessLSTMTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4, # Hafiflettik çünkü LSTM de var
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 1. GÖZLER: CNN Özellik Çıkarıcı
        self.cnn = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=65)

        # 2. BEYİN: Transformer (Global Tahta Stratejisi)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. HAFIZA: Zaman Serisi İçin LSTM
        # Transformer'dan çıkan ardışık 5 'CLS' vektörünü birbirine bağlar
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # 4. KARAR: AlphaZero Dual-Head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (Batch, SeqLen=5, Channels=14, H=8, W=8)
        B, T, C, H, W = x.shape
        
        # 1. Zamanı ve Batch'i geçici olarak birleştir (Her bir tahtayı bağımsız analiz etmek için)
        x_flat = x.view(B * T, C, H, W) # shape: (B*5, 14, 8, 8)

        # 2. CNN Analizi
        features = self.cnn(x_flat) # shape: (B*5, d_model, 8, 8)
        features = features.view(B * T, self.d_model, -1).permute(0, 2, 1) # shape: (B*5, 64, d_model)

        # 3. [CLS] Ajanını Ekle ve Transformer'a Sok
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        sequence = torch.cat((cls_tokens, features), dim=1) # shape: (B*5, 65, d_model)
        
        sequence = self.pos_encoder(sequence)
        encoded = self.transformer(sequence)
        
        # Sadece Ajan tokenlerini (strateji özlerini) alıyoruz
        cls_out = encoded[:, 0, :] # shape: (B*5, d_model)

        # 4. Zaman Boyutunu Geri Getir ve LSTM'e Ver
        lstm_in = cls_out.view(B, T, self.d_model) # shape: (Batch, 5, d_model)
        lstm_out, (hn, cn) = self.lstm(lstm_in)
        
        # LSTM'in en son adımdaki (t=5) nihai çıkarımını alıyoruz
        final_state = lstm_out[:, -1, :] # shape: (Batch, d_model)

        # 5. Tetiği Çek
        policy_logits = self.policy_head(final_state)
        value = self.value_head(final_state)

        return policy_logits, value