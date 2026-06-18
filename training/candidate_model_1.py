import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    # Satranç tahtası 8x8=64 karelik sabit bir alan olduğu için, klasik sinüs dalgaları yerine 
    # modelin kendi kendine öğrenebileceği (Learnable) bir konum kodlaması çok daha keskindir.
    def __init__(self, d_model, max_len=65):
        super().__init__()
        # 65 token: 1 adet [CLS] ajanı + 64 adet satranç karesi
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class ChessTransformer(nn.Module):
    # İsim importlar bozulmasın diye aynı kaldı, ama kalbinde melez bir canavar yatıyor!
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=6, # Çok ağır olmasın diye 6 katman yeterli
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # 1. GÖZLER: CNN Özellik Çıkarıcı (Feature Extractor)
        # Giriş boyutumuz artık 1 boyutlu embedding değil, 14 Kanallı (14x8x8) Tensör!
        self.cnn = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        # Ajan Tokeni (Modelin tahtayı izleyip stratejiyi kendi içine çekeceği [CLS] tokeni)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=65)

        # 2. BEYİN: Transformer Omurgası
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. KARAR: Çift Başlı AlphaZero Tetiği (Dual-Head)
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
        # x shape: (Batch, 14, 8, 8)
        batch_size = x.size(0)

        # 1. Tahtayı CNN'den geçir ve pikselleri analiz et
        features = self.cnn(x) # shape: (Batch, 256, 8, 8)
        
        # 2. Transformer'a girmesi için haritayı (8x8) düzleştir
        # (Batch, 256, 64) formatına gelir
        features = features.view(batch_size, self.d_model, -1) 
        # (Batch, 64, 256) formatına çeviriyoruz (Dizi mantığı)
        features = features.permute(0, 2, 1) 

        # 3. Ajanı (CLS Token) dizinin en başına ekle -> Toplam 65 token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls_tokens, features), dim=1) # shape: (Batch, 65, 256)

        # 4. Koordinatları ekle ve Transformer'a sok
        sequence = self.pos_encoder(sequence)
        encoded = self.encoder(sequence)

        # 5. Tahtayı analiz eden Ajanı cımbızla çek (En baştaki token, yani 0. index)
        cls_out = encoded[:, 0, :]

        # 6. Tetiği çek
        policy_logits = self.policy_head(cls_out)
        value = self.value_head(cls_out)

        return policy_logits, value