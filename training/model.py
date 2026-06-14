import torch
import torch.nn as nn

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=13, d_model=128, nhead=4, num_layers=4, dim_feedforward=512):
        super(ChessTransformer, self).__init__()
        
        # vocab_size=13 handles empty squares (0) + 12 distinct chess pieces
        self.piece_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding for 64 squares so the model knows board geometry
        self.position_embedding = nn.Embedding(64, d_model)
        
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dual Heads for move prediction (64 squares total)
        self.fc_from = nn.Linear(d_model * 64, 64)
        self.fc_to = nn.Linear(d_model * 64, 64)
        
    def forward(self, x):
        # x shape: (batch_size, 64)
        batch_size = x.size(0)
        
        # Generate position indices (0 to 63)
        positions = torch.arange(0, 64, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Combine piece information with structural position information
        out = self.piece_embedding(x) + self.position_embedding(positions)
        
        # Pass through the Transformer layers
        out = self.transformer_encoder(out) # shape: (batch_size, 64, d_model)
        
        # Flatten the representation for the dense output layers
        out = out.view(batch_size, -1)
        
        # Predict source and destination square probabilities
        logits_from = self.fc_from(out)
        logits_to = self.fc_to(out)
        
        return logits_from, logits_to

if __name__ == "__main__":
    # Quick sanity check
    model = ChessTransformer()
    dummy_input = torch.randint(0, 13, (2, 64)) # Batch of 2 fake chess boards
    out_from, out_to = model(dummy_input)
    print("Model check passed!")
    print("Output shapes (From/To):", out_from.shape, out_to.shape)