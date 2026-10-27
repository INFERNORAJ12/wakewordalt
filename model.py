import torch.nn as nn

class WakeWordClassifier(nn.Module):
    """
    Lightweight classification head mapping 16-frame audio embeddings
    (1536 dimensions) to a binary detection probability [0.0 - 1.0].
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 32):
        super(WakeWordClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)