import torch
from packages import PositionEncoder
from packages import SIREN
import torch.nn as nn

class WindModel(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.pos_encoder = PositionEncoder(num_frequencies) # lon, lat, time positional encoding
        self.encoder = SIREN(in_features=63)                              

    def forward(self, lon_lat, time):
        x = self.pos_encoder(lon_lat, time)  # 포지셔널 인코딩
        #x = torch.cat([pos_enc], dim=-1)     # input
        return self.encoder(x)