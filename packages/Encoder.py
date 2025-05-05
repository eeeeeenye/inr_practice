import torch
from packages import PositionEncoder
from packages import SIREN
import torch.nn as nn

class WindModel(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.pos_encoder = PositionEncoder(num_frequencies) # lon, lat, time positional encoding

        # Positional Encoding의 출력 차원 : 2(lon_lat) + 2(sin/cos) * num_frequencies
        encoded_dim = 2 + 4 * num_frequencies
        self.encoder = SIREN(in_features=encoded_dim)                              

    def forward(self, lon_lat):
        # print(f"{lon_lat} here is Encoder before printing x")
        x = self.pos_encoder(lon_lat)  # 포지셔널 인코딩
        #x = torch.cat([pos_enc], dim=-1)     # input
        # print(f"{x.shape} here is Encoder after printing x")
        return self.encoder(x)