import torch.nn as nn
import torch

class PositionEncoder(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies

        # 주파수 스케일 지정
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, lon_lat, time):
        # 위도 경도
        inputs = torch.cat([lon_lat, time], dim=-1)

        encodings = [inputs] # 원본 값 포함

        for freq in self.freq_bands:
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn(inputs * freq))

        return torch.cat(encodings, dim=-1)  # (B, 4)