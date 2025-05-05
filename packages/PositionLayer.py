import torch.nn as nn
import torch

class PositionEncoder(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = num_frequencies

        # 주파수 스케일 지정
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        # print(f"{self.freq_bands} here is PositionEncoder. after binding self.freq_bands. (line 11)")

    def forward(self, lon_lat):
        # 위도 경도
        inputs = torch.cat([lon_lat], dim=-1)
        # print(f"{inputs} {inputs.shape} here is PositionEncoder. after binding inputs.(line 16)")

        encodings = [inputs] # 원본 값 포함
        # print(f"{encodings} here is PositionEncoder. after binding encodings.(line 19)")

        for freq in self.freq_bands:
            for fn in [torch.sin, torch.cos]:
                encodings.append(fn(inputs * freq))

        return torch.cat(encodings, dim=-1)  # (B, 4)