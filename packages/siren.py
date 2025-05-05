import torch.nn as nn
from collections import OrderedDict
from packages import sineLayer

class SIREN(nn.Module):
    def __init__(self, in_features=6, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = nn.Sequential(
            sineLayer(in_features, 512, is_first=True, omega_0=first_omega_0),
            sineLayer(512, 256, is_first=False, omega_0=hidden_omega_0),
            sineLayer(256, 256, is_first=False, omega_0=hidden_omega_0),
            sineLayer(256, 256, is_first=False, omega_0=hidden_omega_0),
            nn.Linear(256, 1)  # 여기 인자명 없이 바로
            # nn.Linear(in_features, 512),    # 첫 번째 레이어
            # nn.ReLU(),                      # 활성화 함수
            # nn.Linear(512, 256),            # 두 번째 레이어
            # nn.ReLU(),                      # 활성화 함수
            # nn.Linear(256, 256),            # 세 번째 레이어
            # nn.ReLU(),                      # 활성화 함수
            # nn.Linear(256, 256),            # 네 번째 레이어
            # nn.ReLU(),                      # 활성화 함수
            # nn.Linear(256, 1)               # 출력 레이어
        )

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x

        for i, layer in enumerate(self.net):
            if isinstance(layer, sineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations[f"{layer.__class__.__name__}_{activation_count}"] = intermed
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
                activations[f"{layer.__class__.__name__}_{activation_count}"] = x

            activation_count += 1

        return activations
