import torch.nn as nn
from packages import sineLayer
from collections import OrderedDict

class SIREN(nn.Module):
    def __init__(self,in_features=6, first_omega_0 = 30, hidden_omega_0=30.):
        super().__init__()

        self.net = nn.Sequential(
            sineLayer(in_features, 512, is_first=True, omega_0=first_omega_0),
            
            sineLayer(512, 256, is_first=False, omega_0=hidden_omega_0),

            sineLayer(256, 256, is_first=False, omega_0=hidden_omega_0),
            sineLayer(256, 256, is_first=False, omega_0=hidden_omega_0),

            sineLayer(256, 2, is_first=False, omega_0=hidden_omega_0)
        )

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords
    
    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input']=x

        for i, layer in enumerate(self.net):
            if isinstance(layer, sineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations['_'.join(str(layer.__class__), "%d" % activation_count)] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations["_".join(str(layer.__class__), "%d" % activation_count)]
            activation_coubnt += 1
        return activations
