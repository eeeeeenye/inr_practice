import numpy as np
import torch.nn as nn
import torch

class sineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True , is_first=True,omega_0=0):
        super().__init__()
        self.is_first = is_first
        self.omega_0 = omega_0
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.in_features,
                                        1/self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features)/self.omega_0,
                                            np.sqrt(6 / self.in_features)/self.omega_0)
            
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))                              # omega가 클수록 입력값의 진동 주기 짧아짐

    def forward_with_intermediate(self, x):                             # 중간값 확인용도
        intermediate = self.omega_0 * self.linear(x)
        return torch.sin(intermediate), intermediate

