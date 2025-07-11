
import torch.nn as nn
import torch

#torch.set_default_dtype(torch.float64)

class MLP(nn.Module):

    def __init__(self, input_dim, n_hidden_layers, hidden_dim, output_size):
        super(MLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer (no activation here by default)
        layers.append(nn.Linear(hidden_dim, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

import math

def init_xavier_uniform(module):
    if isinstance(module, nn.Linear):
        in_dim = module.in_features
        out_dim = module.out_features
        limit = math.sqrt(1 / (in_dim + out_dim))
        nn.init.uniform_(module.weight, -limit, limit)
        nn.init.zeros_(module.bias)


input_dim = 3
n_hidden_layers = 3
hidden_dim = 32
output_size = 1

model = MLP(
    input_dim=input_dim,
    n_hidden_layers=n_hidden_layers,
    hidden_dim=hidden_dim,
    output_size=output_size
)

model.apply(init_xavier_uniform)


def psi_nn(x, model, return_log=False):
    if next(model.parameters()).is_cuda:
        x = x.to(next(model.parameters()).device)

    r1_vec = x[:, :3]
    r2_vec = x[:, 3:]

    r1 = torch.norm(r1_vec, dim=1, keepdim=True)
    r2 = torch.norm(r2_vec, dim=1, keepdim=True)
    r12 = torch.norm(r1_vec - r2_vec, dim=1, keepdim=True)

    features = torch.cat([r1, r2, r12], dim=1)

    nn_out = model(features)
    hydrogenic = -2 * (r1 + r2)
    jastrow = r12 / (2.0 * (1.0 + 0.5 * r12))

    log_psi = hydrogenic + jastrow + nn_out

    if return_log:
        return log_psi.squeeze()
    
    return torch.exp(log_psi).squeeze()
