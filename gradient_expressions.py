
import torch

def psi_alpha_1(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    part_1 = torch.exp(-2 * (r1 + r2))
    part_2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    part_3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return (-0.5 * (r12 ** 2) * torch.exp(-alpha_1 * r12) * 1/part_2)

def psi_alpha_2(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    part_1 = torch.exp(-2 * (r1 + r2))
    part_2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    part_3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return (r1 + r2) * r12 * 1/part_3

def psi_alpha_3(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    part_1 = torch.exp(-2 * (r1 + r2))
    part_2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    part_3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return 1/part_3 * (r1 - r2)**2

def psi_alpha_4(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    part_1 = torch.exp(-2 * (r1 + r2))
    part_2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    part_3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return 1/part_3 * (-r12)

def get_psi_alpha(X):
    return torch.stack([
        psi_alpha_1(X), psi_alpha_2(X), psi_alpha_3(X),
        psi_alpha_4(X)
    ])
