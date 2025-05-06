
import torch

def psi_a_second(X):
    """Second order for first part of psi - hessian verified """
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r1_ = x[0] + x[1] + x[2]
    r2_ = y[0] + y[1] + y[2]

    term1 = torch.exp(-2 * (r1 + r2))
    
    return [2 * term1 * ( (2 * X[i] ** 2 * r1 - r1 ** 2 + X[i] ** 2)/(r1 ** 3)) for i in range(3)] + [2 * term1 * ( (2 * X[i] ** 2 * r2 - r2 ** 2 + X[i] ** 2)/(r2 ** 3)) for i in range(3, 6)]

def psi_b_second(X):
    """Second order for second part of psi - hessian verified """
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r12_ = sum(x - y)

    x1 = x[0]
    y1 = y[0]

    return [(torch.exp(-alpha_1 * r12)/2) * (alpha_1 * (alpha_1 * r12 - 2) * ((x[i] - y[i])/ r12) ** 2 + (1 - alpha_1 * r12) * (1/r12 - ((x[i] - y[i]) ** 2/(r12 ** 3)))) for i in range(3)] + [(torch.exp(-alpha_1 * r12)/2) * (alpha_1 * (alpha_1 * r12 - 2) * ((x[i] - y[i])/ r12) ** 2 + (1 - alpha_1 * r12) * (1/r12 - ((x[i] - y[i]) ** 2/(r12 ** 3)))).detach() for i in range(3)]


def psi_c_second(X):
    """Second order for first particle third part of psi - hessian verified """
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r12_ = sum(x - y)

    x1 = x[0]
    y1 = y[0]

    return [alpha_2 * (2 * (x[i] / r1) * (x[i] - y[i])/ r12 + (r1 + r2) * (1/r12 - ((x[i] - y[i]) ** 2 / (r12) ** 3)) + r12 * (1/r1 - x[i]**2 / r1**3)) + alpha_3 * (2 * (x[i] / r1) ** 2 + 2 * (r1 - r2) * ((1/r1) - (x[i] **2) / (r1 ** 3))) - alpha_4 * (1/r12 - ((x[i] - y[i]) ** 2/ (r12) ** 3)) for i in range(3)]