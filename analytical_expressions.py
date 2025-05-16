
import torch

def psi(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    term1 = torch.exp(-2 * (r1 + r2))
    term2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    term3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return term1 * term2 * term3

def psi_a(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    return torch.exp(-2 * (r1 + r2))

def psi_a_first(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r1_ = x[0] + x[1] + x[2]
    r2_ = y[0] + y[1] + y[2]

    term1 = torch.exp(-2 * (r1 + r2))

    return torch.stack([-2 * term1 * (X[i] / r1) for i in range(3)] + [-2 * term1 * (X[i] / r2) for i in range(3, 6)])


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
    
    return torch.stack([2 * term1 * ( (2 * X[i] ** 2 * r1 - r1 ** 2 + X[i] ** 2)/(r1 ** 3)) for i in range(3)] + [2 * term1 * ( (2 * X[i] ** 2 * r2 - r2 ** 2 + X[i] ** 2)/(r2 ** 3)) for i in range(3, 6)])

def psi_b(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    return 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    
def psi_b_first(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r12_ = sum(x - y)

    term1 = torch.exp(-2 * (r1 + r2))
    term2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    term3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

    return torch.stack([(torch.exp(-alpha_1 * r12)/2) * ((x[i] - y[i])/r12) * (1 - alpha_1 * r12) for i in range(3)] + [(torch.exp(-alpha_1 * r12)/2) * ((x[i] - y[i])/r12) * (alpha_1 * r12 - 1) for i in range(3)])

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

    return torch.stack([(torch.exp(-alpha_1 * r12)/2) * (alpha_1 * (alpha_1 * r12 - 2) * ((x[i] - y[i])/ r12) ** 2 + (1 - alpha_1 * r12) * (1/r12 - ((x[i] - y[i]) ** 2/(r12 ** 3)))) for i in range(3)] + [(torch.exp(-alpha_1 * r12)/2) * (alpha_1 * (alpha_1 * r12 - 2) * ((x[i] - y[i])/ r12) ** 2 + (1 - alpha_1 * r12) * (1/r12 - ((x[i] - y[i]) ** 2/(r12 ** 3)))).detach() for i in range(3)])

def psi_c(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    return 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12
    

def psi_c_first(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r1_ = sum(x)
    r2_ = sum(y)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)
    r12_ = sum(x - y)

    i = 0

    return torch.stack([alpha_2 * (r1 + r2) * ((x[i] - y[i])/r12) + alpha_2 * r12 * (x[i]/r1) + 2 * (r1 - r2) * alpha_3 * (x[i]/r1) - alpha_4 * (x[i] - y[i])/r12 for i in range(3)] + [- alpha_2 * (r1 + r2) * ((x[i] - y[i])/r12) + alpha_2 * r12 * (y[i]/r2) - 2 * (r1 - r2) * alpha_3 * (y[i]/r2) + alpha_4 * (x[i] - y[i])/r12 for i in range(3)])

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

    return torch.stack([alpha_2 * (2 * (x[i] / r1) * (x[i] - y[i])/ r12 + (r1 + r2) * (1/r12 - ((x[i] - y[i]) ** 2 / (r12) ** 3)) + r12 * (1/r1 - x[i]**2 / r1**3)) + alpha_3 * (2 * (x[i] / r1) ** 2 + 2 * (r1 - r2) * ((1/r1) - (x[i] **2) / (r1 ** 3))) - alpha_4 * (1/r12 - ((x[i] - y[i]) ** 2/ (r12) ** 3)) for i in range(3)] + [alpha_2 * (-2 * (y[i] / r2) * (x[i] - y[i])/ r12 + (r1 + r2) * (1/r12 - ((x[i] - y[i]) ** 2 / (r12) ** 3)) + r12 * (1/r2 - y[i]**2 / r2**3)) + alpha_3 * (2 * (y[i] / r2) ** 2 - 2 * (r1 - r2) * ((1/r2) - (y[i] **2) / (r2 ** 3))) - alpha_4 * (1/r12 - ((x[i] - y[i]) ** 2/ (r12) ** 3)) for i in range(3)])


def psi_laplacian(X):
    """Full Laplacian - hessian verified """
    
    ap = psi_a(X)
    pa = psi_a_first(X)
    paa = psi_a_second(X)
    bp = psi_b(X)
    pb = psi_b_first(X)
    pbb = psi_b_second(X)
    cp = psi_c(X)
    pc = psi_c_first(X)
    pcc = psi_c_second(X)

    t = 2 * cp * torch.dot(pa, pb) + 2 * bp * torch.dot(pa, pc) + 2 * ap * torch.dot(pb, pc) + bp * cp * sum(paa) + ap * cp * sum(pbb) + bp * ap * sum(pcc)
    return t

def local_energy(X):
    ke = -0.5 * psi_laplacian(X) / psi(X)
    r1 = X[:3]
    r2 = X[3:6]

    potential = -2 / torch.norm(r1) - 2 / torch.norm(r2) + 1 / torch.norm(r1 - r2)

    return ke + potential