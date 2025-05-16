
import torch
from torch.func import jacrev
from torch import vmap
from analytical_expressions import local_energy
from gradient_expressions import get_psi_alpha
from tqdm import tqdm
from training_monitoring import wandb_login, start_wandb_run
import os
import wandb


def psi(X):
    x = X[:3]
    y = X[3:6]
    alpha_1, alpha_2, alpha_3, alpha_4 = X[6:]
    r1 = torch.norm(x)
    r2 = torch.norm(y)
    r12 = torch.norm(x - y)

    term1 = torch.exp(-2 * (r1 + r2))
    term2 = 1 + 0.5 * r12 * torch.exp(-alpha_1 * r12)
    term3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2) ** 2 - alpha_4 * r12 ** 2

    return term1 * term2 * term3

psi_vec = vmap(psi)

def metropolis(N: int, n_runs: int, alphas: torch.tensor):  
    """
    Vectorized metropolis loop
    Over N steps, for n_runs. 
    Alphas passes in must be of same dim as n_runs
    """  
    assert alphas.shape[0] == n_runs        
    L = 1
    r1 = (torch.rand(n_runs, 3) * 2 * L - L)
    r2 = (torch.rand(n_runs, 3) * 2 * L - L)
    max_steps = 500
    sampled_Xs = []
    rejection_ratio = 0

    for i in tqdm(range(N)):
        chose = torch.rand(n_runs).reshape(n_runs, 1)
        dummy = torch.rand(n_runs)

        perturbed_r1 = r1 + 0.5 * (torch.rand(n_runs, 3) * 2 * L - L)
        perturbed_r2 = r2 + 0.5 * (torch.rand(n_runs, 3) * 2 * L - L)

        r1_trial = torch.where(chose < 0.5, perturbed_r1, r1)
        r2_trial = torch.where(chose >= 0.5, perturbed_r2, r2)
        psi_val = psi_vec(torch.cat((r1, r2, alphas), axis=1))
        psi_trial_val = psi_vec(torch.cat((r1_trial, r2_trial, alphas), axis=1))      
        
        psi_ratio = (psi_trial_val / psi_val) ** 2

        #density_comp = psi_trial_val >= psi_val
        dummy_comp = dummy < psi_ratio

        condition = dummy_comp

        rejection_ratio += torch.where(condition, 1./N, 0.0)

        condition = condition.reshape(condition.shape[0], 1)

        # Careful with overwriting
        r1 = torch.where(condition, r1_trial, r1)
        r2 = torch.where(condition, r2_trial, r2)
                
        if i > max_steps:
            sampled_Xs.append(torch.cat((r1, r2, alphas), axis=1))

    return torch.stack(sampled_Xs)

local_e_vec = vmap(local_energy)
local_e_vec_vec = vmap(local_e_vec)

def get_local_energies(X):
    reshaped_X = X.reshape(
        X.shape[1], X.shape[0], X.shape[2])
    return local_e_vec_vec(reshaped_X)

def get_mean_energies(E):
    return torch.mean(torch.mean(E, dim=1))

def dE_dalpha(input):
    return jacrev(local_energy)(input)

dE_dalpha_vec = vmap(dE_dalpha)
dE_dalpha_vec_vec = vmap(dE_dalpha_vec)

def get_dE_dX(X):
    reshaped_X = X.reshape(
        X.shape[1], X.shape[0], X.shape[2])
    return dE_dalpha_vec_vec(X)

def get_gradients_from_expression(X_, E_):
    psi_alpha = vmap(get_psi_alpha)(X_)

    part_1 = psi_alpha - torch.mean(psi_alpha, axis=0)
    part_2 = E_ - torch.mean(E_)
    return torch.mean(part_1.T * part_2, axis=1)

dE_dalpha = vmap(get_gradients_from_expression)

device = torch.device("cuda")
cpu = torch.device("cpu")
E_true = -2.9037243770

alpha_1 = torch.tensor(0.313, dtype=torch.float64, requires_grad=True) # 1.013
alpha_2 = torch.tensor(0.6419, dtype=torch.float64, requires_grad=True) # 0.2119
alpha_3 = torch.tensor(0.3206, dtype=torch.float64, requires_grad=True) # 0.1406
alpha_4 = torch.tensor(0.821, dtype=torch.float64, requires_grad=True) # 0.003

epochs = 10000
alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
losses = []

lr = 0.001
n_walkers = 200
mc_steps = 5000

config = {
    "lr" : lr,
    "alphas": alphas,
    "n_walkers": n_walkers,
    "mc_steps": mc_steps
}

optimizer = torch.optim.Adam(alphas, lr=lr)

# Training monitoring on wandb
wandb_login(running_on_hpc=True)
start_wandb_run(config=config, project_name="vmc")


for i in tqdm(range(epochs)):

    alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
    alphas_metropolis = torch.tensor(alphas).unsqueeze(0).repeat(n_walkers, 1)
    sampled_Xs = metropolis(mc_steps, n_walkers, alphas=alphas_metropolis)

    E = get_local_energies(sampled_Xs)
    mean_E = get_mean_energies(E)
    loss = torch.abs(E_true - mean_E)
    losses.append(loss)

    reshaped_X = sampled_Xs.reshape(
        sampled_Xs.shape[1], sampled_Xs.shape[0], sampled_Xs.shape[2])
    gradients = dE_dalpha(reshaped_X, E)
    gradients = torch.mean(gradients, axis=0)
    external_grads = gradients.detach()

    display_alphas = torch.stack(alphas).detach().tolist()
    display_gradients = external_grads.tolist()
    
    print(f"Mean energy is {mean_E}")
    print(f"Loss is {loss}")
    print(f"Alphas are {display_alphas}")
    print(f"Gradients are {display_gradients}")

    metrics = {
            "mean_energy": mean_E,
            "loss": loss,
            "alphas": display_alphas,
            "gradients": display_gradients
        }

    wandb.log(metrics)

    for p, g in zip(alphas, external_grads):
        p.grad = g

    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.empty_cache()
    del sampled_Xs
    del reshaped_X
    del E


