
import torch
from torch.func import jacrev
from torch import vmap
from analytical_expressions import local_energy
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
    term3 = 1 + alpha_2 * (r1 + r2) * r12 + alpha_3 * (r1 - r2)**2 - alpha_4 * r12

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
    r1 = (torch.rand(n_runs, 3, requires_grad=True) * 2 * L - L)
    r2 = (torch.rand(n_runs, 3, requires_grad=True) * 2 * L - L)
    max_steps = 1000
    sampled_Xs = []
    rejection_ratio = 0

    for i in range(N):
        chose = torch.rand(n_runs).reshape(n_runs, 1)
        dummy = torch.rand(n_runs)

        perturbed_r1 = r1 + 0.5 * (torch.rand(n_runs, 3) * 2 * L - L)
        perturbed_r2 = r2 + 0.5 * (torch.rand(n_runs, 3) * 2 * L - L)

        r1_trial = torch.where(chose < 0.5, perturbed_r1, r1)
        r2_trial = torch.where(chose >= 0.5, perturbed_r2, r2)
        psi_val = psi_vec(torch.cat((r1, r2, alphas), axis=1))
        psi_trial_val = psi_vec(torch.cat((r1_trial, r2_trial, alphas), axis=1))      
        psi_ratio = psi_trial_val / psi_val

        density_comp = psi_trial_val >= psi_val
        dummy_comp = dummy < psi_ratio

        condition = density_comp + dummy_comp

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


device = torch.device("cuda")
cpu = torch.device("cpu")
E_true = -2.9037243770

alpha_1 = torch.tensor(2.013, dtype=torch.float64, requires_grad=True) # 1.013
alpha_2 = torch.tensor(0.6419, dtype=torch.float64, requires_grad=True) # 0.2119
alpha_3 = torch.tensor(2.1406, dtype=torch.float64, requires_grad=True) # 0.1406
alpha_4 = torch.tensor(3.003, dtype=torch.float64, requires_grad=True) # 0.003

epochs = 50
alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
losses = []

lr = 0.001

config = {
    "lr" : lr,
    "alphas": alphas
}

# Step 2: Create Adam optimizer
optimizer = torch.optim.Adam(alphas, lr=lr)

# Training monitoring on wandb
wandb_login(running_on_hpc=False)
start_wandb_run(config=config, project_name="vmc")

for i in tqdm(range(epochs)):

    # Step 5: Zero gradients (as usual in PyTorch)
    alphas = [alpha_1, alpha_2, alpha_3, alpha_4]
    n_steps = 10
    alphas_metropolis = torch.tensor(alphas).unsqueeze(0).repeat(n_steps, 1)
    sampled_Xs = metropolis(10000, n_steps, alphas=alphas_metropolis)

    E = get_local_energies(sampled_Xs.to(device))
    mean_E = get_mean_energies(E.to(cpu))
    loss = torch.abs(E_true - mean_E)

    print(f"Mean energy is {mean_E}")
    print(f"Loss is {loss}")

    metrics = {
            "mean_energy": mean_E,
            "loss": loss,
        }

    wandb.log(metrics)

    losses.append(loss)
    gradients = get_dE_dX(sampled_Xs)

    # Now, assume you already have gradients computed externally:
    # Example: for step t, these are your gradients (replace with actual values)
    gradients_ = torch.mean(torch.mean(gradients, axis=0), axis=0)[6:]
    external_grads = gradients_.detach()

    # Step 3: Assign gradients manually
    for p, g in zip(alphas, external_grads):
        p.grad = g  # assign your externally computed gradient

    # Step 4: Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.empty_cache()
    del sampled_Xs


