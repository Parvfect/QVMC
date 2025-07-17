
import torch
from torch import vmap
from functorch import make_functional, vmap, grad
from tqdm import tqdm
from training_monitoring import wandb_login, start_wandb_run
import os
import wandb
from nn import psi_nn, model
import datetime


def metropolis(N: int, pos: torch.tensor, n_runs: int, model: torch.nn, keep_mc_steps: bool):  
    
    L = 0.5
    
    r1 = pos[:, :3]
    r2 = pos[:, 3:]
    accept_count = 0
    if keep_mc_steps:
        sampled_Xs = []

    for _ in range(N):
        chose = torch.rand(n_runs, 1)

        perturbed_r1 = r1 + L * torch.randn(n_runs, 3)
        perturbed_r2 = r2 + L * torch.randn(n_runs, 3)

        r1_trial = torch.where(chose < 0.5, perturbed_r1, r1)
        r2_trial = torch.where(chose >= 0.5, perturbed_r2, r2)

        x_old = torch.cat((r1, r2), dim=1)
        x_trial = torch.cat((r1_trial, r2_trial), dim=1)

        psi_val = psi_nn(x_old, model, return_log=True)
        psi_trial_val = psi_nn(x_trial, model, return_log=True)

        psi_ratio = torch.exp(2 * (psi_trial_val - psi_val)) 

        rand_uniform = torch.rand(n_runs)

        accepted = psi_ratio > rand_uniform
        accept_count += accepted.sum().item()

        mask = accepted.unsqueeze(1)

        r1 = torch.where(mask, r1_trial, r1)
        r2 = torch.where(mask, r2_trial, r2)


    acceptance_ratio = accept_count / (N * n_runs)
    #print(f"Acceptance ratio: {acceptance_ratio:.4f}")

    return torch.cat((r1, r2), dim=1)

def local_energy(positions, model):
    positions = positions.clone().detach().requires_grad_(True)

    # Split into r1, r2 vectors
    r1_vec = positions[:, 0:3]
    r2_vec = positions[:, 3:6]

    # Scalar distances
    r1 = torch.norm(r1_vec, dim=1, keepdim=True)
    r2 = torch.norm(r2_vec, dim=1, keepdim=True)
    r12 = torch.norm(r1_vec - r2_vec, dim=1, keepdim=True)

    scalar_features = torch.cat([r1, r2, r12], dim=1)

    log_psi_val = psi_nn(positions, model, return_log=False)

    grad_log_psi = torch.autograd.grad(log_psi_val.sum(), positions, create_graph=True)[0]
    grad_norm_sq = (grad_log_psi ** 2).sum(dim=1)

    laplacian = torch.zeros_like(log_psi_val)
    for i in range(positions.shape[1]):
        grad_i = grad_log_psi[:, i]
        grad2_i = torch.autograd.grad(
            grad_i.sum(), positions, create_graph=True
        )[0][:, i]
        laplacian += grad2_i

    kinetic = -0.5 * (laplacian/log_psi_val)

    # Potential energy: -2/r1 -2/r2 + 1/r12
    eps = 1e-8  # avoid div by 0
    potential = -2 / (r1.squeeze() + eps) - 2 / (r2.squeeze() + eps) + 1 / (r12.squeeze() + eps)

    return kinetic + potential


def get_local_energy(sampled_Xs, model):

    return local_energy(sampled_Xs, model)


def get_parameter_gradients(
        x: torch.tensor, E: torch.tensor, model: torch.nn,
        delta_I: float, optimization_type: int, pc: bool = False):
    
    fmodel, params = make_functional(model)
    # Ensure params are on the same device as the model
    device = next(model.parameters()).device
    params = [p.to(device) for p in params]

    def psi_func(params, x):
        return fmodel(params, x.unsqueeze(0)).squeeze()

    grad_log_psi = grad(psi_func)

    grads = vmap(grad_log_psi, in_dims=(None, 0))(params, x)
    flat_grads = torch.cat([g.reshape(x.shape[0], -1) for g in grads], dim=1)

    n_parameters = flat_grads.shape[-1]
    mean_grad = torch.mean(flat_grads, axis=0)
    
    E = E.flatten()
    mean_E = torch.mean(E)

    centered_E = E - mean_E
    centered_grads = flat_grads - mean_grad # Removed centering of grads here

    if optimization_type == 2:  # MinSR
        metric_tensor = (
        centered_grads @ centered_grads.T) / centered_grads.shape[0]
        metric_tensor = metric_tensor + (delta_I * torch.eye(
            metric_tensor.shape[0]).to(device))  # Sorella's trick for zero eigenvalues

        #if torch.linalg.det(metric_tensor) == 0:
        #    print("Cannot be inverted - breaking")
        #    exit()
        
        inv = torch.linalg.solve(metric_tensor, torch.eye(metric_tensor.shape[0]).to(device))
        #inv = torch.linalg.inv(metric_tensor)
        #if pc:  # Pre-conditioning
        #    outer = metric_diag.unsqueeze(0) * metric_diag.unsqueeze(1)
        #    metric_tensor = metric_tensor / torch.sqrt(outer)
        #    grads = torch.mean(-
        #        (grads.T * torch.sqrt(metric_diag)).T, axis=1)
        grads = centered_grads.T @ inv * centered_E
        grads = torch.mean(grads, axis=1)
        
        return grads.reshape(n_parameters, 1)


    grads = centered_grads.T * centered_E
    #grads = grads[:-1, :-1]

    if optimization_type == 1:
        metric_tensor = (
            centered_grads.T @ centered_grads) / centered_grads.shape[0]
        
        metric_diag = torch.diag(metric_tensor)
        
        if pc:  # Pre-conditioning
            outer = metric_diag.unsqueeze(0) * metric_diag.unsqueeze(1)
            pc_matrix = 1 / torch.sqrt(outer)
            metric_tensor = metric_tensor * pc_matrix
            grads = torch.mean(
                (grads.T * torch.sqrt(metric_diag)).T, axis=1)
        else:
            grads = torch.mean(grads, axis=1)

        
        metric_tensor = metric_tensor + (delta_I * torch.eye(
            metric_tensor.shape[0]).to(device))  # Sorella's trick for zero eigenvalues
        

        grads = torch.linalg.solve(metric_tensor, grads)

    else:
        grads = torch.mean(grads, axis=1)
    
    return grads.reshape(n_parameters, 1)

def assign_gradients_to_model(parameter_gradients, model):
    """Assign a flattened gradient vector to model parameters."""
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = parameter_gradients[pointer:pointer + numel].view_as(p).clone()
        pointer += numel
    return 

def get_mean_energies(E):
    return torch.mean(torch.mean(E, dim=1))

def get_variances(E):
    return torch.mean((E - torch.mean(E)) ** 2)


def main(
        epochs:int, warmup_steps:int, mc_steps:int, n_walkers:int, optimization_type:int,
        lr: float, delta_I: float, preconditioned: bool, keep_mc_steps: bool, running_on_hpc: bool,
        model_save_iterations: int, saved_model: bool, saved_model_path: str
    ):
    
    device = torch.device("cuda")
    cpu = torch.device("cpu")
    E_true = -2.9037243770

    losses = []

    if running_on_hpc:
        uid = str(datetime.datetime.now()).replace(
            ' ', '.').replace('-','').replace(':',"")
        savepath = os.path.join(os.environ['HOME'], os.path.join("training_logs", f"{uid}"))
        os.mkdir(savepath)
        model_savepath = os.path.join(savepath, "model.pth")
    else:
        model_savepath = ""


    total_params = sum(p.numel() for p in model.parameters())

    config = {
        "lr" : lr,
        "n_walkers": n_walkers,
        "mc_steps": mc_steps,
        "preconditioned": preconditioned,
        "delta_I": delta_I,
        "n_parameters": total_params,
        "keep_mc_steps": keep_mc_steps,
        "optimization_type": optimization_type,
        "model_savepath": model_savepath
    }

    print(config)
    print()

    optimizer = torch.optim.Adam(model.parameters())

    # Training monitoring on wandb
    wandb_login(running_on_hpc=running_on_hpc)
    start_wandb_run(config=config, project_name="vmc")

    pos = torch.rand((n_walkers, 6))

    for i in tqdm(range(epochs)):

        with torch.no_grad():
            pos = metropolis(warmup_steps, pos, n_walkers, model.to(cpu), keep_mc_steps=False)  
            pos = metropolis(mc_steps, pos, n_walkers, model, keep_mc_steps=keep_mc_steps)

        E = get_local_energy(
            pos.to(device), model.to(device))
        
        variance = get_variances(E)
        mean_energy = torch.mean(E)

        loss = torch.abs(E_true - torch.mean(E))
        losses.append(loss.item())

        r1 = torch.norm(pos[:, :3], dim=-1, keepdim=True)
        r2 = torch.norm(pos[:, 3:], dim=-1, keepdim=True)
        r12 = torch.norm(pos[:, :3] - pos[:, 3:], dim=-1, keepdim=True)
        grad_Xs = torch.cat([r1, r2, r12], dim=-1)

        grads = get_parameter_gradients(
            x=grad_Xs.to(device), E=E.to(device), model=model,
            optimization_type=optimization_type, delta_I=delta_I,
            pc=preconditioned)

        assign_gradients_to_model(grads, model)
        
        print(
            f"Iteration {i}\n"
            f"Mean energy is {mean_energy}\n"
            f"Loss is {loss}\n"
            f"Variance is {variance}\n")

        metrics = {
                "mean_energy": mean_energy,
                "variance": variance,
                "loss": loss
            }

        wandb.log(metrics)

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        del E
        del grad_Xs

        if (i+1) % model_save_iterations == 0 and running_on_hpc:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_savepath)


        if i == 200:
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "110_epoch_sr_pc.pth")