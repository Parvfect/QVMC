# Recreate as the model config file
import argparse
import os 
from datetime import datetime
from cluster_nn import main


parser = argparse.ArgumentParser(
                    prog='QVMC MLP Helium',
                    description='Optimising for the ground state of Helium using a MLP',
                    epilog='Use parser to set some parameters for training')

parser.add_argument('--epochs', type=int, help="Training epochs")
parser.add_argument('--warmup_steps', type=int, help="Number of warmup steps")
parser.add_argument('--mc_steps', type=int, help="Number of metropolis steps")
parser.add_argument('--n_walkers', type=int, help="Number of walkers")
parser.add_argument(
    '--running_on_hpc', action='store_true', help="Flag for running on hpc")
parser.add_argument('--optimization_type', type=int, help="0 - ADAM, 1 - SR, 2 - MinSR")
parser.add_argument('--preconditioned', action='store_true', help="Preconditioning the metric tensor")
parser.add_argument('--delta_I', type=float, help="constant for XX + delta* I addtion in sorella trick")
parser.add_argument('--keep_mc_steps', action='store_true', help="Estimate gradients and energy through all mc steps")
parser.add_argument('--model_save_iterations', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--saved_model', action='store_true', help='Load from saved model')
parser.add_argument('--saved_model_path', type=str, help="Saved model path")


parser.set_defaults(
    epochs=100000, warmup_steps=200, mc_steps=50, n_walkers=4096, running_on_hpc=False, optimization_type=0, preconditioned=False, delta_I=0.04, keep_mc_steps=False, lr=0.01, saved_model=False, saved_model_path="", model_save_iterations=50
    )

args = parser.parse_args()

if __name__ == '__main__':
    epochs = args.epochs
    running_on_hpc = args.running_on_hpc
    lr = args.lr
    warmup_steps = args.warmup_steps
    mc_steps = args.mc_steps
    n_walkers = args.n_walkers
    optimization_type = args.optimization_type
    preconditioned = args.preconditioned
    delta_I = args.delta_I
    keep_mc_steps = args.keep_mc_steps
    model_save_iterations = args.model_save_iterations
    saved_model = args.saved_model
    saved_model_path = args.saved_model_path

    main(
        epochs=epochs, warmup_steps=warmup_steps, mc_steps=mc_steps,
        n_walkers=n_walkers, optimization_type=optimization_type, lr=lr,
        delta_I=delta_I, preconditioned=preconditioned, keep_mc_steps=keep_mc_steps,
        running_on_hpc=running_on_hpc, model_save_iterations=model_save_iterations, saved_model=saved_model, saved_model_path=saved_model_path
    )
