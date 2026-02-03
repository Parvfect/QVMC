# QVMC

Quantum Variational Monte Carlo (QVMC) Project

## Overview

This repository contains code and resources for implementing and experimenting with Quantum Variational Monte Carlo (QVMC) methods. QVMC is a computational technique used to estimate ground state energies and properties of quantum systems using variational principles and Monte Carlo sampling.

## Innards

- `helium_analytical`: Optimization of a 4-parameter (analtyical) helium model, achieving results within 0.003 au of the true energy (David, Carl W. "Compact Singlet S Helium Wave functions (corrected)." (2006).). Implementation of Metropolis sampling, analytical gradients and energy expressions and optimization using ADAM.
- `nn_helium`: Simple neural network ansatz for the helium atom
- 'nnqs_jax' : Starting simple general implementation for NQS in JAX. Ground state, TVMC, etc

## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/Parvfect/QVMC.git
    ```
2. Install dependencies as specified in `requirements.txt`.
3. Run the main script or explore the notebooks for examples.

## Directory Structure

- `src/` - Source code for QVMC algorithms
- `examples/` - Example scripts and usage
- `tests/` - Unit tests
- `helium_analytical/` - Analytical optimization for helium
- `nn_helium/` - Neural network ansatz for helium
- `readme.txt` - Project documentation

## Usage

Refer to the example scripts in the `examples/` directory for typical usage patterns.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.
