## Repository Structure

This repository allows running FL scenarios, attacks and defenses in IID and non-IID using a local server logic. It is organized as follows : A "Server" class allows running a vanilla FL scenario, in which a specified dataset with several instances of a "client" class given a specific "sampler", which distributes data according to the chosen IID/non-IID distribution. Attacks are implemented within the client class. To implement a defense, replace the server with a "DefenseServer" class which inherits the base server with the defense implemented. 
The repository is structured as follows :

- `Client/` - Client class.
- `Configs/` - Configuration files for defenses and training.
- `Defenses/` - Defense server implementations.
- `Models/` - Model architectures.
- `Samplers/` - Data sampling strategies.
- `Server/` - Server-side logic for federated learning.
- `Utils/` - Utility functions.
- `TestMain.py`, `TestMain_new.py` - Main entry points for running experiments.
- `benchmark.sh`, `benchmark_debug.sh` - Scripts to run experiments.
- `requirements.txt` - Python dependencies.

## Getting Started

### Installation

Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
The code uses python version 3.8 for all experiments

### Running Experiments

You can run experiments using the main script TestMain.py, for example with the following command:

```sh
python3 TestMain.py -algo fedCAM_dev -attack SignFlip -ratio 0.3 -dataset FashionMNIST -experiment debug -sampling CAM -lamda 0.3
```

Arguments:
- `-algo` : Defense algorithm (e.g., fedCAM, fedCVAE, fedGuard, noDefense)
- `-attack` : Attack type (e.g., NoAttack, SignFlip, SameValue, AdditiveNoise, NaiveBackdoor, etc.)
- `-ratio` : Ratio of attackers (e.g., 0.3)
- `-dataset` : Dataset to use (e.g., MNIST, FashionMNIST, CIFAR10, FEMNIST)
- `-experiment` : Experiment name (for logging and output)
- `-sampling` : Data sampling method (e.g., IID, CAM, Dirichlet_per_class)
- `-lamda` : Lambda parameter for trust propagation (optional, default: 0.3)

Otherwise, you can run experiments by batch using one of the benchmark scripts, which allow running multiple datasets, attacks and defenses