# Hyperparameters for a federated learning system
# Language: Python

config = {
    "dataset": "MNIST",
    "nb_classes": 10,

    # Clients Settings
    "num_epochs": 3,# 5,  # Number of epochs for training
    "batch_size": 64,# 128,  # Batch size for training
    "lr": 1e-3,# 5e-4,  # Learning rate
    "num_classes": 10,  # Number of classes in the dataset
    "wd": 1e-5,  # Weight decay for Clients model
    
    #FL Settings
    "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
    "nb_rounds": 100,  # Maximum number of communication rounds for federated learning
    "aggregation": "FedAvg",  # Aggregation method for model updates
    "num_clients": 100,  # Total number of clients in the federated learning system
    "nb_clients_per_round": 50,  # Number of clients selected for each round

    #FedGuard CVAE settings :
    "cvae_input_dim": 784, # To change for layer list,
    "condition_dim": 10,
    "latent_dim": 20,
    "nb_samples": 10,
    "cvae_nb_ep": 5,
    "cvae_lr": 1e-2,
    "cvae_wd": 0,
    "cvae_gamma": 1, 

    # Attacks/Defenses Settings
    "with_defence": True,  # Flag indicating if defense mechanism is enabled
    #"size_trigger": 100,  # Trigger size for defense mechanism
    #"attacker_ratio": 0.3,  # Ratio of attackers in the system
    #"attack_type": 'SignFlip',  # Type of attack (e.g., SameValue, AdditiveNoise)
    # 0: 'NoAttack' 1: 'AdditiveNoise', 2: 'SameValue', 3: 'SignFlip',  4: 'NaiveBackdoor', 5: 'SquareBackdoor', 6 : 'SameSample', 7 : "NoiseBackdoor"
    "size_trigger": 250,  # Trigger size for defense mechanism
    "size_test": 500,

    # Parameters of  NaiveBackdoor and SquareBackdoor attacks
    "source": 7,
    "target": 5,
    "square_size": 10,
    "back_noise_avg": 0.5,
    "back_noise_std": 0.5,

    # GeoMEd parameters
    "eps": 1e-8,  # Epsilon value for numerical stability of goeMed
    "iter": 100,  # Maximum number of iterations
}