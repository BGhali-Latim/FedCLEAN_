# Hyperparameters for a federated learning system
# Language: Python

configs = { 
    "CifarCNN" : {
        "nb_classes": 10,
        # Clients Settings
        "num_epochs": 3,  # Number of epochs for training
        "batch_size": 64,  # Batch size for training
        "lr": 1e-3,  # Learning rate
        "num_classes": 10,  # Number of classes in the dataset
        "wd": 1e-5,  # Weight decay for Clients model
        #FL Settings
        "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
        "nb_rounds": 100,  # Maximum number of communication rounds for federated learning
        "aggregation": "FedAvg",  # Aggregation method for model updates
        "num_clients": 100,  # Total number of clients in the federated learning system
        "nb_clients_per_round": 50,  # Number of clients selected for each round
        "with_defence": False,  # Flag indicating if defense mechanism is enabled
        "size_trigger": 250,  # Trigger size for defense mechanism
        "size_test": 500,
    },

    "ResNet" : {
        "dataset": "CIFAR10",
        "nb_classes": 10,
        # Clients Settings
        "num_epochs": 1,  # Number of epochs for training
        "batch_size": 512,  # Batch size for training
        "lr": 1e-5,  # Learning rate
        "num_classes": 10,  # Number of classes in the dataset
        "wd": 1e-6,  # Weight decay for Clients model
        #FL Settings
        "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
        "nb_rounds": 100,  # Maximum number of communication rounds for federated learning
        "aggregation": "FedAvg",  # Aggregation method for model updates
        "num_clients": 100,  # Total number of clients in the federated learning system
        "nb_clients_per_round": 50,  # Number of clients selected for each round
        "with_defence": False,  # Flag indicating if defense mechanism is enabled
        "size_trigger": 250,  # Trigger size for defense mechanism
        "size_test": 500,
    },

    "CNN" : {
        "nb_classes": 10,
        # Clients Settings
        "num_epochs": 3,  # Number of epochs for training
        "batch_size": 64,  # Batch size for training
        "lr": 1e-3,  # Learning rate
        "num_classes": 10,  # Number of classes in the dataset
        "wd": 1e-5,  # Weight decay for Clients model
        #FL Settings
        "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
        "nb_rounds": 100,  # Maximum number of communication rounds for federated learning
        "aggregation": "FedAvg",  # Aggregation method for model updates
        "num_clients": 100,  # Total number of clients in the federated learning system
        "nb_clients_per_round": 50,  # Number of clients selected for each round
        "with_defence": False,  # Flag indicating if defense mechanism is enabled
        "size_trigger": 250,  # Trigger size for defense mechanism
        "size_test": 500,
    },

}