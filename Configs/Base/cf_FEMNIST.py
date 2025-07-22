# Hyperparameters for a federated learning system
# Language: Python

configs = { 
    "CNN" : {
        # Clients Settings
        "num_epochs": 2,# 5,  # Number of epochs for training
        "batch_size": 64,# 128,  # Batch size for training
        "lr": 5e-4,# 5e-4,  # Learning rate
        "num_classes": 10,  # Number of classes in the dataset
        "wd": 1e-5,  # Weight decay for Clients model

        #FL Settings
        "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
        "nb_rounds": 50,  # Maximum number of communication rounds for federated learning
        "aggregation": "FedAvg",  # Aggregation method for model updates
        "num_clients": 100,  # Total number of clients in the federated learning system
        "nb_clients_per_round": 50,  # Number of clients selected for each round
        "size_trigger": 1200,  # Trigger size for defense mechanism
        
        # FEMNIST Settings
        "FEMNIST_num_clients": 3580, # This is fixed because the femnist dataset is already split into clients by handwriting
        "FEMNIST_num_clients_train" : 100, # This is nn number of clients
        "FEMNIST_num_clients_test" : 500, # This is in number of clients
        "FEMNIST_num_clients_trigger" : 50, # This is in number of clients
        "FEMNIST_nb_clients_per_round" : 50, # Number of clients selected for each round for FEMNIST

        "with_defence": False,  # Flag indicating if defense mechanism is enabled
    },

    "CifarCNN" : {
        # Clients Settings
        "num_epochs": 3,# 5,  # Number of epochs for training
        "batch_size": 128,# 128,  # Batch size for training
        "lr": 5e-4,# 5e-4,  # Learning rate
        "num_classes": 10,  # Number of classes in the dataset
        "wd": 1e-5,  # Weight decay for Clients model
        #FL Settings
        "data_dist": "non-IID", # For the moment we are considering only the IID data distribution
        "nb_rounds": 500,  # Maximum number of communication rounds for federated learning
        "aggregation": "FedAvg",  # Aggregation method for model updates
        "num_clients": 100,  # Total number of clients in the federated learning system
        "nb_clients_per_round": 50,  # Number of clients selected for each round
        
        # FEMNIST Settings
        "FEMNIST_num_clients": 3580, # This is fixed because the femnist dataset is already split into clients by handwriting
        "FEMNIST_num_clients_train" : 3030, # This is nn number of clients
        "FEMNIST_num_clients_test" : 500, # This is in number of clients
        "FEMNIST_num_clients_trigger" : 50, # This is in number of clients
        "FEMNIST_nb_clients_per_round" : 303, # Number of clients selected for each round for FEMNIST
        "with_defence": False,  # Flag indicating if defense mechanism is enabled
    }
}