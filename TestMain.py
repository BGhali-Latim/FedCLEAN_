import unittest
import torch
from Models.MLP import MLP, DropoutMLP
from Models.CNN import CNN, GuardCNN, CifarCNN, Net, CifarCNN2, CNNWithDropBlock
from Models.AlexNet import AlexNet
from Models.ConvMixer_reimpl import ConvMixer
from Models.Resnet import CustomResNet

from Configs.cf_NoDef import configs_noDef as cf_ndf
from Configs.cf_fedCAM_dev import configs_fedCAM as cf_dev

from Configs.cf_fedCAM import configs_fedCAM as cf
from Configs.cf_fedCAM_dev import configs_fedCAM as cf_dev
from Configs.cf_fedCAM_prod import configs_fedCAM as cf_prod
from Configs.cf_fedCAM_cos import configs_fedCAM_cos as cf_cos
from Configs.cf_fedCVAE import configs_fedCVAE as cf_cvae
from Configs.cf_fedGuard import configs_fedGuard

from Samplers import IID_sampler, CAMSampler, DirichletSampler, NaturalSampler

# from Configs.cf_fedGuard_1000 import configs_fedGuard as cf_fedGuard_1000
import argparse
import random


class TestMain(unittest.TestCase):

    def test_main(
        self,
        algo="fedCam",
        attack_type="NoAttack",
        attacker_ratio=0.3,
        dataset="FashionMNIST",
        experiment="debug",
        sampling = "IID",
        lamda = 0.3
    ):
        # Fix seed
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Set the device to GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Set model
        #model = CNN().to(device)
        model = CNNWithDropBlock().to(device)
        #model = AlexNet().to(device)
        #model = CustomResNet().to(device)
        #model = ConvMixer(dim = 576, depth = 1, kernel_size = 5, patch_size = 9, device = device).to(device)

        samplers = {
            "IID" : IID_sampler.ClientSampler(cf_ndf), 
            "CAM" : CAMSampler.CAMSampler(cf_ndf), 
            "Dirichlet_per_class" : DirichletSampler.DirichletSampler(cf_ndf, method = "Dirichlet_per_class"),
            "Dirichlet_per_client" : DirichletSampler.DirichletSampler(cf_ndf, method = "Dirichlet_per_client"),
        }
        try :
            sampler = samplers[sampling]
        except KeyError : 
            print("Please choose a valid client data sampling strategy")

        # Import defese server
        if algo == "noDefense":
            from Server.Server import Server

            server = Server(
                cf=cf_ndf,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
            )
        elif algo == "fedCAM":
            from Defenses.FedCAMServer import DefenseServer

            server = DefenseServer(
                cf=cf,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
            )
        elif algo == "fedCAM_dev":
            from Defenses.FedCAMServer_dev import DefenseServer

            server = DefenseServer(
                cf=cf_dev,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
                lamda = lamda
            )
        elif algo == "fedCAM_prod":
            from Defenses.FedCAMServer_prod import DefenseServer
            server = DefenseServer(
                cf=cf_prod,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                experiment_name=experiment,
            )
        elif algo == "fledge":
            from Defenses.FledgeServer import DefenseServer
            server = DefenseServer(
                cf=cf_ndf,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
            )
        # elif algo == "fedCAM_conv":
        # from Defenses.FedCAMServer_conv import DefenseServer
        # server = DefenseServer(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCAM_cos":
            from Defenses.FedCAMServer_cos import DefenseServer

            server = DefenseServer(
                cf=cf_cos,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                experiment_name=experiment,
            )
        # elif algo == "fedCAM2_cos":
        # from Defenses.FedCAM_cos_2 import Server
        # server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        # elif algo == "fedCWR":
        # from Defenses.FedCWR import Server
        # server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        # elif algo == "fedCAM2":
        # from Defenses.FedCAM2 import Server
        # server = Server(cf=cf, model=model, attack_type=attack_type, attacker_ratio=attacker_ratio, dataset=dataset)
        elif algo == "fedCVAE":  # FedCVAE in this case
            from Defenses.FedCVAEServer import DefenseServer

            server = DefenseServer(
                cf=cf_cvae,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
            )
        elif algo == "fedGuard":
            from Defenses.FedGuardServer import DefenseServer

            server = DefenseServer(
                cf=configs_fedGuard,
                model=model,
                attack_type=attack_type,
                attacker_ratio=attacker_ratio,
                dataset=dataset,
                sampler = sampler,
                experiment_name=experiment,
            )
        else:
            print("Please specify a valid -algo argument (e.g., fedCam, fedCvae)")

        # Run scenario
        print("started")
        server.run(log=True)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script corresponds to the implementation of FedCVAE and FedCAM"
    )

    # Add an -algo argument to specify the algorithm
    parser.add_argument("-algo", type=str, help="The name of the defense system")
    parser.add_argument("-attack", type=str, help="The type of attack")
    parser.add_argument("-ratio", type=float, help="The ratio of attackers")
    parser.add_argument("-dataset", type=str, default="FashionMNIST")
    parser.add_argument("-experiment", type=str, default="debug")
    parser.add_argument("-sampling", type=str, default="IID")
    parser.add_argument("-lamda", type=float, default=0.3)

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the TestMain class
    test_instance = TestMain()

    # Call the test_main function with the specified algorithm from the arguments
    if args.algo:
        test_instance.test_main(
            algo=args.algo,
            attack_type=args.attack,
            attacker_ratio=args.ratio,
            dataset=args.dataset,
            experiment=args.experiment,
            sampling = args.sampling,
            lamda = args.lamda
        )
    else:
        # Print a message if the -algo argument is not specified in the command line
        print("Please specify the -algo argument in the command line.")
