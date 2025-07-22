from copy import deepcopy
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
from Models.autoencoders import VAE
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
import random


class Server:
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cf = cf

        # Saving directory
        if not cf["with_defence"]:
            self.dir_path = f"Results/{self.cf['dataset']}/NoDefence/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        else:
            self.dir_path = f"Results/{self.cf['dataset']}/FedCAM2/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"

        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        self.activation_size = cf["vae_input_dim"]
        self.num_classes = cf["num_classes"]
        self.nb_rounds = cf["nb_rounds"]
        self.global_model = (
            model.to(self.device)
            if model
            else MLP(self.activation_size).to(self.device)
        )
        self.defence = cf["with_defence"]
        self.attack_type = attack_type
        self.attacker_ratio = attacker_ratio

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": attacker_ratio,
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"],
        }

        self.config_vae = {
            "vae_nb_ep": cf["vae_nb_ep"],
            "vae_lr": cf["vae_lr"],
            "vae_wd": cf["vae_wd"],
            "vae_gamma": cf["vae_gamma"],
        }

        # self.train_for_test = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        # self.train_for_test_loader = DataLoader(self.train_for_test, cf["batch_size"], shuffle=False, drop_last=False)
        # self.class_indexes = [(self.train_for_test.targets == idx).nonzero().reshape(-1) for idx in range(self.cf["nb_classes"])]
        # self.class_datasets = [Subset(self.train_for_test,self.class_indexes[idx]) for idx in range(self.cf["nb_classes"])]

        print("distributing data among clients")
        if self.cf["data_dist"] == "IID":
            self.train_data = Utils.distribute_non_iid_data_among_clients(
                self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"]
            )
        elif self.cf["data_dist"] == "non-IID":
            self.train_data = Utils.distribute_non_iid_data_among_clients(
                self.config_FL["num_clients"], cf["batch_size"], dataset=cf["dataset"]
            )
        print("generating clients")
        self.clients = Utils.gen_clients(
            self.config_FL,
            self.attack_type,
            self.train_data,
            backdoor_label=cf["source"],
        )

        self.are_attackers = np.array(
            [int(client.is_attacker) for client in self.clients]
        )
        self.are_benign = np.array(
            [int(not (client.is_attacker)) for client in self.clients]
        )

        self.vae_trained = False

        self.trigger_loader, self.test_loader = Utils.get_test_data(
            cf["size_trigger"], cf["dataset"]
        )

        self.vae = VAE(
            input_dim=cf["vae_input_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"],
        ).to(self.device)

        self.accuracy = []
        self.accuracy_on_train = []
        self.class_accuracies = []
        self.accuracy_backdoor = []
        self.nb_attackers_history = []
        self.nb_attackers_passed_defence_history = []
        self.nb_benign_history = []
        self.nb_benign_passed_defence_history = []

        self.attacker_precision_hist = []
        self.attacker_recall_hist = []

        self.attacker_profiles_per_round = []
        self.passing_attacker_profiles_per_round = []
        self.benign_profiles_per_round = []
        self.passing_benign_profiles_per_round = []

        self.best_accuracy, self.best_round = 0, 0

        self.histo_selected_clients = torch.tensor([])

    def train_vae(self):
        if self.vae_trained:
            print("vae is already trained, skipping re-training.")
            return
        num_epochs = self.config_vae["vae_nb_ep"]
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.config_vae["vae_lr"],
            weight_decay=self.config_vae["vae_wd"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[200, 300], gamma=self.config_vae["vae_gamma"]
        )
        model = deepcopy(self.global_model)
        model.eval()
        # act_means = torch.zeros((self.num_classes, self.activation_size)).to(self.device)
        act_means = torch.normal(
            mean=0.0,
            std=1,
            size=(self.num_classes, self.activation_size),
            device=self.device,
        )
        act_means = torch.tanh(act_means)
        for epoch in range(num_epochs):
            train_loss = 0
            loop = tqdm(self.trigger_loader, leave=True)
            for batch_idx, _ in enumerate(loop):
                recon_batch, mu, logvar = self.vae(act_means)
                loss = Utils.cvae_loss(recon_batch, act_means, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1))
            scheduler.step()
        self.vae_trained = True

    def compute_reconstruction_error(self, selected_clients):
        self.vae.eval()
        clients_re = torch.zeros((len(selected_clients), self.num_classes)).to(
            self.device
        )
        act_means = torch.zeros(
            (len(selected_clients), self.num_classes, self.activation_size)
        ).to(self.device)

        for client_num, client in enumerate(selected_clients):
            client.model.eval()
            for data, label in self.trigger_loader:
                data, label = data.to(self.device), label.to(self.device)
                activation = client.model.get_activations(data).detach()
                # Since we have one batch, we don't t need to divide by the number of batch for act_means
                for i in range(self.num_classes):
                    act_means[client_num][i] = torch.mean(activation[label == i], dim=0)

        # mean = torch.mean(act_means, dim=0)
        # act_means = (act_means - mean)
        gm = compute_geometric_median(
            act_means.cpu(), weights=None
        )  # equivalent to `weights = torch.ones(n)`.
        gm = gm.median.to(self.device)

        # print("here5", act_means)
        act_means = act_means - gm
        # print("here2", act_means)
        act_means = torch.tanh(act_means)

        for i, client_act in enumerate(act_means):
            recon_batch, mu, logvar = self.vae(client_act)
            mse = F.mse_loss(recon_batch, client_act, reduction="none")
            mse = torch.mean(mse, dim=1)
            clients_re[i] = mse

        # Normalization step
        # # Calcul de la moyenne et de l'écart type pour chaque colonne
        # mean = clients_re.mean(dim=0)
        # std = clients_re.std(dim=0)
        #
        # # Normalisation du tensor
        # clients_re = (clients_re - mean) / std

        # Calculer les valeurs min et max pour chaque colonne (classe)
        min_vals = clients_re.min(dim=0)[0]
        max_vals = clients_re.max(dim=0)[0]

        # Appliquer la normalisation Max-Min
        clients_re = (clients_re - min_vals) / (max_vals - min_vals)
        res = clients_re.mean(dim=1)

        return res.tolist()

    def run(self):
        t_start = datetime.datetime.now()

        if self.defence:
            if not self.vae_trained:
                self.train_vae()
                self.vae_trained = True
                # if not self.cf["skip_vae"]:
                #    self.train_vae()
                #    self.vae_trained = True

        # REVERT
        total_attackers_passed = 0
        total_attackers = 0

        for rounds in range(self.cf["nb_rounds"]):
            torch.cuda.empty_cache()

            selected_clients = Utils.select_clients(
                self.clients, self.config_FL["nb_clients_per_round"]
            )

            for client in tqdm(selected_clients):
                client.set_model(deepcopy(self.global_model).to(self.device))
                client.train(self.cf)

            if self.defence:
                clients_re = self.compute_reconstruction_error(selected_clients)
                clients_re_np = np.array(clients_re)
                valid_values = clients_re_np[np.isfinite(clients_re_np)]

                max_of_re = np.max(valid_values)
                mean_of_re = np.mean(valid_values)

                clients_re_without_nan = np.where(
                    np.isnan(clients_re_np) | (clients_re_np == np.inf),
                    max_of_re,
                    clients_re_np,
                )

                selected_clients_array = np.array(selected_clients)
                good_updates = selected_clients_array[
                    clients_re_without_nan < mean_of_re
                ]
                for client in selected_clients_array[
                    clients_re_without_nan >= mean_of_re
                ]:
                    client.suspect()
            else:
                good_updates = selected_clients

            self.histo_selected_clients = torch.cat(
                (
                    self.histo_selected_clients,
                    torch.tensor([client.id for client in good_updates]),
                )
            )

            nb_attackers = np.array(
                [client.is_attacker for client in selected_clients]
            ).sum()
            nb_benign = np.array(
                [not client.is_attacker for client in selected_clients]
            ).sum()
            nb_attackers_passed = np.array(
                [client.is_attacker for client in good_updates]
            ).sum()
            nb_benign_passed = np.array(
                [not client.is_attacker for client in good_updates]
            ).sum()

            self.nb_attackers_history.append(nb_attackers)
            self.nb_attackers_passed_defence_history.append(nb_attackers_passed)
            self.nb_benign_history.append(nb_benign)
            self.nb_benign_passed_defence_history.append(nb_benign_passed)

            print(
                "Total of Selected Clients ",
                len(selected_clients),
                ", Number of attackers ",
                nb_attackers,
                ", Total of attackers passed defense ",
                nb_attackers_passed,
                " out of ",
                len(good_updates),
                " total updates",
            )

            # Aggregation step
            self.global_model.load_state_dict(Utils.aggregate_models(good_updates))

            # Add the accuracy of the current global model to the accuracy list
            # self.accuracy_on_train.append(Utils.test(self.global_model, self.device, self.train_for_test_loader))
            # self.class_accuracies.append(Utils.get_class_accuracies(self.global_model, self.device, self.class_datasets, nb_classes=self.cf["nb_classes"]))
            self.accuracy.append(
                test_acc := Utils.test(self.global_model, self.device, self.test_loader)
            )

            if test_acc > self.best_accuracy:
                self.best_accuracy, self.best_round = test_acc, rounds + 1

            if self.attack_type in [
                "NaiveBackdoor",
                "SquareBackdoor",
                "NoiseBackdoor",
                "MajorityBackdoor",
                "TargetedBackdoor",
            ]:
                self.accuracy_backdoor.append(
                    Utils.test_backdoor(
                        self.global_model,
                        self.device,
                        self.test_loader,
                        self.attack_type,
                        self.cf["source"],
                        self.cf["target"],
                        self.cf["square_size"],
                    )
                )

            print(
                f"Round {rounds + 1}/{self.cf['nb_rounds']} server test accuracy: {self.accuracy[-1] * 100:.2f}%"
            )
            if self.attack_type in [
                "NaiveBackdoor",
                "SquareBackdoor",
                "NoiseBackdoor",
                "MajorityBackdoor",
                "TargetedBackdoor",
            ]:
                print(
                    f"Round {rounds + 1}/{self.cf['nb_rounds']} attacker accuracy: {self.accuracy_backdoor[-1] * 100:.2f}%"
                )

            # Clean up after update step (to save GPU memory)
            for client in tqdm(selected_clients):
                client.remove_model()

            # REVERT
            total_attackers_passed += nb_attackers_passed
            total_attackers += nb_attackers

            round_suspects = np.array(
                [int(client.is_suspect) for client in selected_clients]
            )
            round_unsespected = np.array(
                [int(not (client.is_suspect)) for client in selected_clients]
            )
            round_attackers = np.array(
                [int(client.is_attacker) for client in selected_clients]
            )
            round_benign = np.array(
                [int(not (client.is_attacker)) for client in selected_clients]
            )

            TP = sum(round_suspects * round_attackers)
            FP = sum(round_suspects * round_benign)
            FN = sum(round_unsespected * round_attackers)

            self.attacker_precision_hist.append(TP / (TP + FP + 1e-10))
            self.attacker_recall_hist.append(TP / (TP + FN + 1e-10))
            print(f"attacker detection recall : {self.attacker_recall_hist[-1]}")

            attacker_profile_this_round = []
            passing_attacker_profile_this_round = []
            benign_profile_this_round = []
            passing_benign_profile_this_round = []

            for client in selected_clients:
                if client.is_attacker:
                    attacker_profile_this_round.append(client.num_samples)
                else:
                    benign_profile_this_round.append(client.num_samples)

            for client in good_updates:
                if client.is_attacker:
                    passing_attacker_profile_this_round.append(client.num_samples)
                else:
                    passing_benign_profile_this_round.append(client.num_samples)

            self.passing_attacker_profiles_per_round.append(
                passing_attacker_profile_this_round
            )
            self.attacker_profiles_per_round.append(attacker_profile_this_round)
            self.passing_benign_profiles_per_round.append(
                passing_benign_profile_this_round
            )
            self.benign_profiles_per_round.append(benign_profile_this_round)

        print(f"finished running server in {datetime.datetime.now() - t_start}")

        # Save the training config
        Utils.save_to_json(self.cf, self.dir_path, "run_config.json")

        # Saving The accuracies of the Global model on the testing set and the backdoor set
        Utils.save_to_json(
            self.accuracy, self.dir_path, f"test_accuracy_{self.cf['nb_rounds']}"
        )
        if self.attack_type in [
            "NaiveBackdoor",
            "SquareBackdoor",
            "NoiseBackdoor",
            "MajorityBackdoor",
            "TargetedBackdoor",
        ]:
            Utils.save_to_json(
                self.accuracy_backdoor,
                self.dir_path,
                f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}",
            )

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.pdf"
        Utils.plot_hist(
            self.histo_selected_clients,
            x_info="Clients",
            y_info="Frequencies",
            title_info="",
            bins=1000,
            save_path=save_path,
        )

        # Saving the percentage of attackers blocked
        Utils.save_to_json(
            (total_attackers_passed / total_attackers) * 100,
            self.dir_path,
            f"successful_attacks",
        )
        # Detection qulity metrics to JSON
        Utils.save_to_json(
            self.attacker_precision_hist,
            self.dir_path,
            f"attacker_detection_precision_{self.cf['nb_rounds']}",
        )
        Utils.save_to_json(
            self.attacker_recall_hist,
            self.dir_path,
            f"attacker_detection_recall_{self.cf['nb_rounds']}",
        )
        # Attacker profiles to json
        Utils.save_to_json(
            self.attacker_profiles_per_round,
            self.dir_path,
            f"attacker_profiles_{self.cf['nb_rounds']}",
        )
        Utils.save_to_json(
            self.passing_attacker_profiles_per_round,
            self.dir_path,
            f"passing_attacker_profiles_{self.cf['nb_rounds']}",
        )
        Utils.save_to_json(
            self.benign_profiles_per_round,
            self.dir_path,
            f"benign_profiles_{self.cf['nb_rounds']}",
        )
        Utils.save_to_json(
            self.passing_benign_profiles_per_round,
            self.dir_path,
            f"passing_benign_profiles_{self.cf['nb_rounds']}",
        )

        # Plotting the testing accuracy of the global model
        title_info = f"Test Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Test_Accuracy_{self.cf['nb_rounds']}.pdf"
        Utils.plot_accuracy(
            self.accuracy,
            x_info="Round",
            y_info="Test Accuracy",
            title_info=title_info,
            save_path=save_path,
        )

        if self.attack_type in [
            "NaiveBackdoor",
            "SquareBackdoor",
            "NoiseBackdoor",
            "MajorityBackdoor",
            "TargetedBackdoor",
        ]:
            # Plotting the backdoor accuracy
            title_info = f"Backdoor Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
            save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Backdoor_Accuracy_{self.cf['nb_rounds']}.pdf"
            Utils.plot_accuracy(
                self.accuracy_backdoor,
                x_info="Round",
                y_info="backdoor Accuracy",
                title_info=title_info,
                save_path=save_path,
            )

        ## Plotting the histogram of the defense system
        # Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
        #                     self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
        #                     self.attack_type, self.defence, self.dir_path, success_rate=f"{(1-(total_attackers_passed/total_attackers))*100:.2f}%",
        #                     attacker_ratio=self.attacker_ratio)

        # Print some stats
        print(
            f"In total : Number of attackers : {total_attackers}, \
        \nTotal of attackers passed defense : {total_attackers_passed} ({(total_attackers_passed/total_attackers)*100:.2f}%)"
        )

        print(
            f"Best test accuracy : {self.best_accuracy*100:.2f}. Achieved in {self.best_round} rounds"
        )
