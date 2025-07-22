from copy import deepcopy, copy
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from Models.MLP import MLP
from Utils.Utils import Utils
import datetime 
from torch.utils.tensorboard import SummaryWriter

class Server():
    def __init__(self, cf=None, model=None, attack_type=None, attacker_ratio=None, dataset = None, sampler = None, experiment_name = 'debug'):
        super().__init__()
        # No defense server
        self.defence = False

        # Experiment config
        self.cf = cf
        self.dataset = dataset
        self.sampler = sampler

        # Saving directory
        self.dir_path = f"Results/{experiment_name}/{self.dataset}/{sampler.name}/NoDefense/{self.cf['data_dist']}_{int(attacker_ratio * 100)}_{attack_type}"
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path) 

        # FL parameters
        self.nb_rounds = cf["nb_rounds"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = model.to(self.device) if model else MLP().to(self.device)
        self.config_FL = {"num_clients": cf["FEMNIST_num_clients_train"] if self.dataset == "FEMNIST" else cf["num_clients"],
                        "nb_clients_per_round": cf["FEMNIST_nb_clients_per_round"] if self.dataset == "FEMNIST" else cf["nb_clients_per_round"],
                        "attackers_ratio": attacker_ratio,
                        "batch_size": cf["batch_size"]}
        self.aggregation = cf['aggregation']

        # SCAFFOLD
        variate_model = deepcopy(self.global_model)
        self.global_variate = variate_model.parameters()

        # Attack scenario parameters
        self.attack_type = attack_type
        self.attacker_ratio = attacker_ratio

        # Evaluation metrics 
        self.accuracy = []
        self.accuracy_on_train = []
        self.class_accuracies = []
        self.accuracy_backdoor = []
        self.nb_attackers_history = []
        self.nb_attackers_passed_defence_history = []
        self.nb_benign_history = []
        self.nb_benign_passed_defence_history = []
        self.total_attackers_passed = 0 
        self.total_benign_blocked = 0 
        self.total_attackers = 0
        self.total_benign = 0
        self.attacker_precision_hist = []
        self.attacker_recall_hist = []
        self.benign_recall_hist = []
        self.pred_labels = []
        self.true_labels = []
        self.attacker_profiles_per_round = []
        self.passing_attacker_profiles_per_round = []
        self.benign_profiles_per_round = []
        self.passing_benign_profiles_per_round = []
        self.best_accuracy, self.best_round = 0, 0
        self.histo_selected_clients = torch.tensor([])

        # Distribute train data 
        # print("distributing data among clients")
        # if self.cf['data_dist'] == "IID" :
            # self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=self.dataset)
        # elif self.cf['data_dist'] == "non-IID" :
            # self.train_data = Utils.distribute_non_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"], dataset=self.dataset)
        print("distributing data among clients")
        if self.cf['data_dist'] == "IID" :
            self.train_data = self.sampler.distribute_iid_data(self.dataset)
        elif self.cf['data_dist'] == "non-IID" :
            self.train_data = self.sampler.distribute_non_iid_data(self.dataset)

        # Get test data
        #print("getting test data")
        #if self.dataset == "FEMNIST" :
        #    print("splitting train into trigger and test")
        #    if (cf["FEMNIST_num_clients_train"]
        #        +cf["FEMNIST_num_clients_test"]
        #        +cf["FEMNIST_num_clients_trigger"]) != cf["FEMNIST_num_clients"] : 
        #        # Provided invalid client repartition
        #        print("Invalid client split for FEMNIST") 
        #        exit()
        #    else :
        #        self.train_data, self.trigger_loader, self.test_loader = Utils.split_train(self.train_data, 
        #                                                                                   cf["FEMNIST_num_clients_train"], 
        #                                                                                   cf["FEMNIST_num_clients_trigger"], 
        #                                                                                   cf["FEMNIST_num_clients_test"])
        #else : 
        #    print("loading trigger and test sets")
        #    self.trigger_loader, self.test_loader = Utils.get_test_data(cf["size_trigger"], self.dataset)
        print("loading trigger and test sets")
        self.trigger_loader, self.test_loader = self.sampler.get_test_data(cf["size_trigger"], self.dataset)
        
        # Generate clients
        print("generating clients")
        self.clients = Utils.gen_clients(self.config_FL["num_clients"], self.attacker_ratio, self.attack_type, self.train_data)
        self.are_attackers = np.array([int(client.is_attacker) for client in self.clients])
        self.are_benign = np.array([int(not(client.is_attacker)) for client in self.clients])

        # Utils.plot_distrib(self.clients, title = "Dirichlet distribution", file_path=f"{sampler.name}_bar.pdf")
        # Utils.compare_distribs(self.clients, file_path = f"{sampler.name}_combined.pdf", storage = "cam_clients", load = True)

        self.lr = self.cf['lr']
        
        #self.train_for_test = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        #self.train_for_test_loader = DataLoader(self.train_for_test, cf["batch_size"], shuffle=False, drop_last=False)
        #self.class_indexes = [(self.train_for_test.targets == idx).nonzero().reshape(-1) for idx in range(self.cf["nb_classes"])]
        #self.class_datasets = [Subset(self.train_for_test,self.class_indexes[idx]) for idx in range(self.cf["nb_classes"])]

        #for sample in self.trigger_loader :
        #    print(f"trigger data sample : {sample[0].size()}")
        #    print(f"trigger label sample : {sample[1].size()}, {sample[1]}")
        #    
        #for sample in self.test_loader :
        #    print(f"test data sample : {sample[0].size()}")
        #    print(f"test label sample : {sample[1].size()}, {sample[1]}")        

    def prepare_defense(self):
        pass 

    def compute_client_errors(self, clients): 
        pass 

    def filter_clients(self, clients): 
        # No defense
        return clients

    def run(self, log = False):
        t_start = datetime.datetime.now()

        if log :
            writer = SummaryWriter()

        self.prepare_defense()

        for rounds in range(self.cf["nb_rounds"]):
            torch.cuda.empty_cache()

            #if rounds+1 % 100 == 0 :
            #    self.lr /= 10

            selected_clients = Utils.select_clients(self.clients, self.config_FL["nb_clients_per_round"])

            # Local model training                                        
            for client in tqdm(selected_clients):
                client.set_model(deepcopy(self.global_model).to(self.device))
                #match self.aggregation :
                #    case "FedAvg" :
                #        client.train(self.cf, lr = self.lr, strategy = "FedAvg")
                #    case "FedProx" : 
                #        client.train(self.cf, lr = self.lr, strategy = "FedProx", global_model = self.global_model)
                #    case "SCAFFOLD" : 
                #        client.train(self.cf, lr = self.lr, strategy = "SCAFFOLD", global_variate = self.global_variate)
                if self.aggregation == "FedAvg" :
                    client.train(self.cf, lr = self.lr, strategy = "FedAvg")
                elif self.aggregation == "FedProx" : 
                    client.train(self.cf, lr = self.lr, strategy = "FedProx", global_model = self.global_model)
                elif self.aggregation == "SCAFFOLD" : 
                    client.train(self.cf, lr = self.lr, strategy = "SCAFFOLD", global_variate = self.global_variate)
                
            
            # Detection step
            print("filtering and aggregating...")
            good_updates = self.filter_clients(selected_clients)
            print("filtered clients")

            # Aggregation step
            if len(good_updates) != 0 :
                self.global_model.load_state_dict(Utils.aggregate_models(good_updates))
            print("aggregated local updates")

            if self.aggregation == "SCAFFOLD" : 
                # Update global variate 
                for client in tqdm(selected_clients):
                    for g_variate, l_variate in zip(self.global_variate, client.control_variate): 
                        g_variate.data += l_variate.data/self.config_FL["num_clients"]

            # Clean up after update step (to save GPU memory)
            for client in selected_clients:
                client.remove_model()

            # Evaluate accuracy on test
            self.accuracy.append(test_acc := Utils.test(self.global_model, self.device, self.test_loader))
            if test_acc > self.best_accuracy :
                self.best_accuracy, self.best_round = test_acc, rounds+1
            
            # Print round recap
            print(f"Round {rounds + 1}/{self.cf['nb_rounds']} server test accuracy: {self.accuracy[-1] * 100:.2f}%")
            
            # Evaluate backdoor if attack is one 
            if Utils.is_backdoor(self.attack_type): 
                self.accuracy_backdoor.append(Utils.test_backdoor(self.global_model, self.device, self.test_loader,
                                                  self.attack_type, self.cf["source"],
                                                  self.cf["target"], self.cf["square_size"]))
                print(f"Round {rounds + 1}/{self.cf['nb_rounds']} backdoor accuracy: {self.accuracy_backdoor[-1] * 100:.2f}%")

            # Recap detection
            nb_attackers = np.array([client.is_attacker for client in selected_clients]).sum()
            nb_benign = np.array([not client.is_attacker for client in selected_clients]).sum()
            nb_attackers_passed = np.array([client.is_attacker for client in good_updates]).sum()
            nb_benign_passed = np.array([not client.is_attacker for client in good_updates]).sum()
            nb_benign_blocked = nb_benign-nb_benign_passed
            print("Total of Selected Clients :", len(selected_clients), "| Number of attackers :", nb_attackers,
                  "| Total of attackers passed defense :", nb_attackers_passed, "| Benigns blocked :", nb_benign_blocked)
            
            # Evaluate detection
            for client in good_updates : 
                client.unsuspect()
            round_suspects = np.array([int(client.is_suspect) for client in selected_clients])
            round_unsespected = np.array([int(not(client.is_suspect)) for client in selected_clients])
            round_attackers = np.array([int(client.is_attacker) for client in selected_clients])
            round_benign = np.array([int(not(client.is_attacker)) for client in selected_clients])
            TP = sum(round_suspects*round_attackers)
            FP = sum(round_suspects*round_benign)
            FN = sum(round_unsespected*round_attackers)
            round_attacker_recall = TP/(TP+FN+1e-10)
            round_attacker_precision = TP/(TP+FP+1e-10)
            benign_TP = FN = sum(round_unsespected*round_benign)
            benign_FN = FP
            round_benign_recall = benign_TP/(benign_TP+benign_FN+1e-10)
            for client in good_updates : 
                client.suspect()

            # Paint attacker and client profiles
            attacker_profile_this_round = []
            passing_attacker_profile_this_round = []
            benign_profile_this_round = []
            passing_benign_profile_this_round = []
            for client in selected_clients:
                if client.is_attacker :
                    attacker_profile_this_round.append(client.num_samples)
                else : 
                    benign_profile_this_round.append(client.num_samples)
            for client in good_updates:
                if client.is_attacker :
                    passing_attacker_profile_this_round.append(client.num_samples)
                else : 
                    passing_benign_profile_this_round.append(client.num_samples)

            # Update metric histories
            self.total_attackers_passed += nb_attackers_passed
            self.total_benign_blocked += nb_benign_blocked
            self.total_attackers += nb_attackers
            self.total_benign += nb_benign
            self.pred_labels.extend(round_suspects.tolist())
            self.true_labels.extend(round_attackers.tolist())
            self.nb_attackers_history.append(nb_attackers)
            self.nb_attackers_passed_defence_history.append(nb_attackers_passed)
            self.nb_benign_history.append(nb_benign)
            self.nb_benign_passed_defence_history.append(nb_benign_passed)
            self.attacker_recall_hist.append(round_attacker_recall)
            self.attacker_precision_hist.append(round_attacker_precision)
            self.benign_recall_hist.append(round_benign_recall)
            self.passing_attacker_profiles_per_round.append(passing_attacker_profile_this_round)
            self.attacker_profiles_per_round.append(attacker_profile_this_round)
            self.passing_benign_profiles_per_round.append(passing_benign_profile_this_round)
            self.benign_profiles_per_round.append(benign_profile_this_round)
            self.histo_selected_clients = torch.cat((self.histo_selected_clients,
                                         torch.tensor([client.id for client in good_updates])))
            
            # Log to tensorboard
            if log :
                writer.add_scalar("Accuracy/test", test_acc, rounds)
                writer.add_scalar("Detection/Undetected attackers", nb_attackers_passed, rounds)
                writer.add_scalar("Detection/Blocked benigns", nb_benign_blocked, rounds)
                writer.flush()
        
        print(f"finished running server in {datetime.datetime.now() - t_start}")

        # Print some stats 
        print(f"In total : Number of attackers : {self.total_attackers}, \
        \nTotal of attackers passed defense : {self.total_attackers_passed} ({(self.total_attackers_passed/self.total_attackers)*100:.2f}%) \
        \nTotal of benigns blocked : {self.total_benign_blocked} ({(self.total_benign_blocked/self.total_benign)*100:.2f}%)")         
        print(f"Best test accuracy : {self.best_accuracy*100:.2f}. Achieved in {self.best_round} rounds")

        # Save the training config 
        Utils.save_to_json(self.cf, self.dir_path, "run_config.json")

        # Save metrics in recap file 
        Utils.save_latex_ready_metrics([f"{(self.total_benign_blocked/self.total_benign)*100:.2f}%",
                                        f"{(self.total_attackers_passed/self.total_attackers)*100:.2f}%",
                                        f"{self.best_accuracy*100:.2f}"], self.dir_path, "latex_metrics")

        # Saving The accuracies of the Global model on the testing set and the backdoor set
        Utils.save_to_json(self.accuracy, self.dir_path, f"test_accuracy_{self.cf['nb_rounds']}")
        if Utils.is_backdoor(self.attack_type):                   
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path,f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")
        
        # Saving the percentage of attackers blocked
        Utils.save_to_json((self.total_attackers_passed/self.total_attackers)*100, self.dir_path, f"successful_attacks")
        Utils.save_to_json((self.total_benign_blocked/self.total_benign)*100, self.dir_path, f"blocked_benigns_blocked")

        # Detection metrics to JSON
        Utils.save_to_json(self.attacker_precision_hist, self.dir_path, f"attacker_detection_precision_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.attacker_recall_hist, self.dir_path, f"attacker_detection_recall_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.benign_recall_hist, self.dir_path, f"benign_detection_recall_{self.cf['nb_rounds']}")

        # Metrics for ROC curve to json 
        Utils.save_to_json(self.pred_labels, self.dir_path, f"ROC_pred_labels")
        Utils.save_to_json(self.true_labels, self.dir_path, f"ROC_real_labels")

        # Attacker profiles to json
        Utils.save_to_json(self.attacker_profiles_per_round, self.dir_path, f"attacker_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.passing_attacker_profiles_per_round, self.dir_path, f"passing_attacker_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.benign_profiles_per_round, self.dir_path, f"benign_profiles_{self.cf['nb_rounds']}")
        Utils.save_to_json(self.passing_benign_profiles_per_round, self.dir_path, f"passing_benign_profiles_{self.cf['nb_rounds']}")

        # Plot the accuracies of the Global model on the testing set 
        title_info = f"Test Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Test_Accuracy_{self.cf['nb_rounds']}.png"
        Utils.plot_accuracy(self.accuracy, x_info='Round', y_info='Test Accuracy', title_info=title_info, save_path=save_path)
        
        # Plot the backdoor accuracies on the testing set 
        if Utils.is_backdoor(self.attack_type):            
            title_info = f"Backdoor Accuracy per Round for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
            save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Backdoor_Accuracy_{self.cf['nb_rounds']}.png"
            Utils.plot_accuracy(self.accuracy_backdoor, x_info='Round', y_info='backdoor Accuracy', title_info=title_info, save_path=save_path)
        
        # Plot detection metrics
        title_info = f"Blocked attackers and benigns for {self.attacker_ratio * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_detection_{self.cf['nb_rounds']}.png"
        Utils.plot_recall(self.attacker_recall_hist, self.benign_recall_hist, self.nb_rounds, title_info = title_info,
                               save_path = save_path)

        # Plotting the histogram of the defense system
        Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
                             self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
                             self.attack_type, self.defence, self.dir_path, 
                             success_rate=f"{(1-(self.total_attackers_passed/self.total_attackers))*100:.2f}%",
                             attacker_ratio=self.attacker_ratio)
        
        # Plotting the historgram of selected clients 
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=100,
                                save_path=save_path)
        
        # TODO plot attacker profiles
