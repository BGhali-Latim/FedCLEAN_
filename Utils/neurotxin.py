import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np

def neurotoxin(train_data, model, local_model_param, optimizer, device):

    ###---------------param--------------------###
    params = {}
    params['gradmask_ratio'] = 0.95
    params['retrain_poison'] = 20
    params['poison_label_swap'] = 9
    ###---------------param--------------------###
    weight_accumulator = dict()
    for name in local_model_param.keys():
        weight_accumulator[name] = torch.zeros_like(local_model_param[name])


    model.load_state_dict(local_model_param)
    poisoned_train_data = poison_dataset(train_data.dataset)
    model.train()

    # print('P o i s o n - n o w ! ----------')

    # get gradient mask use global model and clearn data
    if params['gradmask_ratio'] != 1 :

        # Get the mask of bottom-k%
        mask_grad_list = grad_mask_cv(model, train_data, device, ratio=params['gradmask_ratio'])
    else:
        mask_grad_list = None

    # Attack for 10 rounds
    for internal_epoch in range(params['retrain_poison']):
        
        param = train_cv_poison(params, model, optimizer, mask_grad_list, device,
                                poisoned_train_data, train_data)
        model.load_state_dict(param)

    return params, model.state_dict()

def train_cv_poison(params, model, poison_optimizer, mask_grad_list, device,
                    poisoned_train_data, benign_train_data):

    #Change labels to generate adversarial samples and train
    benign_train_data = torch.utils.data.DataLoader(benign_train_data.dataset, batch_size=128, shuffle=True)
    for (x1, x2) in zip(poisoned_train_data, benign_train_data):
        inputs_p, labels_p = x1
        inputs_c, labels_c = x2
        inputs = torch.cat((inputs_p,inputs_c))

        # Change clean sample's 7 label to 9
        for pos in range(labels_c.size(0)):
            if labels_c[pos] == 7:
                labels_c[pos] = params['poison_label_swap']

        # All poison data labels changed to 9
        for pos in range(labels_p.size(0)):
            labels_p[pos] = params['poison_label_swap']

        labels = torch.cat((labels_p,labels_c))

        inputs, labels = inputs.to(device), labels.to(device)
        poison_optimizer.zero_grad()

        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward(retain_graph=True)

        # The attacker only retains the gradient upload server with the bottom-k% position
        if params['gradmask_ratio'] != 1:
            #model = apply_grad_mask(model, mask_grad_list, device)
            mask_grad_list_copy = iter(mask_grad_list)
            for name, parms in model.named_parameters():
                if (parms.requires_grad) and (parms.grad!=None):
                    parms.grad = parms.grad * next(mask_grad_list_copy).to(device)
        poison_optimizer.step()

    return model.state_dict()

def grad_mask_cv(model, dataset_clearn, device, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""
    model.train()
    model.zero_grad()

    #train_data = torch.utils.data.DataLoader(dataset_clearn, batch_size=128, shuffle=True)
    train_data = dataset_clearn
    for batch_id, batch in enumerate(train_data):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward(retain_graph=True)

    mask_grad_list = []
    
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if (parms.requires_grad) and (parms.grad!=None):
            grad_list.append(parms.grad.abs().view(-1))

            # Sum the gradients
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            k_layer += 1

    grad_list = torch.cat(grad_list)
    # Take the smallest k percentiles of each group of gradients 
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    mask_flat_all_layer = torch.zeros(len(grad_list))
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    # mask_grad_list is a 0,1 matrix, marking the k percentile positions of the smallest gradients in each group
    for _, parms in model.named_parameters():
        if (parms.requires_grad) and (parms.grad!=None):

            gradients_length = len(parms.grad.abs().view(-1))

            mask_flat = mask_flat_all_layer[count:count + gradients_length ]
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()))

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()
    return mask_grad_list

def test_poison_cv(params, data_source, model, device):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    num_data = 0.0
    poisoned_test_data = poison_dataset(data_source)
    for batch_id, batch in enumerate(poisoned_test_data):

        for pos in range(len(batch[0])):
            batch[1][pos] = params['poison_label_swap']

        data, target = batch
        data = data.to(device)
        target = target.to(device)
        data.requires_grad_(False)
        target.requires_grad_(False)

        output = model(data)
        total_loss += nn.functional.cross_entropy(output, target,
                                          reduction='sum').data.item()  # sum up batch loss
        num_data += target.size(0)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    acc = 100.0 * (float(correct) / float(num_data))

    return acc

###############################################################################################
###############################################################################################
###############################################################################################
def poison_dataset(data):
    size_of_secret_dataset = 500
    batch_size = 128
    # base_case
    indices = list()

    ### Base case sample attackers training and testing data
    range_no_id = sample_poison_data(5, data)
    

    while len(indices) < size_of_secret_dataset:
        # print(len(indices), ' / ', size_of_secret_dataset)
        range_iter = random.sample(range_no_id,
                                   np.min([batch_size, len(range_no_id) ]))
        indices.extend(range_iter)

    poison_images_ind = indices
    ### self.poison_images_ind_t = list(set(range_no_id) - set(indices))

    return torch.utils.data.DataLoader(data,
                       batch_size=batch_size,
                       sampler=torch.utils.data.sampler.SubsetRandomSampler(poison_images_ind))

def sample_poison_data(target_class, test_dataset):
    cifar_poison_classes_ind = []
    for ind, x in enumerate(test_dataset):
        img, label = x

        if label == target_class:
            cifar_poison_classes_ind.append(ind)

    return cifar_poison_classes_ind
###############################################################################################
###############################################################################################
###############################################################################################