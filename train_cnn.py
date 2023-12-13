# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchsummary import summary

import matplotlib.pyplot as plt
import random

import time 
import calendar
import os
import argparse
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def get_timeString():

    time_stamp = time.time() 
    struct_time = time.localtime(time_stamp)
    timeString = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)

    return timeString

def simple_acc(number, decimal, complement):

    number = round(number, decimal)
    number = str(number)
    len_number = len(number)
    complement = complement - len_number
    while complement != 0:
        number += '0'
        complement -= 1

    return number

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained RL model on MNIST dataset.')
    parser.add_argument('--dataset', dest='dataset')
    parser.add_argument('--random_seed', default=None, dest='random_seed')
    parameter_args = parser.parse_args()

    dataset_name = parameter_args.dataset
    if type(parameter_args.random_seed) == str:
        random_seed = int(parameter_args.random_seed)
    elif parameter_args.random_seed == None:
        random_seed = None

    folder_path = f'../..'

    # %%
    # Datasets
    # dataset_name = 'Mnist'
    # random_seed = 1
    if random_seed == None: random_seed = np.random.randint(np.iinfo(np.int32).max // 2)

    epochs = 300
    batch_size = 128
    num_workers = 4
    set_seed(random_seed)

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == 'Fashion':
        train_data = datasets.FashionMNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data = datasets.FashionMNIST(root='data/', train=False, download=True, transform=data_transform)
        valid_size, learning_rate = 0.1, 0.005
    elif dataset_name == 'Mnist':
        train_data =  datasets.MNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.MNIST(root='data/', train=False, download=True, transform=data_transform)
        valid_size, learning_rate = 1/6, 0.05
    elif dataset_name == 'Cifar10':
        train_data =  datasets.CIFAR10(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.CIFAR10(root='data/', train=False, download=True, transform=data_transform)
        valid_size, learning_rate = 0.1, 0.005

    ######### Build valid dataset #########

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    ######### Build valid dataset #########

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    timeString = get_timeString()
    device = torch.device('cuda:0')
    
    record_path = f'{folder_path}/records/CNN/{dataset_name}'
    model_path = f'{folder_path}/models/CNN/{timeString}/{dataset_name}'
    best_model  = f'{model_path}/best.pt'
    if not os.path.isdir(record_path):
        os.makedirs(record_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    
    record_txt = f'{record_path}/record.txt'

    # %%
    # Models
    channel = 3 if dataset_name == 'Cifar10' else 1
    fully_channel = (8*7*7) if dataset_name == 'Cifar10' else (8*6*6)

    class CNN(nn.Module):

        def __init__(self):
            super().__init__()


            self.embedding_trainning = False
            self.network = nn.Sequential(
                nn.Conv2d(channel, 8, kernel_size=10, stride=5, padding=5),
                nn.ReLU(),
                nn.Flatten(1)
                )
            self.layer1 = nn.Linear(fully_channel, 256)
            self.layer2 = nn.Linear(256, 10)
            
        def forward(self, xb):
            
            xb = self.network(xb)
            xb  = self.layer1(xb) 
            xb  = F.relu(xb) 
            xb  = self.layer2(xb) 

            return xb
        
    class DNN(nn.Module):

        def __init__(self):
            super().__init__()

            self.network = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(3600, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
                )
            
        def forward(self, xb):
            xb = self.network(xb)
            return xb
        

    # %%

    model = CNN().to(device)

    optimizer_mlp = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    valid_acc_list = []
    best_acc = 0 

    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            ########### shift test ###########
    
            batsh_s, channels, width, height = inputs.shape
            shift_unit = 1
            s_u = shift_unit * 2
            copy_tensor = torch.zeros((batsh_s, channels, width+s_u, height+s_u)).to('cuda')
    
            init_pos = 1
            ver_pos = shift_unit + random.randint(-1, 1)
            her_pos = shift_unit + random.randint(-1, 1)
    
            copy_tensor[:, :, ver_pos: ver_pos+width, her_pos: her_pos+height] = inputs
            inputs = copy_tensor[:, :, init_pos: init_pos+width, init_pos: init_pos+height]
            
            ########### shift test ###########
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()

            running_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_acc = round(correct / total, 4)
        train_acc = 100 * train_acc
        train_acc_list.append(train_acc)
            
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in validloader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_acc = round(correct / total, 4)
        valid_acc = 100 * valid_acc
        valid_acc_list.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model, best_model)

    train_max_index = train_acc_list.index(max(train_acc_list))
    valid_max_index = valid_acc_list.index(max(valid_acc_list))

    max_trian_acc = max(train_acc_list)
    max_valid_acc = max(valid_acc_list)
    max_trian_acc = simple_acc(max_trian_acc , 3, 5)
    max_valid_acc = simple_acc(max_valid_acc , 3, 5)

    model = torch.load(best_model)

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_acc = round(correct / total, 4)
    valid_acc = 100 * valid_acc
    
    info =  f"Time: {timeString} | train acc: {max_trian_acc} at {train_max_index} | valid acc: {max_valid_acc} at {valid_max_index} | test acc: {valid_acc} | random seed: {random_seed}"
    print(info)

    with open(record_txt, "a+") as outfile:
        outfile.writelines(info)
        outfile.writelines('\n')

    # %%



