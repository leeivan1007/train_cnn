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


    # %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained RL model on MNIST dataset.')
    parser.add_argument('--dataset', dest='dataset')
    parser.add_argument('--epochs', dest='epochs')
    parser.add_argument('--train_times', dest='train_times')
    parser.add_argument('--learning_rate', dest='learning_rate')
    parser.add_argument('--valid_size', dest='valid_size')
    parameter_args = parser.parse_args()

    dataset_name = parameter_args.dataset
    epochs = int(parameter_args.epochs)
    train_times = int(parameter_args.train_times)
    learning_rate = float(parameter_args.learning_rate)
    valid_size = float(parameter_args.valid_size)

    # %%
    # Datasets


    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize([60, 60]),
    ])

    # dataset_name = 'Fashion'
    # epochs = 50
    # train_times = 100
    # learning_rate = 0.05 # Mnist 0.05 Fashion 0.005 Cifar 0.005
    # valid_size = 1/6
    
    random_seed = 3
    batch_size = 128
    num_workers = 4

    if dataset_name == 'Fashion':
        train_data = datasets.FashionMNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data = datasets.FashionMNIST(root='data/', train=False, download=True, transform=data_transform)
    elif dataset_name == 'Mnist':
        train_data =  datasets.MNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.MNIST(root='data/', train=False, download=True, transform=data_transform)
    elif dataset_name == 'Cifar10':
        train_data =  datasets.CIFAR10(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.CIFAR10(root='data/', train=False, download=True, transform=data_transform)

    ######### Build valid dataset #########

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    ######### Build valid dataset #########


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    # trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda:0')
    record_path = f'records/CNN/{dataset_name}'
    if not os.path.isdir(record_path):
        os.mkdir(record_path)

    record_txt = f'{record_path}/{get_timeString()}.txt'

    with open(record_txt, "a+") as outfile:
        outfile.writelines(f'Dataset: {dataset_name}')
        outfile.writelines('\n')


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
    for t in range(train_times):

        model = CNN().to(device)

        optimizer_mlp = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        train_acc_list = []
        test_acc_list = []

        for epoch in range(epochs):
            
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

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
                for data in testloader:
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_acc = round(correct / total, 4)
            test_acc = 100 * test_acc
            test_acc_list.append(test_acc)

            # print(f'Train Acc: {train_acc} | Test Acc: {test_acc}')

        train_max_index = train_acc_list.index(max(train_acc_list))
        test_max_index = test_acc_list.index(max(test_acc_list))

        max_trian_acc = max(train_acc_list)
        max_test_acc = max(test_acc_list)
        max_trian_acc = simple_acc(max_trian_acc , 3, 5)
        max_test_acc = simple_acc(max_test_acc , 3, 5)

        # param = sum([param.nelement() for param in model.parameters()])
        print(f'T: {t+1} | lr: {learning_rate} | Best train acc: {max_trian_acc} at {train_max_index} | Best test acc: {max_test_acc} at {test_max_index}')
        info =  f"T: {get_timeString()} | lr: {learning_rate} | Best train acc: {max_trian_acc} at {train_max_index} | Best test acc: {max_test_acc} at {test_max_index}"
        
        with open(record_txt, "a+") as outfile:
            outfile.writelines(info)
            outfile.writelines('\n')

    # %%


    # %%



