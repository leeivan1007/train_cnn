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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained RL model on MNIST dataset.')
    parser.add_argument('--dataset', dest='dataset')
    parameter_args = parser.parse_args()

    dataset_name = parameter_args.dataset # cat_dog  mnist

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
    # Datasets
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize([60, 60]),
    ])

    dataset_name = 'Mnist'

    if dataset_name == 'Fashion':
        train_data = datasets.FashionMNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data = datasets.FashionMNIST(root='data/', train=False, download=True, transform=data_transform)
    elif dataset_name == 'Mnist':
        train_data =  datasets.MNIST(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.MNIST(root='data/', train=False, download=True, transform=data_transform)
    elif dataset_name == 'Cifar10':
        train_data =  datasets.CIFAR10(root='data/', train=True, download=True, transform=data_transform)
        test_data =  datasets.CIFAR10(root='data/', train=False, download=True, transform=data_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)

    device = torch.device('cuda:0')
    record_path = 'records/CNN'
    if not os.path.isdir(record_path):
        os.makedirs(record_path)

    record_txt = f'{record_path}/{get_timeString()}.txt'

    with open(record_txt, "a+") as outfile:
        outfile.writelines(f'Dataset: {dataset_name}')
        outfile.writelines('\n')


    # %%
    # Models
    class CNN(nn.Module):

        def __init__(self):
            super().__init__()

            self.embedding_trainning = False
            self.network = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=10, stride=5, padding=5),
                nn.ReLU(),
                nn.Flatten(1)
                )
            self.layer1 = nn.Linear(8 * 6 * 6, 256)
            self.layer2 = nn.Linear(256, 10)
            
        def forward(self, xb):
            
            # print(xb.shape)
            xb = self.network(xb)
            # print(xb.shape)
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
    epochs = 400
    train_times = 100

    learning_rate = [
        0.045,
        0.05,
        0.055,
    ]


    # %%
    for t in range(train_times):

        for lr in learning_rate:

            model = CNN().to(device)

            optimizer_mlp = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
            print(f'T: {t+1} | lr: {lr} | Best train acc: {max_trian_acc} at {train_max_index} | Best test acc: {max_test_acc} at {test_max_index}')
            info =  f"Time: {get_timeString()} | lr: {lr} | Best train acc: {max_trian_acc} at {train_max_index} | Best test acc: {max_test_acc} at {test_max_index}"
            
            with open(record_txt, "a+") as outfile:
                outfile.writelines(info)
                outfile.writelines('\n')



