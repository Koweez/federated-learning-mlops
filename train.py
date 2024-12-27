import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Iterable
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
dataset = datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)

train_size = 600
eval_size = 100
rest = len(dataset) - train_size - train_size - eval_size

train_dataset1, train_dataset2, eval_dataset, _ = random_split(dataset, [train_size, train_size, eval_size, rest])

train_loader1 = DataLoader(train_dataset1, batch_size=50, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=50, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=50, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  
        x = self.pool(self.relu(self.conv3(x)))  
        x = x.view(-1, 128 * 4 * 4)              
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def average_model_parameters(models: Iterable[nn.Module], average_weights: list[float] = [1/2, 1/2]):
    model_params = [model.state_dict() for model in models]
    new_params = OrderedDict()
    for key in model_params[0].keys():
        new_params[key] = sum([model_params[i][key] * average_weights[i] for i in range(len(model_params))])
    return new_params

def update_model(model: nn.Module, new_params: OrderedDict):
    model.load_state_dict(new_params)
    
def client_update(model: nn.Module, train_loader: DataLoader, epochs:int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def federated_averaging(rounds: int, train_loaders: list[DataLoader], eval_loader: DataLoader,
                        global_model: nn.Module, epochs: int, lr: float = 0.001):
    global_model.to(device)
    global_weights = global_model.state_dict()
    client_models = [CNN().to(device) for _ in train_loaders]
    
    global_accuaracies = []
    client_accuracies = [[] for _ in train_loaders]
    
    for r in range(rounds):
        print(f'Round {r+1}/{rounds}')
        
        client_weights = []
        for i, train_loader in enumerate(train_loaders):
            client_models[i].load_state_dict(global_weights)
            updated_weights = client_update(client_models[i], train_loader, epochs, lr)
            client_weights.append(updated_weights)
            
        average_weights = [1/len(train_loaders)] * len(train_loaders)
        global_weights = average_model_parameters(client_models, average_weights)
        update_model(global_model, global_weights)
        
        global_accuracy = evaluate(global_model, eval_loader)
        global_accuaracies.append(global_accuracy)
        
        for i, client_model in enumerate(client_models):
            client_accuracy = evaluate(client_model, eval_loader)
            client_accuracies[i].append(client_accuracy)
            
    return global_accuaracies, client_accuracies
            
        
def evaluate(model: nn.Module, eval_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

batch_sizes = [50, 20, 10]

for batch_size in batch_sizes:
    print(f'Batch Size: {batch_size}')
    train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    
    global_model = CNN().to(device)
    global_accuracies, client_accuracies = federated_averaging(10, [train_loader1, train_loader2], eval_loader, global_model, 20, 0.001)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(global_accuracies, label='Global Model Accuracy', marker='o')
    for i, client_acc in enumerate(client_accuracies):
        plt.plot(client_acc, label=f'Client {i+1} Accuracy', marker='o')
    plt.title(f'Federated Averaging Performance (Batch Size: {batch_size})')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.show()
