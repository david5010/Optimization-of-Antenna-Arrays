from models.FF_1H import *
from datasets.AntDataset import AntDataset, AntDataset2D
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import KFold

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(train_loader)

def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            total_loss += loss.item()

    return total_loss/len(test_loader)

def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, num_epochs, device):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        test_loss = evaluate(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def cross_validate(fold, model, optimizer, criterion, epochs, data, batch_size, device):

    kf = KFold(n_splits = fold, shuffle = True)
    training_avg_loss = []
    testing_avg_loss = []
    for k, (train, test) in enumerate(kf.split(np.arange(len(data)))):
        print(f'{k}-Fold')
        train_loader = DataLoader(AntDataset(data[train]), batch_size=batch_size)
        test_loader = DataLoader(AntDataset(data[test]), batch_size=batch_size)
        new_fold = model
        new_fold.to(device)
        train_loss, test_loss = train_and_evaluate(new_fold, optimizer, criterion, train_loader, test_loader, epochs, device)
        training_avg_loss.append(train_loss)
        testing_avg_loss.append(test_loss)
        model.apply(weight_reset)
        new_fold.apply(weight_reset)

    return training_avg_loss, testing_avg_loss

