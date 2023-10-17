import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        # Perform any additional preprocessing or transformations if needed
        return x, y



class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # Add more convolutional layers if needed
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(9 * 9 * 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # print(f"Conv: {x.shape}")
        x = x.view(x.size(0), -1) 
        # x = x.flatten()
        # print(f"Flatten: {x.shape}")
        x = self.fc_layers(x)
        return x

def load_data():
    path="../Datasets/Hex/dataset270k.txt"
    with open(path) as file:
        data=file.readlines()
    data=[f.strip().split(" ") for f in data]
    X=[list(f[0]) for f in data]
    X=[list(map(int, f)) for f in X]
    # X=[f[:0] for f in X]
    Y=[1 if f[1]=='w' else 0 for f in data ]
    tempData = list(zip(X, Y))
    random.shuffle(tempData)
    X, Y = zip(*tempData)
    print(len(X))
    # dataset = np.empty([len(X), 2,6,6])
    dataset = []
    for i, x in enumerate(X):
        first_layer_array = np.reshape(x[:36], (6, 6))
        second_layer_array = np.reshape(x[36:], (6, 6))
        dataset.append( np.stack((first_layer_array, second_layer_array)))
        # print(dataset[0].shape)
        # break
    return dataset, Y

def load_go_data():
    # go_data = pd.read_csv("/Users/charug18/Drive/Work/PhD/Projects/Go Winner Prediction/Go_binary_data_9x9.csv", delimiter=",")
    go_data = pd.read_csv("Winner Prediction/Go Winner Prediction/Go_binary_data_9x9.csv", delimiter=",")
    data = go_data.iloc[:,0]
    labels = go_data.iloc[:,1]
    go_board = []
    Y=[int(f) for f in labels]
    tempData = list(zip(data, Y))
    random.shuffle(tempData)
    X, Y = zip(*tempData)
    # go_string = []
    for x in X:
        black = [1 if bit == "1" else 0 for bit in x]
        white = [1 if bit == "2" else 0 for bit in x]
        black_array = np.reshape(black, (9, 9))
        white_array = np.reshape(white, (9, 9))
        go_board.append( np.stack((black_array, white_array)))
    return go_board, Y
    
        # go_string.append( black+white)  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

value_net = ValueNetwork()
print(f'The model has {count_parameters(value_net):,} trainable parameters.')
exit(0)
X, Y = load_go_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
Y_train = torch.unsqueeze(Y_train, 1)
Y_test = torch.unsqueeze(Y_test, 1)
print(X_train.shape, Y_train.shape)
# exit(0)
# Create an instance of your custom dataset
dataset_train = CustomDataset(X_train, Y_train)
dataset_test = CustomDataset(X_test, Y_test)

# dataset = CustomDataset(flattened_data)


# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Iterate over the data loader
# for batch in data_loader:
#     # Perform your training or evaluation logic here
#     # Each batch will contain `batch_size` flattened tensors
#     print(batch.shape)
# exit(0)

# Create an instance of the ValueNetwork
value_net = ValueNetwork()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(value_net.parameters(), lr=0.01, momentum=0.9)

# Create a data loader for the dataset
batch_size = 128
data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
device = torch.device("cpu")
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)
value_net = value_net.to(device)


# Define the number of epochs
num_epochs = 10
num_batches = X_test.shape[0]//batch_size
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_data, batch_labels in data_loader_train:

        # Forward pass
        output = value_net(batch_data)
        
        # Compute the loss
        # loss = criterion(output, batch_labels)
        loss = criterion(output, batch_labels)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the running loss
        running_loss += loss.item()
        
    # Print the average loss for each epoch
    epoch_loss = running_loss / len(X_train)
    print(f"Epoch {epoch+1} Loss: {epoch_loss}")
    
    # Evaluation
    with torch.no_grad():
        # Disable gradient computation
        accuracy = 0.0
        # Set the threshold value
        threshold = 0.5
        start_time = time.time()
        for batch_data, batch_labels in data_loader_test:
            
            # Forward pass
            output = value_net(batch_data)
            
            # Convert sigmoid output to binary labels
            binary_labels = torch.where(output > threshold, torch.tensor(1), torch.tensor(0))
            
            # Compute the accuracy
            accuracy+= accuracy_score(batch_labels, binary_labels)
            # loss = criterion(output, batch_labels)
            
        end_time = time.time()
        # Update the running acc
        accuracy = accuracy/num_batches

        print(f"Epoch {epoch+1}\tAcc: {accuracy:.2f}\tData: {X_test.shape[0]}\tInference Time: {end_time-start_time}")
