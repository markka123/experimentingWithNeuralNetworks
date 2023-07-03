#Implemented a basic neural network for classification from this site: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch


#Step 1 imports:
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

#Step 2 imports:
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

#Step 3 imports:
from torch import nn
from torch import optim

#Step 6 imports:
import itertools



# STEP 1: Create a dataset (This one has 10,000 samples):


X, y = make_circles(n_samples = 10000,
                    noise= 0.05,
                    random_state=26)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

# Visualize the data:

# fig, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
# train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
# train_ax.set_title("Training Data")
# train_ax.set_xlabel("Feature #0")
# train_ax.set_ylabel("Feature #1")

# test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
# test_ax.set_xlabel("Feature #0")
# test_ax.set_title("Testing data")
# plt.show()




#STEP 2: Convert the data to torch tensors:


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
# for batch, (X, y) in enumerate(train_dataloader):
#     print(f"Batch: {batch+1}")
#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#     break





#STEP 3: Implement the neural network (This one is a simple two-layer network that uses ReLU as the activation function)

input_dim = 2
hidden_dim = 10
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x
    
#Test:
model = NeuralNetwork(input_dim, hidden_dim, output_dim)
# print(model)





#STEP 4: Define a cost-function for practise:
learning_rate = 0.1

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)





#STEP 5: Train the network on test-data


num_epochs = 100  #Each epoch is an entire lap through the entire test-dataset
loss_values = [] #To track the loss-values (Evaluate if the network has improved through training)


for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()



#STEP 5.5: Visualize the loss values:

step = np.linspace(0, 100, 10500)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()



#STEP 6: Use model for predictions:


y_pred = [] 
correct = 0
total = 0

with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test = np.append(y_test, y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()

#Accuracy resultat på nettverket:
print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

