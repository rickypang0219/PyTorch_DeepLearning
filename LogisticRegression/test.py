import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define a custom dataset for training
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


# Define the training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# Generate some dummy data for training
features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
labels = [0, 0, 1, 1]

# Create a DataLoader for training
dataset = CustomDataset(features, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Set the input size and number of classes
input_size = 2
num_classes = 2

# Create an instance of the logistic regression model
model = LogisticRegression(input_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=100)

# Test the model on new data
test_features = [[2.5, 3.5], [1.5, 2.5]]
test_labels = [0, 0]
test_dataset = CustomDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

accuracy = correct / total
print("Accuracy: {:.2f}%".format(accuracy * 100))

for i in model.parameters():
    print(i)
