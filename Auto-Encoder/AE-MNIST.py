import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


""" 
This is the script version of Auto Encoder for MNIST dataset 
In this script, we aim at reconstruct the images of MNIST using AutoEncoder
"""

# Define HyperParameters
input_size = 28 * 28  # image total pixel
batch_size = 128
learning_rate = 1e-2
num_epochs = 10


# Define DataTransformation
transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# Load MNIST data
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(
    root="../data", train=True, transform=transform, download=False
)

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)


# Download and load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root="../data", train=False, transform=transform
)

# Create a data loader for the test dataset
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    device = torch.device("mps")
    print(f" Using Device: {device}")
    autoencoder = AutoEncoder().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in train_loader:
            images, _ = data
            images = images.view(images.size(0), -1).to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

    # Plot the Reconstructed images
    # Set the autoencoder model to evaluation mode
    autoencoder.eval()

    # Choose a batch of test images
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # Perform the forward pass through the autoencoder to obtain the reconstructed images
    with torch.no_grad():
        reconstructed_images = autoencoder(images.view(images.size(0), -1))

    # Move the reconstructed images back to the CPU (if using CUDA)
    reconstructed_images = reconstructed_images.cpu()

    # Plot Reconstructed VS Original

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    # Plot the first subplot
    axes[0].imshow(images[0].reshape((28, 28)).cpu())
    axes[0].set_title("Original MNIST")

    # Plot the second subplot
    axes[1].imshow(reconstructed_images[0].reshape((28, 28)))
    axes[1].set_title("Reconstructed")
    plt.show()
