import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

# Load the MNIST dataset
data = DataLoader(
    datasets.MNIST(
        root='./data',  # Specify where to save the dataset
        download=True, 
        train=True, 
        transform=transform
    ), 
    batch_size=32,  # You can specify batch size here
    shuffle=True  # Shuffle the data
)



