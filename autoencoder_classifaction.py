import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Custom Dataset class to load HDF5 data
class PCamDataset(Dataset):
    def __init__(self, h5_file, label_file, transform=None):
        self.image_file = h5_file
        self.label_file = label_file
        self.transform = transform
        with h5py.File(h5_file, 'r') as f:
            self.data_len = len(f['x'])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        with h5py.File(self.image_file, 'r') as f:
            x = f['x'][idx]
        with h5py.File(self.label_file, 'r') as f:
            y = f['y'][idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = PCamDataset('/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_train_x.h5',
                            '/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_train_y.h5', transform=transform)
val_dataset = PCamDataset('/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_valid_x.h5',
                          '/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_valid_y.h5', transform=transform)
test_dataset = PCamDataset('/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_test_x.h5',
                           '/raid/home/vibhor/Anand111/capstone/PCam/Pcam/camelyonpatch_level_2_split_test_y.h5', transform=transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the model
num_epochs = 20

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, _ in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Save the model
torch.save(model.state_dict(), 'autoencoder_model.pth')

# Evaluate the model
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/len(test_loader)}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
