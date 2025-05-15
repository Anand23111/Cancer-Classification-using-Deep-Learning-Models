import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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

# Define the CNN model
class PCamCNN(nn.Module):
    def __init__(self):
        super(PCamCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCamCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the model
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        # Ensure the labels are squeezed to the correct shape
        labels = labels.squeeze()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Ensure the labels are squeezed to the correct shape
            labels = labels.squeeze()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'pcam_model.pth')

# Evaluate the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        # Ensure the labels are squeezed to the correct shape
        labels = labels.squeeze()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        test_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Loss: {test_loss/len(test_loader)}')
print(f'Test Accuracy: {100 * correct / total}%')
