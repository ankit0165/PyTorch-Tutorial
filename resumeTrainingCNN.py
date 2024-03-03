import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # print('input', x.shape)
        x = self.pool(self.relu(self.conv1(x)))
        # print('pool', x.shape)
        x = x.view(-1, 16 * 112 * 112)
        # print('view', x.shape)
        x = self.relu(self.fc1(x))
        # print('relu', x.shape)
        x = self.fc2(x)
        # print('output', x.shape)
        return x

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the path to the root folder containing your class folders
data_root = os.path.join(os.getcwd(), 'dataset',='train')

# Create a dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_root, transform=transform)

# Create a DataLoader
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Hyperparameters
num_epochs = 10
learning_rate = 0.001
num_classes = 2


# Create an instance of the model
model = CNNModel(num_classes)

# Load the saved state dictionary
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'dataset','models','cnn_model_weightsFinal.pth')))

# Initialize loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(inputs.shape, labels.shape, outputs.shape)
        # sys.exit()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'cnn_model_weights.pth')
    print(f'model saved for epoch {epoch+1} ...............')

# Save the model's state dictionary after the training
torch.save(model.state_dict(), 'cnn_model_weightsFinal.pth')
print(f'Final model saved ...................................')