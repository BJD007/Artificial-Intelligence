import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets

# Data transformations for training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),    # Resize images to 224x224, suitable for ResNet
        transforms.RandomHorizontalFlip(), # Data augmentation technique
        transforms.ToTensor(),             # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using ImageNet statistics
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),     # Resize images to 224x224
        transforms.ToTensor(),             # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize using ImageNet statistics
    ]),
}

# Load pre-trained ResNet18 model with weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers except the last few
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1)  # No Sigmoid here
)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Define a learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Load datasets
train_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/train', 
    transform=data_transforms['train']
)

val_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/test', 
    transform=data_transforms['val']
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 0:  # Log every 10 batches
                print(f'Batch {i+1}/{len(train_loader)}, Time elapsed: {time.time() - start_time:.2f}s')

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {time.time() - start_time:.2f}s')

        scheduler.step(epoch_loss)

    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=1)

# Save the trained model
torch.save(model.state_dict(), 'pimple_detection_model.pth')

print("Model trained and saved.")
