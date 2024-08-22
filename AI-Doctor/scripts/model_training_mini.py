import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load pre-trained MobileNetV2 model with weights
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1)
)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

# Load datasets and create small subsets
train_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/train', 
    transform=data_transforms['train']
)
val_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/test', 
    transform=data_transforms['val']
)

# Create smaller subsets for faster training
train_indices = list(range(32))  # Adjust as needed
val_indices = list(range(32))  # Adjust as needed
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

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

            if i % 10 == 0:
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
