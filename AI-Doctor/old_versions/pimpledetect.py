import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the device to be used for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define paths to the training and testing data directories
train_data_dir = '/path/to/ISIC_2020_Training'   # Update this path
test_data_dir = '/path/to/ISIC_2020_Testing'     # Update this path

# Define constants for image dimensions, batch size, and learning rate
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Transformations for the training data, including data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_HEIGHT),    # Randomly resize and crop images to 224x224
    transforms.RandomHorizontalFlip(),           # Randomly flip images horizontally
    transforms.RandomRotation(30),               # Randomly rotate images by up to 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Adjust color
    transforms.ToTensor(),                       # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize images
])

# Transformations for the validation/testing data (no data augmentation, only normalization)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # Resize images to 224x224
    transforms.ToTensor(),                       # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize images
])

# Create DataLoader for the training set
train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Create DataLoader for the testing/validation set
val_dataset = datasets.ImageFolder(test_data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

def load_pretrained_model():
    """
    Load ResNet18 pre-trained on ImageNet, replace the final layer for binary classification.
    """
    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer with a new one for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),        # Add a fully connected layer with 128 neurons
        nn.ReLU(),                           # ReLU activation
        nn.Dropout(0.5),                     # Dropout layer for regularization
        nn.Linear(128, 1),                   # Final layer with 1 neuron for binary output
        nn.Sigmoid()                         # Sigmoid activation for binary classification
    )

    return model

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    """
    Train the model with data loaders, loss function, optimizer, and learning rate scheduler.
    """
    # Track training and validation metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Set the model to training mode
    model.train()

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        # Iterate over batches from the train_loader
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss and correct predictions count
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()  # Threshold the outputs for binary prediction
            running_corrects += torch.sum(preds == labels.data)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Append epoch loss and accuracy to tracking lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Evaluate on the validation set
        val_loss, val_acc = evaluate_model(model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'pimple_detection_model.pth')

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, criterion):
    """
    Evaluate the model on the validation dataset and calculate performance metrics.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize metrics
    running_loss = 0.0
    running_corrects = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update running loss and correct predictions count
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

    # Calculate average loss and accuracy for the validation set
    val_loss = running_loss / len(val_dataset)
    val_acc = running_corrects.double() / len(val_dataset)

    return val_loss, val_acc.item()

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Plot training and validation accuracy and loss over epochs.
    """
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def visualize_predictions(model, image_path):
    """
    Visualize the model's predictions on a sample image.
    """
    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = torchvision.io.read_image(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Predict using the model
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    # Display the image with prediction confidence
    image_np = image.cpu().numpy().squeeze().transpose((1, 2, 0))
    image_np = np.clip(image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    if prediction > 0.5:
        plt.title(f'Pimple Detected: Confidence {prediction:.2f}')
    else:
        plt.title(f'No Pimple Detected: Confidence {prediction:.2f}')
    plt.axis('off')
    plt.show()

def main():
    # Load the pre-trained model
    model = load_pretrained_model().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # Adam optimizer

    # Define learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the model and obtain training/validation history
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, criterion, optimizer, scheduler, num_epochs=20
    )

    # Plot training and validation history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)

    # Visualize predictions on a sample image
    sample_image_path = '/path/to/sample_image.jpg'  # Update this path
    visualize_predictions(model, sample_image_path)

if __name__ == '__main__':
    main()
