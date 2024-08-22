import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# Define model (use the same model architecture as in training)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1)
)

# Load the trained model
model.load_state_dict(torch.load('pimple_detection_model.pth'))
model.eval()

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            preds = (probs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            # Break after processing a limited number of batches
            # To achieve fast evaluation, break after a small number of batches
            if total_samples >= 1000:  # Adjust this number based on your needs
                break

    accuracy = running_corrects.double() / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')

# Load validation dataset and DataLoader
val_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/test', 
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

# Reduce the size of validation data to speed up evaluation
# You can use a subset of the validation dataset for faster evaluation
val_subset_indices = list(range(0, 50))  # Adjust the range based on your needs
val_subset = Subset(val_dataset, val_subset_indices)

val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4)  # Batch size=8

# Evaluate the model
evaluate_model(model, val_loader)
