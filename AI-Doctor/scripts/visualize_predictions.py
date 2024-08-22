import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define and load the model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1)
)

# Load the trained model
model.load_state_dict(torch.load('pimple_detection_model.pth'))
model.eval()
model.to(device)

# Define the data loader
val_dataset = datasets.ImageFolder(
    '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/test', 
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define visualization function
def visualize_predictions(model, data_loader):
    model.eval()

    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)  # Convert logits to probabilities
        preds = (probs > 0.5).float()

        # Convert to numpy arrays for visualization
        inputs = inputs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        # Plot images with predictions
        num_images = min(8, len(inputs))  # Ensure we don't exceed available images
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        axes = axes.flatten()
        for i in range(num_images):
            img = np.transpose(inputs[i], (1, 2, 0))
            img = np.clip(img, 0, 1)  # Ensure pixel values are in [0, 1]
            axes[i].imshow(img)
            axes[i].set_title(f'Pred: {int(preds[i][0])} | True: {int(labels[i][0])}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

# Visualize model predictions
visualize_predictions(model, val_loader)

