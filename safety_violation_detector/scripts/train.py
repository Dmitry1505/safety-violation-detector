import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from torchvision import models
from utils.metrics import compute_metrics

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def train(config):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_ds = ImageFolder(config['data']['train_dir'], transform=transform)
    val_ds = ImageFolder(config['data']['val_dir'], transform=transform)

    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])

    model = models.resnet18(pretrained=config['model']['pretrained'])
    model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    for epoch in range(config['train']['epochs']):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['train']['epochs']} completed")

    os.makedirs(config['train']['save_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'model.pth'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
