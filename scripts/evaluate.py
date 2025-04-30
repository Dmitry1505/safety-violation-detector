import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
import yaml
import os

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def evaluate(config):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_ds = ImageFolder(config['data']['val_dir'], transform=transform)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, config['model']['num_classes'])
    model.load_state_dict(torch.load(os.path.join(config['train']['save_dir'], 'model.pth')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            y_true.extend(targets.tolist())
            y_pred.extend(predictions)

    metrics = compute_metrics(y_true, y_pred)
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config)
