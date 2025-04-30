import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import argparse
import yaml
import os

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def inference(video_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, config['model']['num_classes'])
    model.load_state_dict(torch.load(os.path.join(config['train']['save_dir'], 'model.pth')))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor = transform(cv2.resize(frame, (224, 224))).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = "Violation" if pred == 1 else "Safe"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inference(args.video_path, config)
