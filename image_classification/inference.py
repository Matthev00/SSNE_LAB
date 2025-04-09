import os

import torch
from PIL import Image
from torchvision import transforms

from models import TinyResNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TinyResNet(input_channels=3, hidden_size=128, dropout=0, num_classes=50)
model.load_state_dict(torch.load("models/best.pth"))
model = model.to(device)

model.eval()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5204, 0.4950, 0.4381], std=[0.2113, 0.2103, 0.2100]
        ),
    ]
)

test_dir = "data/test_all"
results = []

for filename in os.listdir(test_dir):
    if filename.endswith(".JPEG"):
        img_path = os.path.join(test_dir, filename)

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(img)
            predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)

        results.append((filename, predicted.item()))


with open("preds.csv", "w") as f:
    for filename, predicted in results:
        f.write(f"{filename}, {predicted}\n")
