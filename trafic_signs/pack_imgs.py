import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

img_dir = Path("trafic_signs/data/fid_outputs/epoch_5/generated")
img_paths = sorted(list(img_dir.glob("*.png")))[-1000:]

to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

tensor_list = []
for path in img_paths:
    img = Image.open(path).convert("RGB")
    tensor = to_tensor(img)
    tensor_list.append(tensor)

final_tensor = torch.stack(tensor_list)

torch.save(final_tensor.cpu().detach(), "trafic_signs/data/piatek_OstaszewskiMateusz_SadowskiMichal.pt")
