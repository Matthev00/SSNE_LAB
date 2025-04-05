import torch
from models import (
    LeNetPlus,
    TinyVGG,
    CNN_BN,
    TinyResNet,
)

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_models(input_shape=(32, 3, 64, 64), num_classes=50, hidden_size=32):
    models = {
        "LeNetPlus": LeNetPlus(hidden_size=hidden_size, num_classes=num_classes),
        "TinyVGG": TinyVGG(hidden_size=hidden_size, num_classes=num_classes),
        "CNN_BN": CNN_BN(hidden_size=hidden_size, num_classes=num_classes),
        "TinyResNet": TinyResNet(hidden_size=hidden_size, num_classes=num_classes),
    }

    for name, model in models.items():
        print(f"\n🧠 Model: {name}")
        total, trainable = count_params(model)
        print(f"🔢 Total params: {total:,}")
        print(f"🟢 Trainable params: {trainable:,}")
        x = torch.randn(*input_shape)
        y = model(x)
        print(f"✅ Output shape: {y.shape}")


if __name__ == "__main__":
    test_models()