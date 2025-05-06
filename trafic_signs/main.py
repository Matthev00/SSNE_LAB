from pathlib import Path

from utils import set_seeds, weights_init

from data import get_dataloaders


def main():
    """
    Main function to run the script.
    """
    set_seeds()

    DATA_DIR = Path("trafic_signs/data/trafic_32")
    BATCH_SIZE = 64
    VAL_SIZE = 0.2

    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, VAL_SIZE)

    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        break


if __name__ == "__main__":
    main()
