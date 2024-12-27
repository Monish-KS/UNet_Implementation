import torch
import matplotlib.pyplot as plt
from model import UNET
from utils import get_loaders, save_predictions_as_imgs, check_accuracy
from albumentations.pytorch import ToTensorV2
import albumentations as A


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
NUM_WORKERS = 2
PIN_MEMORY = True
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"


test_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)



def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

def plot_sample_predictions(test_loader, model, device, num_samples=5):
    model.eval()
    for idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()

        images = images.cpu()
        preds = preds.cpu()
        masks = masks.cpu()

        for i in range(min(num_samples, images.shape[0])):
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(images[i].permute(1, 2, 0))
            ax[0].set_title("Input Image")
            ax[1].imshow(masks[i][0], cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[2].imshow(preds[i][0], cmap="gray")
            ax[2].set_title("Prediction")
            plt.show()
        break


def plot_metrics(metrics):
    plt.figure(figsize=(10, 5))

    if "loss" in metrics:
        plt.subplot(1, 2, 1)
        plt.plot(metrics["loss"], label="Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    if "accuracy" in metrics:
        plt.subplot(1, 2, 2)
        plt.plot(metrics["accuracy"], label="Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(CHECKPOINT_PATH, model)

    test_loader, _ = get_loaders(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        test_transform,
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    accuracy = check_accuracy(test_loader, model, device=DEVICE)
    print(f"Test Accuracy: {accuracy:.2f}%")

    plot_sample_predictions(test_loader, model, DEVICE)

    metrics = {
        "loss": [0.6, 0.5, 0.4, 0.3],  
        "accuracy": [70, 75, 80, 85],  
    }
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
