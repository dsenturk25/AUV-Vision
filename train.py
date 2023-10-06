# from preprocess.preprocess import *
import os
from data_setup import create_dataloaders, investigate_data
from torchvision import datasets, transforms
import utils, engine, model_builder
import torch

BATCH_SIZE = 32
NUM_WORKERS = 0
HIDDEN_UNITS = 16
LEARNING_RATE = 0.001
EPOCHS = 25
SEED = 42

# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=TRAIN_DATA_PATH,
        test_dir=TEST_DATA_PATH,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,  # type: ignore
    )

    model = model_builder.OceanGate(
        input_channels=3, hidden_units=HIDDEN_UNITS, output_channels=len(class_names)
    ).to(device)

    model.load_state_dict(torch.load(f="models/OceanGateV0.pth"))

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=EPOCHS,
        optimizer=optimizer,  # type: ignore
        loss_fn=loss_fn,
        accuracy_fn=utils.accuracy_fn,
        device=device,  # type: ignore
    )

    utils.save_model(model=model, target_dir="models", model_name="OceanGateV0.pth")


if __name__ == "__main__":
    main()
