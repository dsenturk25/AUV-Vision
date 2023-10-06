import torch
import torchvision
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from PIL import Image


def pred_and_plot_on_custom_data(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: torchvision.transforms,  # type: ignore
    class_names: List[str] = None,  # type: ignore
    device: str = None,  # type: ignore
):
    custom_image = Image.fromarray(image)

    custom_image_transformed = transform(custom_image)  # type: ignore

    custom_image_transformed = custom_image_transformed.unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        pred_logits = model(custom_image_transformed)

        pred_probs = torch.softmax(pred_logits, dim=1)

        pred_label = pred_probs.argmax(dim=1)

        # plt.imshow(custom_image_transformed.permute(1, 2, 0))
        # plt.axis("off")

        # title = f"Pred: {class_names[pred_label]} | Prob: {(pred_probs[pred_label] * 100):.1f} %"

        # plt.title(title)
        return class_names[pred_label], pred_probs
