from prediction import pred_and_plot_on_custom_data
from torchvision import transforms
import model_builder
import torch
from preprocess.preprocess import *
import cv2

HIDDEN_UNITS = 16
class_names = ["net", "not_net"]

model = model_builder.OceanGate(
    input_channels=3, hidden_units=HIDDEN_UNITS, output_channels=len(class_names)
)

model.load_state_dict(torch.load(f="../OceanGate2/models/OceanGateV0.pth"))

transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ]
)

image = cv2.imread("./preprocess/data/img4.png", cv2.IMREAD_UNCHANGED)


def main():
    rectangles, morph = preprocess(image=image)

    for i in rectangles:
        x, y, w, h = i[0], i[1], i[2], i[3]

        rect_image = image[
            y : y + h,
            x : x + w,
        ]

        rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)

        pred_label, pred_prob = pred_and_plot_on_custom_data(
            model=model,
            image=rect_image,
            transform=transform,  # type: ignore
            class_names=class_names,
        )

        if pred_label == "net":
            probability = pred_prob[0][0] * 100

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 255, 255), -1)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY)
            res = cv2.bitwise_and(thresh, morph)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            res = cv2.bitwise_and(res, img_rgb)

            gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            # Create a mask for non-black regions (assuming black is [0, 0, 0] in BGR)
            mask = (gray_image > 0).astype(np.uint8) * 255

            # Increase the green channel in non-black regions to make them more green
            res[
                mask > 0, 1
            ] += 50  # Increase the greenness (adjust the value as needed)

            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

            result = res + img_rgb

            text = f"Pred: Ghost {pred_label} | Prob: {probability:.2f} %"
            cv2.putText(
                result,
                text,
                (x, y + 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                10,
            )
            plt.imshow(result)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()
