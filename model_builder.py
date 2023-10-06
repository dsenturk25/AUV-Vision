import torch 
from torch import nn
from torchvision import datasets
import numpy as np

class OceanGate(nn.Module):

  def __init__(self, input_channels, hidden_units, output_channels) -> None:
    super().__init__()


    self.conv_layer1 = nn.Sequential(
      nn.Conv2d(
        in_channels=input_channels,
        out_channels=hidden_units,
        kernel_size=(3, 3),
        stride=0,
        padding=1
      ),
      nn.ReLU(),
      nn.Conv2d(
        in_channels=hidden_units,
        out_channels=hidden_units,
        kernel_size=(3, 3),
        stride=0,
        padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d((2, 2))
    )


    self.conv_layer2 = nn.Sequential(
      nn.Conv2d(
        in_channels=hidden_units,
        out_channels=hidden_units,
        kernel_size=(3, 3),
        stride=0,
        padding=1
      ),
      nn.ReLU(),
      nn.Conv2d(
        in_channels=hidden_units,
        out_channels=hidden_units,
        kernel_size=(3, 3),
        stride=0,
        padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d((2, 2))
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(
        in_features=hidden_units,
        out_features=output_channels
      ),
      nn.ReLU(),
    )

  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_layer1(x)

    x = self.conv_layer2(x)

    x = self.classifier(x)
    return x
