import torch
from torch import nn


class RoadSignGuardCNN(nn.Module):
    """Shared CNN trunk with sign-classification and occlusion-detection heads."""
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Dropout(0.25))
        self.sign_head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.occlusion_head = nn.Sequential(nn.Linear(192, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        features = self.shared(self.features(x))
        return self.sign_head(features), self.occlusion_head(features).squeeze(1)
