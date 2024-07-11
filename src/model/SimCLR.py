import torch.nn as nn
import torchvision.models as models


class SimCLRBranch(nn.Module):
    def __init__(self, feature_size=128) -> None:
        super().__init__()
        assert feature_size <= 128, "[ERROR] Feature size has to be less than 128"
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 4 * feature_size),
            nn.ReLU(),
            nn.Linear(4 * feature_size, feature_size)
        )
        print(self.feature_extractor)
    
    def forward(self, X):
        H = self.feature_extractor(X)
        Z = self.projection_head(H.flatten(start_dim=1))
        return H, Z


class SimCLR(nn.Module):
    def __init__(self, feature_size=512) -> None:
        super().__init__()
        self.branch = SimCLRBranch(feature_size)

    def forward(self, X1, X2):
        H1, Z1 = self.branch(X1)
        H2, Z2 = self.branch(X2)

        return {
            "projection_head": (Z1, Z2),
            "features_extracted": (H1, H2)
        }