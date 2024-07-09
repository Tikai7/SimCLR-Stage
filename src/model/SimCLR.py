import torch.nn as nn
from torchvision import models

class SimCLRBranch(nn.Module):
    def __init__(self, feature_size=512) -> None:
        super().__init__()
        assert feature_size <= 512, "[ERROR] Feature size has to be less than 512"
        resnet = models.resnet50(pretrained=True, weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 4*feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size)
        )
    
    def forward(self, X):
        H = self.feature_extractor(X)
        Z = self.projection_head(H)
        return (H, Z)


class SimCLR(nn.Module):
    def __init__(self, feature_size=512) -> None:
        super().__init__()
        self.branch_1 = SimCLRBranch(feature_size)
        self.branch_2 = SimCLRBranch(feature_size)

    def forward(self, X1, X2):
        H1, Z1 = self.branch_1(X1)
        H2, Z2 = self.branch_2(X2)

        return {
            "projection_head" : (Z1, Z2),
            "features_extracted" : (H1, H2)
        }