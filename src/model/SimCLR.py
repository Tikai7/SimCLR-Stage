import torch
import torch.nn as nn
import torchvision.models as models


class SimCLRBranch(nn.Module):
    def __init__(self, feature_size=128) -> None:
        super().__init__()
        assert feature_size <= 128, "[ERROR] Feature size has to be less than 128"

        self.RESNET_FEATURES_SIZE = 2048
        self.BERT_FEATURES_SIZE = 768

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        self.combined_fc = nn.Sequential([
            nn.Linear(self.RESNET_FEATURES_SIZE , self.BERT_FEATURES_SIZE)
        ])
        self.projection_head = nn.Sequential(
            nn.Linear(self.BERT_FEATURES_SIZE , 4 * feature_size),
            nn.ReLU(),
            nn.Linear(4 * feature_size, feature_size)
        )
    
    def forward(self, X, C=None):
        if C is None:
            C = torch.zeros(self.BERT_FEATURES_SIZE)

        H = self.feature_extractor(X)
        H = self.combined_fc(H) + C
        Z = self.projection_head(H.flatten(start_dim=1))
        
        return H, Z

class SimCLR(nn.Module):
    def __init__(self, feature_size=512) -> None:
        super().__init__()
        self.branch = SimCLRBranch(feature_size)

    def forward(self, X1, X2, C1=None, C2=None):
        H1, Z1 = self.branch(X1, C1)
        H2, Z2 = self.branch(X2, C2)

        return {
            "projection_head": (Z1, Z2),
            "features_extracted": (H1, H2)
        }