import torch.nn as nn
import torchvision.models as models

#florence2

class SimCLRBranch(nn.Module):
    """
        SimCLR branch
        Branch for SimCLR model to extract features and project them to a lower dimension
    """
    
    def __init__(self, feature_size=128) -> None:
        super().__init__()
        self.RESNET_FEATURES_SIZE = 2048

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet    

        self.projection_head = nn.Sequential(
            nn.Linear(self.RESNET_FEATURES_SIZE , 4 * feature_size),
            nn.ReLU(),
            nn.Linear(4 * feature_size, feature_size)
        )
    
    def forward(self, X, C=None):
        H = self.feature_extractor(X)
        Z = self.projection_head(H.flatten(start_dim=1))
        return H, Z
        
class SimCLR(nn.Module):
    """
        SimCLR model implementation
        returns the projection head and the extracted featuress
    """

    def __init__(self, feature_size=128) -> None:
        super().__init__()
        self.branch = SimCLRBranch(feature_size)

    def forward(self, X1, X2, C1=None, C2=None):
        H1, Z1 = self.branch(X1, C1)
        H2, Z2 = self.branch(X2, C2)

        return {
            "projection_head": (Z1, Z2),
            "features_extracted": (H1, H2)
        }