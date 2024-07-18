import torch
import torch.nn as nn
import torchvision.models as models


class SimCLRBranch(nn.Module):
    def __init__(self, feature_size=128, use_context=False) -> None:
        super().__init__()
        assert feature_size <= 128, "[ERROR] Feature size has to be less than 128"

        self.use_context = use_context
        self.RESNET_FEATURES_SIZE = 2048
        self.BERT_FEATURES_SIZE = 768
        self.total_features = self.RESNET_FEATURES_SIZE + self.BERT_FEATURES_SIZE if use_context else self.RESNET_FEATURES_SIZE

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet    
        self.projection_head = nn.Sequential(
            nn.Linear(self.total_features , 4 * feature_size),
            nn.ReLU(),
            nn.Linear(4 * feature_size, feature_size)
        )
    
    def forward(self, X, C=None):
        if C is None and self.use_context:
            C = torch.zeros(X.shape[0], self.BERT_FEATURES_SIZE).to(X.device)
        
        H = self.feature_extractor(X)
        H = torch.cat((H, C), dim=1) if self.use_context else H 
        Z = self.projection_head(H.flatten(start_dim=1))
        
        return H, Z

class SimCLR(nn.Module):
    def __init__(self, feature_size=128, use_context=False) -> None:
        super().__init__()
        self.branch = SimCLRBranch(feature_size, use_context=use_context)

    def forward(self, X1, X2, C1=None, C2=None):
        H1, Z1 = self.branch(X1, C1)
        H2, Z2 = self.branch(X2, C2)

        return {
            "projection_head": (Z1, Z2),
            "features_extracted": (H1, H2)
        }