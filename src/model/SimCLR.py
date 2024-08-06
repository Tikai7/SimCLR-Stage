import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from model.BERT import BertEncoder


class SimCLRBranch(nn.Module):
    def __init__(self, feature_size=128, use_context=False, context_weights=1.0) -> None:
        super().__init__()
        self.use_context = use_context
        self.RESNET_FEATURES_SIZE = 2048
        self.BERT_FEATURES_SIZE = 768
        self.context_weight = context_weights
        self.total_features = self.RESNET_FEATURES_SIZE + self.BERT_FEATURES_SIZE if use_context else self.RESNET_FEATURES_SIZE


        self.bert = BertEncoder()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet    

        self.final_projection_head = nn.Sequential(
            nn.Linear(feature_size + self.BERT_FEATURES_SIZE, 4 * feature_size),
            nn.GELU(),
            nn.Linear(4 * feature_size, feature_size)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.RESNET_FEATURES_SIZE , 4 * feature_size),
            nn.ReLU(),
            nn.Linear(4 * feature_size, feature_size)
        )
    
    def forward(self, X, C="<UNK>"):
        H = self.feature_extractor(X)
        Z = self.projection_head(H.flatten(start_dim=1))

        if self.use_context:
            C = self.bert(C)
            C = F.normalize(C, p=2, dim=1)
            Z = F.normalize(Z, p=2, dim=1)
            Z = torch.cat((Z,C), dim=1)
            Z = self.final_projection_head(Z)

        return H, Z
        
class SimCLR(nn.Module):
    def __init__(self, feature_size=128, use_context=False, context_weights=1.0) -> None:
        super().__init__()
        self.branch = SimCLRBranch(feature_size, use_context=use_context, context_weights=context_weights)

    def forward(self, X1, X2, C1=None, C2=None):
        H1, Z1 = self.branch(X1, C1)
        H2, Z2 = self.branch(X2, C2)

        return {
            "projection_head": (Z1, Z2),
            "features_extracted": (H1, H2)
        }
