import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class NTXentLoss(nn.Module):

    """
    Normalized Temperature-scaled Cross-Entropy Loss
    Args:
        temperature (float): Temperature parameter.
    """

    def __init__(self, temperature=1.0) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, y : torch.Tensor, y_hat : torch.Tensor) -> torch.Tensor:
        y_batch = torch.concat((y,y_hat), dim=0)

        y_batch = F.normalize(y_batch, p=2, dim=1)
        y_couples = torch.exp(y_batch.matmul(y_batch.T.contiguous())/self.temperature)

        mask = torch.eye(y_couples.shape[0], device=y_couples.device, dtype=torch.bool)
        y_couples = y_couples.masked_fill(mask, 0)

        y_sim_neg_couples = y_couples.sum(dim=1)
        y_sim_pos_couples = torch.exp(F.cosine_similarity(y, y_hat) / self.temperature)
        y_sim_pos_couples = torch.concat((y_sim_pos_couples, y_sim_pos_couples), dim=0)
        
        y_loss = -torch.log((y_sim_pos_couples / y_sim_neg_couples).mean())

        return y_loss
    
        # projections_1 = F.normalize(y, p=2, dim=1)
        # projections_2 = F.normalize(y_hat, p=2, dim=1)
        
        # # Compute similarities
        # similarities = torch.matmul(projections_1, projections_2.T) / self.temperature
        
        # batch_size = projections_1.shape[0]
        # contrastive_labels = torch.arange(batch_size).to(similarities.device)
        
        # # Compute the loss
        # loss_1_2 = F.cross_entropy(similarities, contrastive_labels, reduction='mean')
        # loss_2_1 = F.cross_entropy(similarities.T, contrastive_labels, reduction='mean')
        
        # return (loss_1_2 + loss_2_1) / 2