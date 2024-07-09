import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, y, y_hat):
        y_batch = torch.concat((y,y_hat), dim=0)
        y_batch = F.normalize(y_batch, p=2, dim=1)
        y_couples = torch.exp((y_batch.matmul(y_batch.T.contiguous()))/self.temperature)

        mask = torch.eye(y_couples.shape[0], device=y_couples.device, dtype=torch.bool)
        y_couples = y_couples.masked_fill(mask, 0)

        y_sim_neg_couples = y_couples.sum(dim=1)
        y_sim_pos_couples = torch.exp((F.cosine_similarity(y, y_hat) / self.temperature))
        y_sim_pos_couples = torch.concat((y_sim_pos_couples, y_sim_pos_couples), dim=0)


        y_loss = -torch.log((y_sim_pos_couples / y_sim_neg_couples).mean())

        return y_loss