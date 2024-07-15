import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

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


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self.ssim(img1, img2, window=window, size_average=self.size_average)
    
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, window=None, size_average=True):
        if window is None:
            channel = img1.size(1)
            window = self.create_window(window_size, channel)
        
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)