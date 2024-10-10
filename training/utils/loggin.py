import os 
import torch 
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


# logger = None 

# def init_logger(run_dir):
#     global logger 
#     logger = SummaryWriter(log_dir=run_dir)
    
class DiffusionLossLogger:
    logger = None
    def __init__(
        self,
        run_dir: os.PathLike,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        bins: int = 100,
        decay: float = 0.1,
        log_scale: bool = True
    ):
        DiffusionLossLogger.logger = self.logger
        self.logger = SummaryWriter(log_dir=run_dir)
        
        step_indices = torch.arange(bins, dtype=torch.float32)
        self.sigmas = (sigma_max ** (1 / rho) + step_indices / (bins - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.loss_bins = torch.zeros_like(step_indices)
        self.stat_decay = decay
        # matplot
        plt.switch_backend('agg') # show nothing
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.log_scale = log_scale
        
    @torch.no_grad()
    def record_loss(self, losses, sigmas):
        losses = losses.cpu().detach()
        sigmas = sigmas.cpu().detach()
        loss_bin_ids = self.round_sigma(sigmas)
        selected_loss_bins = self.loss_bins[loss_bin_ids]
        self.loss_bins[loss_bin_ids] = selected_loss_bins.lerp(losses, 1 - self.stat_decay)
    
    
    def log_diffusion_loss(self, global_step):
        self.ax.cla()
        self.ax.plot(self.sigmas.numpy(), self.loss_bins.numpy())
        if self.log_scale:
            self.ax.set_xscale('log')    
        self.ax.set_xlabel('$\sigma$')
        self.ax.set_ylabel('Loss')
        self.logger.add_figure("loss/diffusion_loss", self.fig, global_step=global_step)
    
    
    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.reshape(1, -1, 1).to(torch.float32), self.sigmas.reshape(1, -1, 1).to(torch.float32)).argmin(2)
        # result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        
        return index

        