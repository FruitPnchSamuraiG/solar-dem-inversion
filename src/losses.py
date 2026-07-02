import torch.nn as nn
import torch
import numpy as np

class MaskedMSELoss(nn.Module):
    # masked MSE loss for nans
    def __init__(self, R=None, B=None):
        super().__init__()
        # response matrix R
        self.R = R 
        # basis matrix B
        self.B = B

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        diff = pred[mask] - target[mask]
        return torch.mean(diff ** 2)

class JointResynthLoss(nn.Module):
    def __init__(self, R, transformed=False):
        super().__init__()
        self.R = R.T  # [n_bins, C] -> [C, n_bins]
        self.transformed = transformed

    def forward(self, pred, target, aia_obs):
        # unbatched fix
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            aia_obs = aia_obs.unsqueeze(0)
        
        # dem loss
        mask = ~torch.isnan(target)
        dem_diff = pred[mask] - target[mask]
        dem_loss = torch.mean(dem_diff ** 2)

        # resynthesis
        if self.transformed:
            pred = pred ** 2
        
        B, n_bins, H, W = pred.shape
        pred_reshaped = pred.permute(0, 2, 3, 1)  # [B, H, W, n_bins]
        synth = torch.matmul(pred_reshaped, self.R.to(pred.device))  # [B, H, W, C]
        synth = synth.permute(0, 3, 1, 2)  # [B, C, H, W]

        if self.transformed:
            synth = torch.sqrt(torch.clamp(synth, min=0))

        # trick, scale synth to match the dem scale
        # so we have both mse losses in the same scale
        with torch.no_grad():
            scale = pred.mean() / (synth.mean() + 1e-8)
        synth = synth * scale
        aia_obs = aia_obs * scale
        
        # aia loss
        aia_mask = ~torch.isnan(aia_obs)
        aia_diff = synth[aia_mask] - aia_obs[aia_mask]
        aia_loss = torch.mean(aia_diff ** 2)

        return dem_loss + aia_loss

class ClassificationLoss(nn.Module):
    def __init__(self, bins, ignore_index=-1, R=None, B=None):
        super().__init__()
        self.bins = bins
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # response matrix R
        self.R = R 
        # basis matrix B
        self.B = B
    
    def forward(self, cls_pred, bin_targets):
        # cls_pred: [B, 26, 64, H, W] - logits
        # bin_targets: [B, 26, H, W] - target bin indices
        
        B, n_temps, n_bins, H, W = cls_pred.shape
        
        # reshape for cross-entropy: [B*26*H*W, 64] and [B*26*H*W]
        cls_pred_flat = cls_pred.permute(0, 1, 3, 4, 2).reshape(-1, n_bins)  
        bin_targets_flat = bin_targets.reshape(-1)
        
        # mask out invalid targets
        valid_mask = (bin_targets_flat >= 0) & (bin_targets_flat < n_bins)
        if valid_mask.sum() > 0:
            return self.ce_loss(cls_pred_flat[valid_mask], bin_targets_flat[valid_mask])
        else:
            return cls_pred_flat.sum() * 0.0  # preserve grad, avoid tensor creation

class RegressionClassificationLoss(nn.Module):
    def __init__(self, bins, alpha=0.5, R=None, B=None):
        super().__init__()
        self.bins = bins
        self.alpha = alpha  # weight between regression and classification
        self.mse_loss = MaskedMSELoss(R=R, B=B)
        self.cls_loss = ClassificationLoss(bins, R=R, B=B)
        self.R = R
        self.B = B

    def forward(self, reg_pred, cls_pred, dem_target, bin_targets):
        reg_loss = self.mse_loss(reg_pred, dem_target)
        cls_loss = self.cls_loss(cls_pred, bin_targets)
        return self.alpha * reg_loss + (1 - self.alpha) * cls_loss


class RegressionClassificationResynthLoss(nn.Module):
    """RegressionClassificationLoss + AIA resynthesis consistency term.

    Weight schedule (controlled by set_epoch):
      epoch < resynth_start_epoch : rc_weight=1, resynth_weight=0  (reg+cls phase)
      epoch >= resynth_start_epoch: rc_weight=0, resynth_weight=1  (resynth-only phase)
    """
    def __init__(self, bins, alpha=0.5, resynth_weight=0.0, rc_weight=1.0,
                 resynth_start_epoch=100, R=None, transformed=False):
        super().__init__()
        self.bins = bins
        self.alpha = alpha
        self.resynth_weight = resynth_weight
        self.rc_weight = rc_weight
        self.resynth_start_epoch = resynth_start_epoch
        self.transformed = transformed
        self.mse_loss = MaskedMSELoss()
        self.cls_loss = ClassificationLoss(bins)
        self.R = R  # [n_obs, n_temps], consistent with other losses
        self.B = None

    def set_epoch(self, epoch):
        """Switch from reg+cls phase to resynth-only phase at resynth_start_epoch."""
        if epoch < self.resynth_start_epoch:
            self.rc_weight, self.resynth_weight = 1.0, 0.0
        else:
            self.rc_weight, self.resynth_weight = 0.0, 1.0

    def forward(self, reg_pred, cls_pred, dem_target, bin_targets, aia_obs, aia_errors=None):
        reg_loss = self.mse_loss(reg_pred, dem_target)
        cls_loss = self.cls_loss(cls_pred, bin_targets)
        rc_loss = self.alpha * reg_loss + (1 - self.alpha) * cls_loss

        if self.R is None:
            total = self.rc_weight * rc_loss
            return total, {'reg_loss': reg_loss, 'cls_loss': cls_loss, 'resynth_loss': torch.tensor(0.0)}

        # resynthesis loss (mirrors JointResynthLoss logic)
        dem = reg_pred ** 2 if self.transformed else reg_pred
        pred_reshaped = dem.permute(0, 2, 3, 1)  # [B, H, W, n_temps]
        # R has 18 bins (AIA range); pad with zeros if model outputs more (e.g. 26 for XRT extension)
        n_temps = pred_reshaped.shape[-1]
        R = self.R.to(reg_pred.device).T  # [n_temps, n_obs]
        if n_temps > R.shape[0]:
            pad = torch.zeros(n_temps - R.shape[0], R.shape[1], device=reg_pred.device)
            R = torch.cat([R, pad], dim=0)  # [n_temps, n_obs]
        synth = torch.matmul(pred_reshaped, R)  # [B, H, W, n_obs]
        synth = synth.permute(0, 3, 1, 2)  # [B, n_obs, H, W]
        if self.transformed:
            synth = torch.sqrt(torch.clamp(synth, min=0))

        aia_mask = ~torch.isnan(aia_obs)
        # use aiapy photon error estimates if provided, otherwise use max(1, sqrt(x)) approximation
        if aia_errors is not None:
            sigma = aia_errors[aia_mask] + 1e-8
        else:
            sigma = torch.clamp(torch.sqrt(torch.abs(aia_obs[aia_mask])), min=1.0)
        chi_squared = ((synth[aia_mask] - aia_obs[aia_mask]) / sigma) ** 2
        resynth_loss = torch.mean(chi_squared)

        total = self.rc_weight * rc_loss + self.resynth_weight * resynth_loss
        return total, {'reg_loss': reg_loss, 'cls_loss': cls_loss, 'resynth_loss': resynth_loss}
    
def barrier_loss_batch(x, D, I_obs, lb, ub, a_l2=0, a_l1=1.0, mu=1.0, alpha=0, mu_pos=0):
    # x: [B, n_basis]
    # D: [n_obs, n_temps]
    # I_obs: [B, n_obs]
    # lb, ub: [B, n_obs] lower and upper bounds for the observed data

    # per-channel noise sigma derived from the tolerance band: sigma = (ub - lb) / 2
    # dividing by sigma^2 normalises each channel so faint and bright channels
    # contribute equally (chi-squared style), preventing bright channels from
    # dominating and the NN ignoring faint ones (e.g. 94A, 131A)
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8  # [B, n_obs]

    # L2 regularization
    l2_term = 0.5 * a_l2 * torch.sum(x**2, dim=1)  # [B]

    # L1 regularization
    l1_term = a_l1 * torch.sum(torch.abs(x), dim=1)  # [B]

    Dx = torch.matmul(x, D.T)  # [B, n_obs]

    # barrier for x >= 0
    barrier_x = mu_pos * torch.sum(torch.relu(-x), dim=1)  # [B]

    # barrier for Dx >= lb  (normalised by sigma^2)
    barrier_lb = mu * torch.sum(torch.relu(lb - Dx)**2 / sigma2, dim=1)  # [B]

    # barrier for Dx <= ub  (normalised by sigma^2)
    barrier_ub = mu * torch.sum(torch.relu(Dx - ub)**2 / sigma2, dim=1)  # [B]

    # fit term (chi-squared: normalised by sigma^2)
    fit = alpha * torch.sum((Dx - I_obs)**2 / sigma2, dim=1)  # [B]

    total_loss = l2_term + l1_term + barrier_x + barrier_lb + barrier_ub + fit
    return total_loss.mean()

def enet_loss_batch(x, D, I_obs, lb, ub, C=1.0, alpha=1.0, lam=0.9):
    # Elastic Net objective (Athiray & Winebarger 2024), unconstrained:
    #   (1/2C)||o - Dx||^2 + alpha*lam*||x||_1 + (1/2)*alpha*(1-lam)*||x||_2^2
    # x >= 0 is handled by the network's Softplus output, so no barrier needed.
    # The fit term is chi-squared normalised (sigma from the tolerance band,
    # same convention as barrier_loss_batch) so faint channels aren't ignored.
    # lam=1 recovers Lasso; lam=0 recovers ridge.
    sigma2 = ((ub - lb) / 2) ** 2 + 1e-8  # [B, n_obs]

    Dx = torch.matmul(x, D.T)  # [B, n_obs]
    fit = (0.5 / C) * torch.sum((Dx - I_obs) ** 2 / sigma2, dim=1)  # [B]
    l1 = alpha * lam * torch.sum(torch.abs(x), dim=1)               # [B]
    l2 = 0.5 * alpha * (1.0 - lam) * torch.sum(x ** 2, dim=1)       # [B]

    return (fit + l1 + l2).mean()


class BarrierLoss(nn.Module):
    def __init__(self, D, R, B, args=None):
        super().__init__()
        self.D = D
        self.R = R
        self.B = B
        self.a_l2 = args.alpha_l2
        self.a_l1 = args.alpha_l1
        self.mu = args.mu
        self.alpha = args.alpha_fit

    def forward(self, x, aia_obs, lb, ub):
        # lb: lower bound, shape [B, n_obs]
        # ub: upper bound, shape [B, n_obs]
        return barrier_loss_batch(x, self.D, aia_obs, lb, ub,
                                  self.a_l2, self.a_l1, self.mu, self.alpha)