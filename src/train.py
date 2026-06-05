import os
import torch
import torch.distributed as dist
from datetime import datetime
from src.utils import compute_photon_error_bounds
import wandb
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np


def _reduce_tensor(tensor, world_size):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor

def handle_loss_fn(loss_fn, output, im1, im2):
    # this function handles the different loss functions that can be used in training.
    # it computes the loss and returns it along with any additional components needed for logging.
    comp = {}
    if loss_fn.__class__.__name__ == 'JointResynthLoss':
        aia = im1
        dem = im2
        loss = loss_fn(output, aia, dem)
    elif loss_fn.__class__.__name__ == 'BarrierLoss':
        aia = im1
        aia_errors = im2
        
        # flatten the aia observations and basis outputs for loss
        aia_obs_flat = im1.permute(0, 2, 3, 1).reshape(-1, im1.shape[1])  # [B * H * W, n_channels]
        output_flat = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1]) # [B * H * W, n_basis]

        # compute lower and upper bounds and flatten them
        lb, ub = compute_photon_error_bounds(aia_obs=aia, aia_errors=aia_errors)
        lb = lb.reshape(-1, lb.shape[-1])  # [B*H*W, n_channels]
        ub = ub.reshape(-1, ub.shape[-1])  # [B*H*W, n_channels]    

        comp['lb'] = lb
        comp['ub'] = ub
        comp['output_flat'] = output_flat

        loss = loss_fn(output_flat, aia_obs=aia_obs_flat, lb=lb, ub=ub)
    elif loss_fn.__class__.__name__ == 'MaskedMSELoss':
        dem = im2
        output_dem = output
        if output.shape[1] != dem.shape[1]:
            # reconstruct dem
            output_flat = output.permute(0, 2, 3, 1).reshape(-1, output.shape[1])  # [B * H * W, n_basis]
            output_dem = torch.matmul(output_flat, loss_fn.B.T)
            output_dem = output_dem.reshape(im1.size(0), im1.shape[2], im1.shape[3], -1).permute(0, 3, 1, 2) # [B, n_temps, H, W]
        loss = loss_fn(output_dem, dem)

    return loss, comp

def train_one_epoch(epoch, model, data_loader, loss_fn, optimizer, scheduler, device, args, world_size=1, rank=0, is_main_process=True):
    data_time = SmoothedValue()
    batch_time = SmoothedValue()

    model.train()
    end = time.time()
    train_loss = torch.tensor(0.0, device=device)
    #first_batch = next(iter(data_loader)) # FOR TESTING ONLY
    #for i, data in enumerate(itertools.repeat(first_batch)): # FOR TESTING ONLY
    for i, data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global_step = epoch * len(data_loader) + i

        im1, im2 = data
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # forward
        output = model(im1)

        # handle the loss function
        loss, comp = handle_loss_fn(loss_fn, output, im1, im2)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.detach()  # no sync, accumulate on GPU  

        # if scheduler is not None:
        #     scheduler.step(loss.item())

        if is_main_process and i % args.log_loss_interval == 0:
            if i % args.log_image_interval == 0:
                print(f"  ->batch {i} of {len(data_loader)}, loss: {loss.item():.3f}")
                print(f"    data time: {data_time.avg:.1e} batch time: {batch_time.avg:.3f}")
            if args.wandb:
                log_dict = {
                    'train_loss': loss.item(),
                    'batch': i,
                    'epoch': epoch,
                    'data_time': data_time.avg,
                    'batch_time': batch_time.avg,
                    'global_step': global_step,
                }
                for k in ('reg_loss', 'cls_loss', 'resynth_loss'):
                    if k in comp:
                        log_dict[k] = comp[k].item()
                if 'resynth_loss' in comp and 'reg_loss' in comp:
                    log_dict['resynth_reg_ratio'] = (comp['resynth_loss'] / (comp['reg_loss'] + 1e-8)).item()
                wandb.log(log_dict)

        if is_main_process and global_step % args.log_image_interval == 0:
            if loss_fn.__class__.__name__ == 'BarrierLoss':
                output_flat = comp['output_flat']
                lb = comp['lb']
                ub = comp['ub']
                l1_check = torch.sum(torch.abs(output_flat), dim=1)  # [B*H*W]
                Dx = torch.matmul(output_flat, loss_fn.D.T)
                lb_check = torch.sum(torch.relu(lb - Dx), dim=1)   # [B*H*W]
                up_check = torch.sum(torch.relu(Dx - ub), dim=1)  # [B*H*W]
                print('    l1_check:', l1_check.mean().item(), 'lb_check:', lb_check.mean().item(), 'up_check:', up_check.mean().item())
                if args.wandb:
                    wandb.log({'l1_check': l1_check.mean().item(),
                            'lb_check': lb_check.mean().item(),
                                'up_check': up_check.mean().item(),
                                'global_step': global_step,})
            if args.wandb:
                print(f'    logging image samples at step {global_step}')
                log_image_samples(model, loss_fn, im1, im2, output, global_step, is_main_process=is_main_process)

        # measure total batch time
        batch_time.update(time.time() - end)
        end = time.time()

    avg_loss = train_loss / len(data_loader)
    avg_loss = _reduce_tensor(avg_loss, world_size)
    return avg_loss.item()  # single sync at epoch end

def validate_epoch(model, data_loader, loss_fn, device, world_size=1):
    # simple validation function
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for img, dem in data_loader:
            img = img.to(device, non_blocking=True)
            dem = dem.to(device, non_blocking=True)
            # forward
            output = model(img)

            # handle the loss function
            loss, _ = handle_loss_fn(loss_fn, output, img, dem)

            val_loss += loss.detach()

    avg_loss = val_loss / len(data_loader)
    avg_loss = _reduce_tensor(avg_loss, world_size)
    return avg_loss.item()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, device='cuda', args=None, train_sampler=None, world_size=1, rank=0, is_main_process=True, start_epoch=0, initial_train_loss=None, initial_val_loss=None, initial_best_val_loss=None, **kw):
    # handle DDP wrapper - access underlying model for attributes
    base_model = model.module if hasattr(model, 'module') else model

    train_loss_epoch = list(initial_train_loss) if initial_train_loss else []
    train_val = list(initial_val_loss) if initial_val_loss else []
    best_val_loss = initial_best_val_loss

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)

        if is_main_process:
            print('EPOCH {}:'.format(epoch + 1))

        # training
        avg_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device, args, world_size=world_size, rank=rank, is_main_process=is_main_process)
        train_loss_epoch.append(avg_loss)
        
        if scheduler is not None:
            scheduler.step()

        global_step = epoch * len(train_loader) + len(train_loader) - 1
        val_interval = getattr(args, 'val_interval', 1)
        do_val = ((epoch + 1) % val_interval == 0) or (epoch + 1 == epochs)

        if is_main_process:
            print(f"EPOCH {epoch + 1} finished. train_loss: {avg_loss:.6f}", end="")

        if do_val:
            # validation
            val_loss = validate_epoch(model, val_loader, criterion, device, world_size=world_size)
            train_val.append(val_loss)

            if is_main_process:
                print(f"  val_loss: {val_loss:.6f}")
                if args.wandb:
                    epoch_log = {
                        'epoch': epoch,
                        'epoch_avg_train_loss': avg_loss,
                        'val_loss': val_loss,
                        'global_step': global_step,
                    }
                    if hasattr(criterion, 'rc_weight'):
                        epoch_log['rc_weight'] = criterion.rc_weight
                        epoch_log['resynth_weight'] = criterion.resynth_weight
                    wandb.log(epoch_log)
                # save
                save_model(model, optimizer, train_loss_epoch, train_val, epoch, base_model.name, criterion, args=args, is_distributed=world_size > 1, **kw)

                # save best model
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_best_model(model, optimizer, train_loss_epoch, train_val, epoch, base_model.name, criterion, args=args, is_distributed=world_size > 1, **kw)
                    print(f"  → New best model! val_loss: {val_loss:.6f}")
        else:
            if is_main_process:
                print()
                if args.wandb:
                    epoch_log = {
                        'epoch': epoch,
                        'epoch_avg_train_loss': avg_loss,
                        'global_step': global_step,
                    }
                    if hasattr(criterion, 'rc_weight'):
                        epoch_log['rc_weight'] = criterion.rc_weight
                        epoch_log['resynth_weight'] = criterion.resynth_weight
                    wandb.log(epoch_log)

    if is_main_process:
        print("Training complete.")
        print(f"Best validation loss: {best_val_loss:.6f}")
    return train_loss_epoch, train_val

def save_model(model, optimizer, train_loss, val_loss, epoch, model_name, loss_fn, args=None, is_distributed=False, **kw):
    if is_distributed and not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
        return
    wrapped_model = model.module if hasattr(model, "module") else model
    # save the model checkpoint
    folder_name = f"{args.model}-{args.loss}_{args.run_name}_{kw['timestamp']}"
    save_dir = os.path.join("..", "results", "models", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")

    checkpoint = {
        "experiment_name": args.run_name,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state_dict": wrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "timestamp": kw['timestamp'],
        "model_name": model_name,
        "loss_name": loss_fn.__class__.__name__,
        "args": vars(args) if args else {},
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }

    torch.save(checkpoint, save_path)
    print(f"model saved to: {save_path}")

def save_best_model(model, optimizer, train_loss, val_loss, epoch, model_name, loss_fn, args=None, is_distributed=False, **kw):
    """save the best model checkpoint based on validation loss"""
    if is_distributed and not (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0):
        return
    wrapped_model = model.module if hasattr(model, "module") else model
    # save the model checkpoint
    folder_name = f"{args.model}-{args.loss}_{args.run_name}_{kw['timestamp']}"
    save_dir = os.path.join("..", "results", "models", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "model_best.pth")

    checkpoint = {
        "experiment_name": args.run_name,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_state_dict": wrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "timestamp": kw['timestamp'],
        "model_name": model_name,
        "loss_name": loss_fn.__class__.__name__,
        "args": vars(args) if args else {},
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }

    torch.save(checkpoint, save_path)
    print(f"  best model saved to: {save_path}")

def log_image_samples(model, loss_fn, im1, im2, outs, global_step, is_main_process=True):
    if not is_main_process:
        return
    
    # handle DDP wrapper
    base_model = model.module if hasattr(model, 'module') else model
    
    wavelengths = [94, 131, 171, 193, 211, 335]

    # handle tuple output from classification models
    if isinstance(outs, tuple) and len(outs) == 2:
        # for dual-head models: (regression, classification)
        regression_output, classification_output = outs
        dem_pred_cube = regression_output  # use regression output for DEM visualization
    elif base_model.requires_basis:
        output_flat = outs.permute(0, 2, 3, 1).reshape(-1, outs.shape[1])  # [B * H * W, n_basis]
        output_dem = torch.matmul(output_flat, loss_fn.B.T)
        dem_pred_cube = output_dem.reshape(im1.size(0), im1.shape[2], im1.shape[3], -1).permute(0, 3, 1, 2) # [B, n_temps, H, W]
    else:
        dem_pred_cube = outs

    # pick the first sample
    dem = dem_pred_cube[0].detach().cpu().numpy()
    aia = im1[0].detach().cpu().numpy()

    logs = {}
    # DEM channels
    for i in range(dem.shape[0]):
        rgb = apply_cmap(dem[i], "viridis", vmin=0, vmax=10)
        name = f"DEM/ch_{i:02d}"
        logs[f"examples/{name}"] = wandb.Image(rgb, caption=f"DEM bin {i}")

    if im2.shape[2] == dem_pred_cube.shape[2]:
        # then we are directly predicting the DEM
        # save the ground truth DEM
        dem_gt = im2[0].detach().cpu().numpy()
        for i in range(dem_gt.shape[0]):
            rgb = apply_cmap(dem_gt[i], "viridis", vmin=0, vmax=10)
            name = f"DEM_GT/ch_{i:02d}"
            logs[f"examples/{name}"] = wandb.Image(rgb, caption=f"DEM GT bin {i}")

    if loss_fn.R is not None:
        R = loss_fn.R
        n_temps = dem_pred_cube.shape[1]
        if n_temps > R.shape[1]:
            pad = torch.zeros(R.shape[0], n_temps - R.shape[1], device=R.device)
            R = torch.cat([R, pad], dim=1)
        synth = torch.tensordot(R, dem_pred_cube, dims=([1], [1]))
        synth = synth.permute(1, 0, 2, 3)[0].detach().cpu().numpy()
        diff = (aia - synth)

        # AIA / synth / diff
        for i, wl in enumerate(wavelengths):
            cmap_name = f"sdoaia{wl}"
            for kind, arr, cmap in [
                ("AIA",  aia,  cmap_name),
                ("Synth", synth, cmap_name),
                ("Diff", diff, "bwr"),
            ]:
                if kind == "Diff":
                    rgb = apply_cmap(arr[i], cmap, vmin=-20, vmax=20)
                else:
                    rgb = apply_cmap(arr[i], cmap, vmin=aia[i].min(), vmax=aia[i].max())
                name = f"{kind}/{wl}Å"
                logs[f"examples/{name}"] = wandb.Image(rgb, caption=f"{kind} {wl}Å")

    logs['global_step'] = global_step
    wandb.log(logs)

def apply_cmap(arr, cmap_name, vmin=None, vmax=None):
    cmap = plt.get_cmap(cmap_name)
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()
    normed = (arr - vmin) / (vmax - vmin)
    rgba = cmap(normed)            # H×W×4 float64 0–1
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb

class SmoothedValue:
    def __init__(self, window_size=100):
        self.deque = deque(maxlen=window_size)
    def update(self, value):
        self.deque.append(value)
    @property
    def avg(self):
        if not self.deque:
            return 0.0
        return sum(self.deque) / len(self.deque)
