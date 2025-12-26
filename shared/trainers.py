from abc import ABC, abstractmethod
from collections import OrderedDict

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import os
from PIL import Image

EPSILON = 1e-8


def one_hot_cross_entropy(pred, target):
    """Calculate the cross entropy between two one-hot vectors."""
    return -torch.sum(target * torch.log(pred + EPSILON), dim=-1)


# =========================
# Mask + image debug helpers
# =========================
@torch.no_grad()
def red_agent_mask(
    obs: torch.Tensor,
    r_min: float = 0.55,
    g_max: float = 0.35,
    b_max: float = 0.35,
) -> torch.Tensor:
    """
    Heuristic mask for MiniGrid red agent pixels.

    Args:
        obs: (B,3,H,W) float in [0,1], RGB order.
    Returns:
        mask: (B,1,H,W) bool
    """
    assert obs.ndim == 4 and obs.size(1) == 3, f"Expected (B,3,H,W), got {obs.shape}"
    r = obs[:, 0:1]
    g = obs[:, 1:2]
    b = obs[:, 2:3]
    return (r > r_min) & (g < g_max) & (b < b_max)


import os
import numpy as np
from PIL import Image
import torch

@torch.no_grad()
def save_mask_debug_images(
    obs: torch.Tensor,
    obs_recon: torch.Tensor,
    mask: torch.Tensor,
    out_dir: str = "mask_debug",
    step: int = 0,
):
    """
    Save debug images to verify the mask matches the agent pixels AND inspect recon quality.

    Saves:
      - obs_step{step}.png
      - mask_step{step}.png
      - overlay_obs_step{step}.png
      - recon_step{step}.png
      - overlay_recon_step{step}.png
      - absdiff_step{step}.png           (L1 diff averaged over channels, normalized)
      - crop_obs_step{step}.png          (tight crop around mask)
      - crop_recon_step{step}.png
      - crop_overlay_recon_step{step}.png

    Args:
        obs:       (B,3,H,W) float in [0,1]
        obs_recon: (B,3,H,W) float in [0,1]
        mask:      (B,1,H,W) bool/float in {0,1}
        out_dir:   folder to write images
        step:      global step number for filenames
    """
    os.makedirs(out_dir, exist_ok=True)

    # Take first element in batch
    obs0 = obs[0].detach().cpu().clamp(0, 1)        # (3,H,W)
    rec0 = obs_recon[0].detach().cpu().clamp(0, 1)  # (3,H,W)
    m0 = mask[0, 0].detach().cpu()                  # (H,W)

    # To uint8 HWC
    obs_u8 = (obs0.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H,W,3)
    rec_u8 = (rec0.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H,W,3)
    mask_u8 = ((m0.numpy() > 0.5).astype(np.uint8) * 255)              # (H,W)

    # --- overlay helpers ---
    HIGHLIGHT = np.array([255, 0, 255], dtype=np.uint8)  # magenta (won't clash with MiniGrid green goal)
    alpha = 0.65  # blend strength

    def apply_overlay(img_u8: np.ndarray, mask_u8_: np.ndarray) -> np.ndarray:
        out = img_u8.copy()
        m = (mask_u8_ == 255)
        # alpha blend highlight on masked pixels so you can still see the content underneath
        out[m] = (alpha * HIGHLIGHT + (1.0 - alpha) * out[m]).astype(np.uint8)
        return out

    overlay_obs = apply_overlay(obs_u8, mask_u8)
    overlay_rec = apply_overlay(rec_u8, mask_u8)

    # --- abs diff visualization (L1), normalized for visibility ---
    diff = np.abs(obs_u8.astype(np.int16) - rec_u8.astype(np.int16)).astype(np.float32)  # (H,W,3)
    diff = diff.mean(axis=2)  # (H,W)
    mx = float(diff.max())
    if mx > 1e-6:
        diff = diff / mx
    diff_u8 = (diff * 255.0).astype(np.uint8)

    # --- save full images ---
    Image.fromarray(obs_u8).save(os.path.join(out_dir, f"obs_step{step}.png"))
    Image.fromarray(mask_u8).save(os.path.join(out_dir, f"mask_step{step}.png"))
    Image.fromarray(overlay_obs).save(os.path.join(out_dir, f"overlay_obs_step{step}.png"))

    Image.fromarray(rec_u8).save(os.path.join(out_dir, f"recon_step{step}.png"))
    Image.fromarray(overlay_rec).save(os.path.join(out_dir, f"overlay_recon_step{step}.png"))

    Image.fromarray(diff_u8).save(os.path.join(out_dir, f"absdiff_step{step}.png"))

    # --- save a tight crop around the mask (makes blur obvious) ---
    ys, xs = np.where(mask_u8 == 255)
    if len(xs) > 0:
        pad = 6
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, obs_u8.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, obs_u8.shape[1])

        Image.fromarray(obs_u8[y0:y1, x0:x1]).save(os.path.join(out_dir, f"crop_obs_step{step}.png"))
        Image.fromarray(rec_u8[y0:y1, x0:x1]).save(os.path.join(out_dir, f"crop_recon_step{step}.png"))
        Image.fromarray(overlay_rec[y0:y1, x0:x1]).save(os.path.join(out_dir, f"crop_overlay_recon_step{step}.png"))

def get_main_trans_layers(model):
    layers = []
    for module in model.shared_layers:
        if isinstance(module, nn.Linear):
            layers.append(module)
    for module in model.state_head:
        if isinstance(module, nn.Linear):
            layers.append(module)
    return layers


def get_main_trans_activations(model):
    layers = []
    for module in model.shared_layers:
        if isinstance(module, nn.ReLU):
            layers.append(module)
    for module in model.state_head:
        if isinstance(module, nn.ReLU):
            layers.append(module)
    return layers


def record_trans_model_update(trans_model, loss, optimizer, activations=None, grad_clip=0):
    modules = get_main_trans_layers(trans_model)
    norms = {}

    # Compute gradients
    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(trans_model.parameters(), grad_clip)

    # Store the weights before the update
    weights_before_update = [module.weight.data.clone() for module in modules]

    # Update the weights
    optimizer.step()

    grads_flattened = torch.cat([module.weight.grad.view(-1) for module in modules])

    # l0, l1, and l2 norms of the gradients
    grad_l0_norm = torch.norm(grads_flattened, p=0)
    grad_l1_norm = torch.norm(grads_flattened, p=1)
    grad_l2_norm = torch.norm(grads_flattened, p=2)

    # l2 norm of the weight update
    weight_update_flattened = torch.cat(
        [(module.weight.data - weight_before).view(-1)
         for module, weight_before in zip(modules, weights_before_update)])
    weight_change_l2_norm = torch.norm(weight_update_flattened, p=2)

    norms['grad_l0_norm'] = grad_l0_norm.item()
    norms['grad_l1_norm'] = grad_l1_norm.item()
    norms['grad_l2_norm'] = grad_l2_norm.item()
    norms['weight_change_l2_norm'] = weight_change_l2_norm.item()

    if activations is not None:
        activations = torch.concat([a.view(-1) for a in activations])
        norms['activation_l0_norm'] = torch.norm(activations, p=0).item()

    return norms


class ActivationRecorder:
    def __init__(self, modules: nn.Module):
        self.activations = []
        self._hooks = []
        for module in modules:
            hook = module.register_forward_hook(self.record_activation)
            self._hooks.append(hook)

    def record_activation(self, module, input, output):
        self.activations.append(output.detach())

    def reset(self):
        out = self.activations
        self.activations = []
        return out


class BaseRepresentationLearner(ABC):
    def __init__(self, model=None, batch_size=32, update_freq=32, log_freq=100):
        if model is None:
            self._init_model()
        else:
            self.model = model

        assert hasattr(self.model, 'encoder'), 'Model must have an encoder!'
        self.encoder = self.model.encoder
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.log_freq = log_freq

    @abstractmethod
    def _init_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_losses(self, batch_data):
        pass

    @abstractmethod
    def train(self, batch_data):
        pass


class AETrainer(BaseRepresentationLearner):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 256,
        update_freq: int = 128,
        log_freq: int = 100,
        lr: float = 3e-4,
        recon_loss_clip: float = 0,
        grad_clip: float = 0,
    ):
        super().__init__(model, batch_size, update_freq, log_freq)
        self.model = model
        self.recon_loss_clip = recon_loss_clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_step = 0
        self.grad_clip = grad_clip

        # Debug frequency (prints agent-vs-bg MSE)
        self.agent_dbg_freq = 200

    def _init_model(self):
        raise Exception('VAE requires a model to be specified!')

    def calculate_losses(self, batch_data, return_stats=False):
        device = next(self.model.parameters()).device
        sample_size = int(batch_data[0].shape[0] / 2)

        obs = torch.cat(
            [batch_data[0][:sample_size],
             batch_data[2][sample_size:]], dim=0
        ).to(device)

        loss_dict = {}

        # Forward pass
        try:
            obs_recon, quantizer_loss, perplexity, oh_encodings = self.model(obs)
            loss_dict["quantizer_loss"] = quantizer_loss
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            nan = torch.tensor(float("nan"), device=device)
            return {"recon_loss": nan, "quantizer_loss": nan}, ({} if return_stats else {})

        # Handle spatial dimension mismatches (NOTE: bilinear can blur tiny sprites)
        if obs.shape != obs_recon.shape:
            print(f"[warn] Dimension mismatch: input {obs.shape} vs recon {obs_recon.shape} -> interpolating")
            obs_recon = F.interpolate(
                obs_recon,
                size=obs.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # -------------------------
        # Agent mask + diagnostics
        # -------------------------
        mask = red_agent_mask(obs).float()  # (B,1,H,W) in {0,1}

        # Optional: dilate by 1 pixel to include edges of the agent sprite
        # (often helps stop “edge blur”)
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)  # still (B,1,H,W)

        mask3 = mask.expand(-1, 3, -1, -1)  # (B,3,H,W)

        # Save mask debug images occasionally
        if (self.train_step % self.mask_img_freq) == 0:
            try:
                save_mask_debug_images(
                    obs=obs,
                    obs_recon=obs_recon,
                    mask=mask,
                    out_dir=self.mask_img_dir,
                    step=self.train_step,
                )
                print(f"[VQVAE dbg] saved obs/recon/mask images to '{self.mask_img_dir}' at step {self.train_step}")
            except Exception as e:
                print(f"[VQVAE dbg] failed to save mask debug images: {e}")
        # Compute debug MSEs occasionally
        dbg_mse_agent = None
        dbg_mse_bg = None
        dbg_agent_frac = None
        if (self.train_step % self.agent_dbg_freq) == 0 and (obs.shape == obs_recon.shape):
            sqerr_dbg = (obs_recon - obs).pow(2)
            agent_pixels = mask3.sum().clamp_min(1.0)
            bg_pixels = (1.0 - mask3).sum().clamp_min(1.0)

            dbg_mse_agent = (sqerr_dbg * mask3).sum() / agent_pixels
            dbg_mse_bg = (sqerr_dbg * (1.0 - mask3)).sum() / bg_pixels
            dbg_agent_frac = mask.mean().item()

            print(f"[VQVAE dbg] step={self.train_step} agent_frac={dbg_agent_frac:.6f} "
                  f"mse_agent={dbg_mse_agent.item():.6f} mse_bg={dbg_mse_bg.item():.6f}")

        # -------------------------
        # NEW: weighted recon loss
        # -------------------------
        # Standard squared error
        sq = (obs - obs_recon).pow(2)  # (B,3,H,W)

        # Upweight agent pixels so the tiny sprite matters in the loss
        alpha = 200.0  # try 10, 20, 30; start with 30 for MiniGrid agent_frac ~0.0027
        weights = 1.0 + alpha * mask3
        sq = sq * weights

        # Optional clipping (kept compatible with your previous behavior)
        if self.recon_loss_clip > 0:
            sq = torch.max(sq, torch.tensor(self.recon_loss_clip, device=device))
        if (self.train_step % self.agent_dbg_freq) == 0:
            # weighted average squared error in agent vs bg regions
            w_agent = (sq * mask3).sum() / (mask3.sum().clamp_min(1.0))
            w_bg = (sq * (1.0 - mask3)).sum() / ((1.0 - mask3).sum().clamp_min(1.0))
            print(
                f"[VQVAE dbg] weighted_mse_agent={w_agent.item():.6f} weighted_mse_bg={w_bg.item():.6f} alpha={alpha}")

        # Sum over pixels, mean over batch (same aggregation style as before)
        recon_loss = sq.reshape(sq.shape[0], -1).sum(-1).mean()
        loss_dict["recon_loss"] = recon_loss

        # -------------------------
        # Stats (optional)
        # -------------------------
        if return_stats:
            stats = {}

            if dbg_mse_agent is not None:
                stats["dbg_mse_agent"] = dbg_mse_agent.detach()
            if dbg_mse_bg is not None:
                stats["dbg_mse_bg"] = dbg_mse_bg.detach()
            if dbg_agent_frac is not None:
                stats["dbg_agent_frac"] = torch.tensor(dbg_agent_frac, device=device)

            # Codebook usage stats (best-effort)
            if hasattr(self.model, "quantizer") and hasattr(self.model.quantizer, "embeddings"):
                try:
                    if oh_encodings is not None:
                        codebook_usage = oh_encodings.sum(dim=0)
                        if len(codebook_usage.shape) > 1:
                            codebook_usage = codebook_usage.sum(dim=tuple(range(1, len(codebook_usage.shape))))

                        total_usage = codebook_usage.sum()
                        active_codes = (codebook_usage > 0).sum()

                        stats["codebook_active_codes"] = active_codes.float()
                        stats["codebook_total_usage"] = total_usage.float()
                        stats["codebook_max_usage"] = codebook_usage.max().float()
                        stats["codebook_min_usage"] = codebook_usage.min().float()
                        stats["codebook_usage_entropy"] = -torch.sum(
                            (codebook_usage / (total_usage + 1e-8)) *
                            torch.log(codebook_usage / (total_usage + 1e-8) + 1e-8)
                        )
                except Exception as e:
                    print(f"[VQVAE dbg] Error calculating codebook stats: {e}")

            if perplexity is not None:
                stats["perplexity"] = perplexity

            return loss_dict, stats

        return loss_dict

    def train(self, batch_data):
        loss_dict, stats = self.calculate_losses(batch_data, return_stats=True)
        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'AE train step {self.train_step} | Loss: {loss.item():.4f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.4f}'
            print(log_str)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.train_step += 1

        return loss_dict, stats


class VAETrainer(BaseRepresentationLearner):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 256,
        update_freq: int = 128,
        log_freq: int = 100,
        lr: float = 3e-4,
        recon_loss_clip: float = 0,
        grad_clip: float = 0,
    ):
        super().__init__(model, batch_size, update_freq, log_freq)
        self.model = model
        self.recon_loss_clip = recon_loss_clip
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_step = 0

    def _init_model(self):
        raise Exception('VAE requires a model to be specified!')

    def calculate_losses(self, batch_data):
        device = next(self.model.parameters()).device
        sample_size = int(batch_data[0].shape[0] / 2)
        obs = torch.cat([batch_data[0][:sample_size], batch_data[2][sample_size:]], dim=0).to(device)
        obs_recon, mu, sigma = self.model(obs, return_all=True)

        kl_div = 0.5 * (1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        kl_div = kl_div.view(kl_div.shape[0], -1).sum(-1)

        recon_loss = (obs - obs_recon) ** 2
        if self.recon_loss_clip > 0:
            recon_loss = torch.max(recon_loss, torch.tensor(self.recon_loss_clip, device=device))
        recon_loss = recon_loss.reshape(recon_loss.shape[0], -1).sum(-1)

        losses = -kl_div + recon_loss
        return losses

    def train(self, batch_data):
        losses = self.calculate_losses(batch_data)
        loss = losses.mean()

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            print(f'VAE train step {self.train_step} | Loss: {loss.item():.4f}')

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.train_step += 1

        vis_loss = loss.item() - self.recon_loss_clip * np.prod(batch_data[0].shape[1:])
        return vis_loss, {}


class VQVAETrainer(BaseRepresentationLearner):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 256,
        update_freq: int = 128,
        log_freq: int = 100,
        lr: float = 3e-4,
        recon_loss_clip: float = 0,
        grad_clip: float = 0,
    ):
        super().__init__(model, batch_size, update_freq, log_freq)
        self.model = model
        self.recon_loss_clip = recon_loss_clip
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_step = 0
        self.mi_coefs = torch.linspace(0, 0.002, 2000)

        # Debug controls
        self.agent_dbg_freq = 200
        self.mask_img_freq = 1000
        self.mask_img_dir = "mask_debug"

    def _init_model(self):
        raise Exception('VQVAE requires a model to be specified!')

    def calculate_losses(self, batch_data, return_stats: bool = False):
        device = next(self.model.parameters()).device
        sample_size = int(batch_data[0].shape[0] / 2)

        # Same sampling strategy as your existing code
        obs = torch.cat(
            [batch_data[0][:sample_size], batch_data[2][sample_size:]],
            dim=0
        ).to(device)

        loss_dict = {}

        # -------------------------
        # Forward pass
        # -------------------------
        try:
            obs_recon, quantizer_loss, perplexity, oh_encodings = self.model(obs)
            loss_dict["quantizer_loss"] = quantizer_loss
        except Exception as e:
            print(f"[VQVAE] Error in model forward pass: {e}")
            nan = torch.tensor(float("nan"), device=device)
            if return_stats:
                return {"recon_loss": nan, "quantizer_loss": nan}, {}
            return {"recon_loss": nan, "quantizer_loss": nan}

        # -------------------------
        # Handle spatial mismatch
        # -------------------------
        if obs.shape != obs_recon.shape:
            # IMPORTANT: nearest preserves crisp pixel-art much better than bilinear
            print(
                f"[VQVAE warn] Dimension mismatch: input {obs.shape} vs recon {obs_recon.shape} -> resizing (nearest)")
            obs_recon = F.interpolate(
                obs_recon,
                size=obs.shape[-2:],
                mode="nearest"
            )

        # -------------------------
        # Build agent mask (always)
        # -------------------------
        # mask: (B,1,H,W) float {0,1}
        mask = red_agent_mask(obs).float()

        # Optional dilation to include sprite edges (helps avoid blurry outline)
        # Keep it on; it is cheap and usually beneficial for tiny agents.
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

        mask3 = mask.expand(-1, 3, -1, -1)  # (B,3,H,W)

        # -------------------------
        # Save debug images (independent)
        # -------------------------
        if (self.train_step % self.mask_img_freq) == 0:
            try:
                save_mask_debug_images(
                    obs=obs,
                    obs_recon=obs_recon,
                    mask=mask,
                    out_dir=self.mask_img_dir,
                    step=self.train_step,
                )
                print(f"[VQVAE dbg] saved obs/recon/mask images to '{self.mask_img_dir}' at step {self.train_step}")
            except Exception as e:
                print(f"[VQVAE dbg] failed to save mask debug images: {e}")

        # -------------------------
        # Debug MSE agent vs bg
        # -------------------------
        dbg_mse_agent = None
        dbg_mse_bg = None
        dbg_agent_frac = None
        if (self.train_step % self.agent_dbg_freq) == 0:
            sqerr_dbg = (obs_recon - obs).pow(2)
            agent_pixels = mask3.sum().clamp_min(1.0)
            bg_pixels = (1.0 - mask3).sum().clamp_min(1.0)

            dbg_mse_agent = (sqerr_dbg * mask3).sum() / agent_pixels
            dbg_mse_bg = (sqerr_dbg * (1.0 - mask3)).sum() / bg_pixels
            dbg_agent_frac = float(mask.mean().item())

            print(
                f"[VQVAE dbg] step={self.train_step} agent_frac={dbg_agent_frac:.6f} "
                f"mse_agent={dbg_mse_agent.item():.6f} mse_bg={dbg_mse_bg.item():.6f}"
            )

        # -------------------------
        # Weighted reconstruction loss
        # -------------------------
        # Standard squared error per pixel/channel
        sq = (obs - obs_recon).pow(2)  # (B,3,H,W)

        # Upweight agent pixels
        # Your agent_frac ~0.0027; alpha needs to be large-ish to matter.
        # Start at 50-200. If you see agent still blurry, increase.
        alpha = 200.0
        weights = 1.0 + alpha * mask3
        sq = sq * weights

        # Optional: print weighted region losses occasionally (helps sanity check)
        if (self.train_step % self.agent_dbg_freq) == 0:
            w_agent = (sq * mask3).sum() / (mask3.sum().clamp_min(1.0))
            w_bg = (sq * (1.0 - mask3)).sum() / ((1.0 - mask3).sum().clamp_min(1.0))
            print(
                f"[VQVAE dbg] weighted_mse_agent={w_agent.item():.6f} weighted_mse_bg={w_bg.item():.6f} alpha={alpha}"
            )

        # Clip handling (NOTE: your previous code used torch.max which actually sets a *minimum*;
        # keep the same behavior for compatibility, though naming is misleading.)
        if self.recon_loss_clip > 0:
            sq = torch.max(sq, torch.tensor(self.recon_loss_clip, device=device))

        # Aggregate: sum pixels, mean batch (keeps your original style)
        recon_loss = sq.reshape(sq.shape[0], -1).sum(-1).mean()
        loss_dict["recon_loss"] = recon_loss

        # -------------------------
        # Stats (optional)
        # -------------------------
        if return_stats:
            stats = {}

            if dbg_mse_agent is not None:
                stats["dbg_mse_agent"] = dbg_mse_agent.detach()
            if dbg_mse_bg is not None:
                stats["dbg_mse_bg"] = dbg_mse_bg.detach()
            if dbg_agent_frac is not None:
                stats["dbg_agent_frac"] = torch.tensor(dbg_agent_frac, device=device)

            # Codebook usage stats (best-effort)
            if oh_encodings is not None:
                try:
                    codebook_usage = oh_encodings.sum(dim=0)
                    if len(codebook_usage.shape) > 1:
                        codebook_usage = codebook_usage.sum(dim=tuple(range(1, len(codebook_usage.shape))))

                    total_usage = codebook_usage.sum()
                    active_codes = (codebook_usage > 0).sum()

                    stats["codebook_active_codes"] = active_codes.float()
                    stats["codebook_total_usage"] = total_usage.float()
                    stats["codebook_max_usage"] = codebook_usage.max().float()
                    stats["codebook_min_usage"] = codebook_usage.min().float()
                    stats["codebook_usage_entropy"] = -torch.sum(
                        (codebook_usage / (total_usage + 1e-8)) *
                        torch.log(codebook_usage / (total_usage + 1e-8) + 1e-8)
                    )
                except Exception as e:
                    print(f"[VQVAE dbg] Error calculating codebook stats: {e}")

            if perplexity is not None:
                stats["perplexity"] = perplexity

            return loss_dict, stats

        return loss_dict

    def train(self, batch_data):
        loss_dict, stats = self.calculate_losses(batch_data, return_stats=True)

        # Stop on NaN/Inf losses
        nan_losses = []
        for loss_name, loss_value in loss_dict.items():
            if torch.isnan(loss_value).any() or torch.isinf(loss_value).any():
                nan_losses.append(loss_name)
                try:
                    print(f" NaN/Inf detected in {loss_name}: {loss_value.item():.6f}")
                except Exception:
                    print(f" NaN/Inf detected in {loss_name}")
        if nan_losses:
            print(f" Stopping training due to NaN/Inf in losses: {nan_losses}")
            return loss_dict, stats

        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'VQVAE train step {self.train_step} | Total Loss: {loss.item():.6f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.6f}'
            print(log_str)

        self.optimizer.zero_grad()
        try:
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        except Exception as e:
            print(f" Error during backward pass or optimization: {e}")
            return loss_dict, stats

        self.train_step += 1
        return loss_dict, stats


class DiscreteTransitionTrainer:
    def __init__(
        self,
        transition_model: nn.Module,
        encoder: nn.Module,
        log_freq: int = 100,
        log_norms: bool = False,
        lr: float = 1e-3,
        incl_encoder: bool = False,
        grad_clip: float = 0,
    ):
        self.model = transition_model
        self.encoder = encoder
        self.log_freq = log_freq
        self.log_norms = log_norms
        self.train_step = 0
        self.default_gamma = 0.99
        self.grad_clip = grad_clip
        self.incl_encoder = incl_encoder

        if self.incl_encoder:
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if self.log_norms:
            self.activation_recorder = ActivationRecorder(get_main_trans_activations(self.model))

    def _init_model(self):
        raise Exception('DiscreteTransitionTrainer requires a model to be specified!')

    def calculate_accuracy(self, batch_data):
        device = next(self.model.parameters()).device
        obs = batch_data[0].to(device)
        acts = batch_data[1].to(device)
        next_obs = batch_data[2].to(device)

        with torch.no_grad():
            encodings = self.encoder.encode(obs)
            next_encodings = self.encoder.encode(next_obs)
        pred_next_encodings = self.model(encodings, acts)
        accuracy = (next_encodings == pred_next_encodings).float().mean()
        return accuracy

    def calculate_losses(self, batch_data, n=1):
        if n == 1:
            batch_data = [x.unsqueeze(1) for x in batch_data]
        assert batch_data[0].shape[1] == n, 'n steps does not match batch size!'

        device = next(self.model.parameters()).device
        initial_obs = batch_data[0][:, 0].to(device)

        encodings = self.encoder.encode(initial_obs)
        if not self.incl_encoder:
            encodings = encodings.detach()

        losses = OrderedDict()
        batch_size = initial_obs.shape[0]
        loss_mask = torch.ones(batch_size, device=device, requires_grad=False)

        for i in range(n):
            acts = batch_data[1][:, i].to(device)
            next_obs = batch_data[2][:, i].to(device)
            rewards = batch_data[3][:, i].to(device)
            dones = batch_data[4][:, i].to(device)
            gammas = (1 - dones.float()) * self.default_gamma

            with torch.no_grad():
                next_encodings = self.encoder.encode(next_obs)

            oh_outcomes = None
            if self.model.stochastic == 'categorical':
                oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
                if len(batch_data) > 5:
                    target_outcomes = batch_data[5][:, i].long().to(device)
                    state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
                    losses[f'{i + 1}_step_state_disc_loss'] = state_disc_loss.masked_select(
                        loss_mask.bool()).mean()
                    oh_outcomes = oh_outcomes.detach()

            next_logits_pred, reward_preds, gamma_preds, stoch_logits = self.model(
                encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

            if self.model.stochastic == 'categorical':
                stoch_probs = F.softmax(stoch_logits, dim=1)
                outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
                losses[f'{i + 1}_step_outcome_loss'] = outcome_losses.masked_select(loss_mask[:, None].bool()).mean()

            state_loss = F.cross_entropy(next_logits_pred, next_encodings, reduction='none')
            state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
            state_loss = state_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_state_loss'] = state_loss.mean()

            reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
            reward_loss = reward_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_reward_loss'] = reward_loss.mean()

            gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
            gamma_loss = gamma_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_gamma_loss'] = gamma_loss.mean()

            with torch.no_grad():
                mask_changes = dones.float().nonzero().squeeze()
                loss_mask.scatter_(0, mask_changes, 0)

            encodings = next_logits_pred.argmax(dim=1).detach()

        return losses

    def train(self, batch_data, n=1):
        loss_dict = self.calculate_losses(batch_data, n)
        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'DTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.4f}'
            print(log_str)

        self.train_step += 1
        self.optimizer.zero_grad()

        if self.log_norms:
            norm_data = record_trans_model_update(
                self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
        else:
            norm_data = {}
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss_dict, norm_data


class UniversalVQTransitionTrainer:
    def __init__(
        self,
        transition_model: nn.Module,
        encoder: nn.Module,
        log_freq: int = 100,
        log_norms: bool = False,
        lr: float = 1e-3,
        incl_encoder: bool = False,
        loss_type: str = 'cross_entropy',
        grad_clip: float = 0,
    ):
        self.model = transition_model
        self.encoder = encoder
        self.log_freq = log_freq
        self.log_norms = log_norms
        self.train_step = 0
        self.default_gamma = 0.99
        self.grad_clip = grad_clip
        self.incl_encoder = incl_encoder
        self.use_rand_mask = getattr(transition_model, 'rand_mask', None) is not None

        if self.incl_encoder:
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if loss_type == 'cross_entropy':
            assert encoder.encoder_type == 'vqvae', 'Cross entropy loss requires a VQVAE encoder!'
            self.loss_fn = F.cross_entropy
        elif loss_type == 'mse':
            self.loss_fn = F.mse_loss
        else:
            raise Exception(f'Unknown loss type: {loss_type}')

        if self.log_norms:
            self.activation_recorder = ActivationRecorder(get_main_trans_activations(self.model))

    def _init_model(self):
        raise Exception('DiscreteTransitionTrainer requires a model to be specified!')

    def calculate_accuracy(self, batch_data):
        device = next(self.model.parameters()).device
        obs = batch_data[0].to(device)
        acts = batch_data[1].to(device)
        next_obs = batch_data[2].to(device)

        with torch.no_grad():
            encodings = self.encoder.encode(obs)
            next_encodings = self.encoder.encode(next_obs)
        pred_next_encodings = self.model(encodings, acts)
        comparison = next_encodings == pred_next_encodings
        comparison = comparison.view(comparison.shape[0], -1).all(dim=1)
        return comparison.float().mean()

    def calculate_losses(self, batch_data, n=1):
        if n == 1:
            batch_data = [x.unsqueeze(1) for x in batch_data]
        assert batch_data[0].shape[1] == n, 'n steps does not match batch size!'

        device = next(self.model.parameters()).device
        initial_obs = batch_data[0][:, 0].to(device)

        encodings = self.encoder.encode(initial_obs)
        if not self.incl_encoder:
            encodings = encodings.detach()

        losses = OrderedDict()
        batch_size = initial_obs.shape[0]
        loss_mask = torch.ones(batch_size, device=device, requires_grad=False)

        for i in range(n):
            acts = batch_data[1][:, i].to(device)
            next_obs = batch_data[2][:, i].to(device)
            rewards = batch_data[3][:, i].to(device)
            dones = batch_data[4][:, i].to(device)
            gammas = (1 - dones.float()) * self.default_gamma

            with torch.no_grad():
                next_encodings = self.encoder.encode(next_obs)

            oh_outcomes = None
            if self.model.stochastic == 'categorical':
                oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
                if len(batch_data) > 5:
                    target_outcomes = batch_data[5][:, i].long().to(device)
                    state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
                    losses[f'{i + 1}_step_state_disc_loss'] = state_disc_loss.masked_select(
                        loss_mask.bool()).mean()
                    oh_outcomes = oh_outcomes.detach()

            next_logits_pred, reward_preds, gamma_preds, stoch_logits = self.model(
                encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

            if self.model.stochastic == 'categorical':
                stoch_probs = F.softmax(stoch_logits, dim=1)
                outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
                losses[f'{i + 1}_step_outcome_loss'] = outcome_losses.masked_select(loss_mask[:, None].bool()).mean()

            if self.loss_fn == F.cross_entropy:
                state_loss = F.cross_entropy(next_logits_pred, next_encodings, reduction='none')
            else:
                if next_encodings.dtype == torch.long:
                    next_encodings_oh = F.one_hot(next_encodings, num_classes=next_logits_pred.shape[1])
                    next_encodings_oh = rearrange(next_encodings_oh, 'b ... c -> b c ...').float()
                else:
                    next_encodings_oh = next_encodings

                if self.use_rand_mask:
                    next_encodings_oh = next_encodings_oh * self.model.rand_mask[None]
                state_loss = F.mse_loss(next_logits_pred, next_encodings_oh, reduction='none')

            state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
            state_loss = state_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_state_loss'] = state_loss.mean()

            reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
            reward_loss = reward_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_reward_loss'] = reward_loss.mean()

            gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
            gamma_loss = gamma_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_gamma_loss'] = gamma_loss.mean()

            with torch.no_grad():
                mask_changes = dones.float().nonzero().squeeze()
                loss_mask.scatter_(0, mask_changes, 0)
                encodings = self.model.logits_to_state(next_logits_pred.detach())

        return losses

    def train(self, batch_data, n=1):
        loss_dict = self.calculate_losses(batch_data, n)
        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'Universal VQ Trans Model train step {self.train_step} | Loss: {loss.item():.4f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.4f}'
            print(log_str)

        self.train_step += 1
        self.optimizer.zero_grad()

        if self.log_norms:
            norm_data = record_trans_model_update(
                self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
        else:
            norm_data = {}
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss_dict, norm_data


class ContinuousTransitionTrainer:
    def __init__(
        self,
        transition_model: nn.Module,
        encoder: nn.Module,
        log_freq: int = 100,
        log_norms: bool = False,
        lr: float = 1e-3,
        grad_clip: float = 0,
        e2e_loss: bool = False,
    ):
        self.model = transition_model
        self.encoder = encoder
        self.log_freq = log_freq
        self.log_norms = log_norms
        self.train_step = 0
        self.default_gamma = 0.99
        self.grad_clip = grad_clip
        self.e2e_loss = e2e_loss

        if self.e2e_loss:
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.encoder.parameters()), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if self.log_norms:
            self.activation_recorder = ActivationRecorder(get_main_trans_activations(self.model))

    def calculate_losses(self, batch_data, n=1):
        if n == 1:
            batch_data = [x.unsqueeze(1) for x in batch_data]

        device = next(self.model.parameters()).device
        initial_obs = batch_data[0][:, 0].to(device)

        encodings = self.encoder.encode(initial_obs)
        if not self.e2e_loss:
            encodings = encodings.detach()

        losses = OrderedDict()
        batch_size = initial_obs.shape[0]
        loss_mask = torch.ones(batch_size, device=device, requires_grad=False)

        for i in range(n):
            acts = batch_data[1][:, i].to(device)
            next_obs = batch_data[2][:, i].to(device)
            rewards = batch_data[3][:, i].to(device)
            dones = batch_data[4][:, i].to(device)
            gammas = (1 - dones.float()) * self.default_gamma

            with torch.no_grad():
                next_encodings = self.encoder.encode(next_obs, as_long=False)

            next_encodings = next_encodings.reshape(next_obs.shape[0], self.encoder.latent_dim)

            oh_outcomes = None
            if self.model.stochastic == 'categorical':
                oh_outcomes, outcome_logits = self.model.discretize(next_encodings, return_logits=True)
                if len(batch_data) > 5:
                    target_outcomes = batch_data[5][:, i].long().to(device)
                    state_disc_loss = F.cross_entropy(outcome_logits, target_outcomes, reduction='none')
                    losses[f'{i + 1}_step_state_disc_loss'] = state_disc_loss.masked_select(
                        loss_mask.bool()).mean()
                    oh_outcomes = oh_outcomes.detach()

            next_encodings_pred, reward_preds, gamma_preds, stoch_logits = self.model(
                encodings, acts, oh_outcomes=oh_outcomes, return_logits=True, return_stoch_logits=True)

            if self.model.stochastic == 'categorical':
                stoch_probs = F.softmax(stoch_logits, dim=1)
                outcome_losses = one_hot_cross_entropy(stoch_probs, oh_outcomes.detach())
                losses[f'{i + 1}_step_outcome_loss'] = outcome_losses.masked_select(
                    loss_mask[:, None].bool()).mean()

            if self.e2e_loss:
                obs_recon = self.encoder.decode(next_encodings_pred)
                recon_loss = F.mse_loss(next_obs, obs_recon, reduction='none').reshape(next_obs.shape[0], -1)
                losses[f'{i + 1}_step_recon_loss'] = recon_loss.masked_select(
                    loss_mask[:, None].bool()).reshape(-1, recon_loss.shape[1]).sum(dim=1).mean()

            state_loss = F.mse_loss(
                next_encodings_pred.view(next_obs.shape[0], self.encoder.latent_dim),
                next_encodings,
                reduction='none'
            )
            losses[f'{i + 1}_step_state_loss'] = state_loss.masked_select(loss_mask[:, None].bool()).mean()
            if self.e2e_loss:
                losses[f'{i + 1}_step_state_loss'] = losses[f'{i + 1}_step_state_loss'].detach()

            reward_loss = F.mse_loss(reward_preds.squeeze(), rewards, reduction='none')
            losses[f'{i + 1}_step_reward_loss'] = reward_loss.masked_select(loss_mask.bool()).mean()

            gamma_loss = F.mse_loss(gamma_preds.squeeze(), gammas, reduction='none')
            losses[f'{i + 1}_step_gamma_loss'] = gamma_loss.masked_select(loss_mask.bool()).mean()

            mask_changes = dones.float().nonzero().squeeze()
            loss_mask.scatter_(0, mask_changes, 0)
            encodings = self.model.logits_to_state(next_encodings_pred.detach())
            if not self.e2e_loss:
                encodings = encodings.detach()

        return losses

    def train(self, batch_data, n=1):
        loss_dict = self.calculate_losses(batch_data, n)
        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'CTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.4f}'
            print(log_str)

        self.train_step += 1
        self.optimizer.zero_grad()

        if self.log_norms:
            norm_data = record_trans_model_update(
                self.model, loss, self.optimizer, self.activation_recorder.reset(), self.grad_clip)
        else:
            norm_data = {}
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss_dict, norm_data


class TransformerTransitionTrainer:
    def __init__(
        self,
        transition_model: nn.Module,
        encoder: nn.Module,
        log_freq: int = 100,
        lr: float = 1e-3,
        grad_clip: float = 0,
    ):
        self.model = transition_model
        self.encoder = encoder
        self.log_freq = log_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_step = 0
        self.default_gamma = 0.99
        self.grad_clip = grad_clip

    def calculate_losses(self, batch_data, n=1):
        """
        Args:
            batch_data: List, of shape [5, batch_size, n_steps, ...]
            n: int, number of steps to train on
        """
        if n == 1:
            batch_data = [x.unsqueeze(1) for x in batch_data]

        device = next(self.model.parameters()).device

        initial_obs = batch_data[0][:, 0].to(device)
        with torch.no_grad():
            encodings = self.encoder.encode(initial_obs)

        losses = OrderedDict()
        batch_size = initial_obs.shape[0]
        loss_mask = torch.ones(batch_size, device=device, requires_grad=False)

        for i in range(n):
            acts = batch_data[1][:, i].to(device)
            next_obs = batch_data[2][:, i].to(device)
            rewards = batch_data[3][:, i].to(device)
            dones = batch_data[4][:, i].to(device)
            gammas = (1 - dones.float()) * self.default_gamma

            with torch.no_grad():
                next_encodings = self.encoder.encode(next_obs)

            sequence_length = next_encodings.shape[1]
            if self.model.model_type.lower() == 'transformer':
                mask = self.model.get_tgt_mask(sequence_length).to(device)
            elif self.model.model_type.lower() == 'transformerdec':
                mask = self.model.get_tgt_mask(encodings.shape[1] + 1, sequence_length).to(device)
            else:
                raise ValueError(f'Unknown model type: {self.model.model_type}')

            if self.model.training:
                next_logits_pred, reward_preds, gamma_preds = self.model(
                    encodings, acts, next_encodings, tgt_mask=mask, return_logits=True)
            else:
                next_logits_pred, reward_preds, gamma_preds = self.model(
                    encodings, acts, return_logits=True)

            state_loss = F.cross_entropy(
                next_logits_pred.permute(0, 2, 1), next_encodings, reduction='none')

            state_loss = state_loss.view(state_loss.shape[0], -1).sum(dim=1)
            state_loss = state_loss.masked_select(loss_mask.bool())
            losses[f'{i + 1}_step_state_loss'] = state_loss.mean()

            with torch.no_grad():
                mask_changes = dones.float().nonzero().squeeze()
                loss_mask.scatter_(0, mask_changes, 0)

            encodings = next_logits_pred.argmax(dim=2).detach()

        return losses

    def train(self, batch_data, n=1):
        loss_dict = self.calculate_losses(batch_data, n)
        loss = torch.sum(torch.stack(tuple(loss_dict.values())))

        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'TransformerTransModel train step {self.train_step} | Loss: {loss.item():.4f}'
            for loss_name, loss_value in loss_dict.items():
                log_str += f' | {loss_name}: {loss_value.item():.4f}'
            print(log_str)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.train_step += 1

        return loss_dict
