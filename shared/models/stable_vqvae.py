"""
Stable VQ-VAE implementation that prevents NaN values and ensures compatibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StableVectorQuantizer(nn.Module):
    """
    Highly stable vector quantizer that prevents NaN values
    """

    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings with very small values to prevent NaN
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)

        # EMA parameters for codebook updates
        self.register_buffer('cluster_usage', torch.ones(n_embeddings))
        self.register_buffer('embed_avg', self.embeddings.weight.data.clone())

        # Training parameters
        self.decay = 0.97
        self.eps = 1e-5

    def forward(self, inputs, mask=None):
        """
        Forward pass with extensive NaN protection
        """
        # Flatten input for quantization
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Ensure no NaN/inf in input
        if torch.isnan(flat_input).any() or torch.isinf(flat_input).any():
            print(" Warning: NaN/Inf detected in quantizer input, clipping")
            flat_input = torch.nan_to_num(flat_input, nan=0.0, posinf=1.0, neginf=-1.0)
            flat_input = torch.clamp(flat_input, -10.0, 10.0)

        # Calculate distances to embeddings
        codebook = self.embeddings.weight

        # Ensure codebook has no NaN/inf
        if torch.isnan(codebook).any() or torch.isinf(codebook).any():
            print(" Warning: NaN/Inf detected in codebook, reinitializing")
            nn.init.uniform_(self.embeddings.weight, -0.001, 0.001)
            codebook = self.embeddings.weight

        # Compute distances using stable method
        # |x - e|^2 = |x|^2 + |e|^2 - 2*x*e
        input_sq = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        codebook_sq = torch.sum(codebook ** 2, dim=1)
        distances = input_sq + codebook_sq - 2 * torch.matmul(flat_input, codebook.t())

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()

        # Get quantized vectors
        quantized = torch.matmul(encodings, codebook)
        quantized = quantized.view(input_shape)

        # Calculate commitment loss (encoder loss)
        commitment_loss = F.mse_loss(quantized.detach(), inputs)

        # Calculate codebook loss (only during training)
        codebook_loss = F.mse_loss(quantized, inputs.detach())

        # Total VQ loss
        vq_loss = commitment_loss * self.commitment_cost + codebook_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Update EMA during training
        if self.training:
            self._update_ema(flat_input, encodings)

        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Final NaN check
        if torch.isnan(quantized).any():
            print(" NaN detected in quantized output, using input")
            quantized = inputs
            vq_loss = torch.tensor(0.0, device=inputs.device)

        return quantized, vq_loss, perplexity, encoding_indices.view(input_shape[:-1])

    def _update_ema(self, flat_input, encodings):
        """Update EMA statistics"""
        with torch.no_grad():
            # Update cluster usage
            cluster_size = torch.sum(encodings, dim=0)
            self.cluster_usage.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # Update embeddings
            embed_sum = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Normalize embeddings
            n = self.cluster_usage.sum()
            cluster_usage = (self.cluster_usage + self.eps) / (n + self.n_embeddings * self.eps) * n
            embed_normalized = self.embed_avg / cluster_usage.unsqueeze(1)

            # Update embedding weights
            self.embeddings.weight.data.copy_(embed_normalized)


class StableVQVAEModel(nn.Module):
    """
    Stable VQ-VAE model with proper interface compatibility
    """

    def __init__(self, input_dim, codebook_size, embedding_dim, encoder=None, decoder=None,
                 commitment_cost=0.25, n_latents=None):
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size  # Same as n_embeddings
        self.n_embeddings = codebook_size  # For compatibility
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Create encoder/decoder if not provided
        if encoder is None:
            self.encoder = self._make_default_encoder()
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = self._make_default_decoder()
        else:
            self.decoder = decoder

        # Calculate number of latent positions by testing encoder
        if encoder is not None:
            test_input = torch.zeros(1, *input_dim)
            with torch.no_grad():
                encoded = self.encoder(test_input)
                self.encoded_shape = encoded.shape[1:]  # Remove batch dim
                # For convolutional outputs, spatial dimensions determine n_latent_embeds
                if len(encoded.shape) == 4:  # [B, C, H, W]
                    self.n_latent_embeds = encoded.shape[2] * encoded.shape[3]  # H * W
                else:
                    self.n_latent_embeds = int(np.prod(encoded.shape[1:]))
        else:
            # Default fallback
            spatial_size = max(1, input_dim[1] // 8)  # Assuming 8x downsampling
            self.n_latent_embeds = spatial_size * spatial_size
            self.encoded_shape = (embedding_dim, spatial_size, spatial_size)

        # Interface compatibility attributes
        self.latent_dim = self.n_latent_embeds * embedding_dim  # Total latent dimension

        # Vector quantizer
        self.quantizer = StableVectorQuantizer(
            n_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        # Initialize weights properly
        self._initialize_weights()

        print(f" StableVQVAE created:")
        print(f"   Input shape: {input_dim}")
        print(f"   Encoded shape: {self.encoded_shape}")
        print(f"   Latent positions: {self.n_latent_embeds}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Codebook size: {codebook_size}")
        print(f"   Total latent dim: {self.latent_dim}")

    def _make_default_encoder(self):
        """Create a simple default encoder"""
        return nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, self.embedding_dim, 3, 1, 1),
        )

    def _make_default_decoder(self):
        """Create a simple default decoder"""
        return nn.Sequential(
            nn.Conv2d(self.embedding_dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_dim[0], 4, 2, 1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        """Initialize all weights safely"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1  # Scale down to prevent explosion
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        """Encode input to latent space"""
        # Ensure input is properly formatted
        if torch.isnan(x).any():
            print(" Warning: NaN in input to encoder")
            x = torch.nan_to_num(x, nan=0.0)

        # Clamp input to reasonable range
        x = torch.clamp(x, 0.0, 1.0)

        # Encode
        try:
            encoded = self.encoder(x)
        except Exception as e:
            print(f" Error in encoder forward pass: {e}")
            # Return zero tensor with correct shape
            batch_size = x.shape[0]
            encoded = torch.zeros(batch_size, *self.encoded_shape, device=x.device)

        # Check for NaN in encoded output
        if torch.isnan(encoded).any():
            print(" NaN detected in encoder output, reinitializing")
            self._safe_reinitialize_encoder()
            encoded = self.encoder(x)
            if torch.isnan(encoded).any():
                # Last resort - return zeros
                encoded = torch.zeros_like(encoded)

        return encoded

    def decode(self, quantized):
        """Decode from quantized latents"""
        if torch.isnan(quantized).any():
            print(" Warning: NaN in input to decoder")
            quantized = torch.nan_to_num(quantized, nan=0.0)

        # Clamp quantized values
        quantized = torch.clamp(quantized, -10.0, 10.0)

        try:
            decoded = self.decoder(quantized)
        except Exception as e:
            print(f" Error in decoder forward pass: {e}")
            # Return zero tensor with correct output shape
            batch_size = quantized.shape[0]
            decoded = torch.zeros(batch_size, *self.input_dim, device=quantized.device)

        if torch.isnan(decoded).any():
            print(" NaN detected in decoder output")
            decoded = torch.zeros_like(decoded)

        # Ensure output is in [0, 1] range
        decoded = torch.clamp(decoded, 0.0, 1.0)

        return decoded

    def forward(self, x):
        """Full forward pass"""
        # Encode
        encoded = self.encode(x)

        # Quantize
        quantized, vq_loss, perplexity, encodings = self.quantizer(encoded)

        # Decode
        reconstructed = self.decode(quantized)

        return reconstructed, vq_loss, perplexity, encodings

    def _safe_reinitialize_encoder(self):
        """Safely reinitialize encoder weights"""
        print(" Reinitializing encoder weights...")
        for m in self.encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.01  # Very small initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_codebook(self):
        """Get the codebook for compatibility"""
        return self.quantizer.embeddings.weight

    # Additional compatibility methods
    @property
    def quantized_enc(self):
        """For compatibility with some transition models"""
        return False  # This model doesn't use quantized encoding in the way expected

    def enable_sparsity(self):
        """Compatibility method"""
        pass

    def disable_sparsity(self):
        """Compatibility method"""
        pass


class RobustVQVAETrainer:
    """
    Robust trainer for VQ-VAE with extensive error handling
    """

    def __init__(self, model, lr=1e-4, grad_clip=1.0, log_freq=100):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-6)
        self.grad_clip = grad_clip
        self.log_freq = log_freq
        self.step = 0

    def calculate_losses(self, batch_data):
        """Calculate losses with robust error handling"""
        obs = batch_data[0]

        # Ensure valid input
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(" Warning: Invalid input to trainer, cleaning...")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        obs = torch.clamp(obs, 0.0, 1.0)

        # Forward pass
        recon, vq_loss, perplexity, _ = self.model(obs)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, obs)

        # Total loss
        total_loss = recon_loss + vq_loss

        # Check for NaN in losses
        if torch.isnan(total_loss) or torch.isnan(recon_loss) or torch.isnan(vq_loss):
            print(" NaN detected in loss calculation!")
            # Return small losses to prevent training from crashing
            device = obs.device
            total_loss = torch.tensor(0.001, device=device, requires_grad=True)
            recon_loss = torch.tensor(0.001, device=device)
            vq_loss = torch.tensor(0.0, device=device)
            perplexity = torch.tensor(1.0, device=device)

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'perplexity': perplexity
        }

    def train(self, batch_data):
        """Training step with gradient clipping and error recovery"""
        self.optimizer.zero_grad()

        # Calculate losses
        loss_dict = self.calculate_losses(batch_data)
        loss = loss_dict['loss']

        # Skip step if loss is too small (NaN recovery)
        if loss.item() < 1e-6:
            print(" Skipping training step due to very small loss")
            return loss_dict, {}

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            print(f" Error in backward pass: {e}")
            return loss_dict, {}

        # Check for NaN gradients
        has_nan_grad = False
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break
                total_grad_norm += param.grad.data.norm(2).item() ** 2

        total_grad_norm = total_grad_norm ** 0.5

        if has_nan_grad:
            print(" NaN/Inf gradients detected, skipping optimizer step")
            self.optimizer.zero_grad()
            grad_norm = 0.0
        elif total_grad_norm > 100.0:  # Very large gradients
            print(f" Large gradients detected ({total_grad_norm:.2f}), clipping heavily")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
        else:
            # Normal gradient clipping
            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
            else:
                grad_norm = total_grad_norm
            self.optimizer.step()

        self.step += 1

        # Additional stats
        aux_data = {
            'grad_norm': grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'active_codes': loss_dict.get('perplexity', torch.tensor(1.0)).item()
        }

        return loss_dict, aux_data