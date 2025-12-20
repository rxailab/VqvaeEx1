import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discrete_mbrl.env_helpers import make_env
from discrete_mbrl.model_construction import construct_ae_model


class SimpleArgs:
    """Simple argument container for model construction"""

    def __init__(self, **kwargs):
        # Default values for VQ-VAE
        self.env_name = kwargs.get('env_name', 'MiniGrid-SimpleCrossingS9N1-v0')
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.ae_model_type = kwargs.get('ae_model_type', 'vqvae')
        self.embedding_dim = kwargs.get('embedding_dim', 128)
        self.filter_size = kwargs.get('filter_size', 8)
        self.ae_model_version = kwargs.get('ae_model_version', '2')
        self.codebook_size = kwargs.get('codebook_size', 1024)
        self.latent_dim = kwargs.get('latent_dim', 81)
        self.ae_grad_clip = 1.0
        self.learning_rate = 0.0003
        # Logging attributes required by model construction
        self.wandb = False
        self.comet_ml = False
        # Additional attributes that might be needed
        self.extra_info = None
        self.repr_sparsity = 0.0
        self.sparsity_type = 'random'
        # Disable param logging during model loading
        self._skip_param_logging = True


def load_model_and_get_observation(model_path, env_name, device='cpu'):
    """Load trained model and get a sample observation"""

    # Create environment and get observation shape
    env = make_env(env_name)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new gym API

    # Convert to tensor and add batch dimension
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    input_shape = obs_tensor.shape[1:]
    print(f"Input shape: {input_shape}")

    # Create model arguments
    args = SimpleArgs(
        env_name=env_name,
        device=device,
        ae_model_type='vqvae',
        embedding_dim=128,
        codebook_size=1024,
        latent_dim=81
    )

    # Construct model
    model, _ = construct_ae_model(input_shape, args, load=False)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, obs_tensor, env


def compare_reconstruction(model, obs_tensor, env_name, device='cpu', num_samples=4):
    """Compare original vs reconstructed observations"""

    model.eval()
    obs_tensor = obs_tensor.to(device)

    with torch.no_grad():
        # Get multiple samples from environment
        reconstructions = []
        originals = []

        for i in range(num_samples):
            # Encode and decode
            if hasattr(model, 'forward'):
                if model.encoder_type == 'vqvae':
                    recon, _, _, _ = model(obs_tensor)
                else:
                    recon = model(obs_tensor)
            else:
                # Fallback to encode/decode
                encoded = model.encode(obs_tensor)
                recon = model.decode(encoded)

            # Convert to numpy and handle different tensor formats
            original = obs_tensor[0].cpu().numpy()
            reconstructed = recon[0].cpu().numpy()

            # Handle different channel orders (CHW vs HWC)
            if original.shape[0] <= 4:  # Channels first
                original = np.transpose(original, (1, 2, 0))
            if reconstructed.shape[0] <= 4:  # Channels first
                reconstructed = np.transpose(reconstructed, (1, 2, 0))

            # Ensure values are in [0, 1] range
            original = np.clip(original, 0, 1)
            reconstructed = np.clip(reconstructed, 0, 1)

            # Handle single channel images
            if original.shape[-1] == 1:
                original = original.squeeze(-1)
            if reconstructed.shape[-1] == 1:
                reconstructed = reconstructed.squeeze(-1)

            originals.append(original)
            reconstructions.append(reconstructed)

            # Get next observation for variety
            if i < num_samples - 1:
                # Create new environment and take random steps to get different scene
                env = make_env(env_name)
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                # Take some random actions to get a different scene
                for _ in range(np.random.randint(1, 15)):
                    action = np.random.randint(0, env.action_space.n)
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        obs, _, done, _ = step_result
                    else:
                        obs, _, done, _, _ = step_result
                    if done:
                        obs = env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    return originals, reconstructions


def plot_comparison(originals, reconstructions, save_path=None):
    """Plot original vs reconstructed images side by side"""

    num_samples = len(originals)
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(originals[i], cmap='viridis' if len(originals[i].shape) == 2 else None)
        axes[0, i].set_title(f'Original {i + 1}', fontsize=12)
        axes[0, i].axis('off')

        # Reconstruction
        axes[1, i].imshow(reconstructions[i], cmap='viridis' if len(reconstructions[i].shape) == 2 else None)
        axes[1, i].set_title(f'Reconstructed {i + 1}', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare original vs reconstructed scenes')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--env_name', type=str, default='MiniGrid-SimpleCrossingS9N1-v0',
                        help='Environment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of scenes to compare')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save comparison image')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    try:
        # Load model and get observations
        model, obs_tensor, env = load_model_and_get_observation(
            args.model_path, args.env_name, device)

        # Compare reconstructions
        originals, reconstructions = compare_reconstruction(
            model, obs_tensor, args.env_name, device, args.num_samples)

        # Plot results
        plot_comparison(originals, reconstructions, args.save_path)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()