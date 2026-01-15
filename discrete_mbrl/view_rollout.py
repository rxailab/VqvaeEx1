#!/usr/bin/env python3
"""
Visualize trained RL model rollouts on MiniGrid environments.

Usage:
    python view_rollout.py \
        --env_name MiniGrid-Empty-5x5-v0 \
        --model_path ./models/MiniGrid-Empty-5x5-v0/final_model_reward_0.9804.pt \
        --ae_model_type vqvae \
        --codebook_size 512 \
        --embedding_dim 128 \
        --filter_size 9 \
        --n_episodes 5 \
        --device cuda
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discrete_mbrl'))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not found. Install with: pip install Pillow")

try:
    from IPython.display import HTML, display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Import project modules
from discrete_mbrl.env_helpers import make_env, preprocess_obs
from discrete_mbrl.model_construction import construct_ae_model
from discrete_mbrl.training_helpers import get_args, freeze_model, make_argparser
from discrete_mbrl.model_free.rl_utils import interpret_layer_sizes

# Import for building policy network
from shared.models import mlp


def load_model_free_checkpoint(model_path, encoder, args, device):
    """Load the saved model-free checkpoint and reconstruct policy/critic"""

    # Handle PyTorch 2.6+ security changes
    try:
        # Try with weights_only=True first (safer)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Note: Loading with weights_only=False due to: {type(e).__name__}")
        # Fall back to weights_only=False for older checkpoints
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Model info: {checkpoint.get('model_info', 'N/A')}")

    # Get dimensions
    if args.ae_model_type == 'vqvae':
        input_dim = args.embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    # Get action dimension from environment
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n
    env.close()

    # Parse hidden sizes
    policy_hidden = interpret_layer_sizes(getattr(args, 'policy_hidden', [256, 256]))
    critic_hidden = interpret_layer_sizes(getattr(args, 'critic_hidden', [256, 256]))

    # Build MLP kwargs
    mlp_kwargs = {
        'activation': getattr(args, 'rl_activation', 'relu'),
        'discrete_input': args.ae_model_type == 'vqvae',
    }
    if args.ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim

    # Reconstruct policy and critic
    policy = mlp([input_dim] + policy_hidden + [act_dim], **mlp_kwargs)
    critic = mlp([input_dim] + critic_hidden + [1], **mlp_kwargs)

    # Load weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    policy = policy.to(device)
    critic = critic.to(device)
    policy.eval()
    critic.eval()

    return policy, critic


def get_action(policy, state, device, deterministic=True):
    """Get action from policy"""
    with torch.no_grad():
        state_tensor = state.unsqueeze(0).to(device)
        logits = policy(state_tensor)

        if deterministic:
            action = logits.argmax(dim=-1).item()
        else:
            dist = Categorical(logits=logits)
            action = dist.sample().item()

    return action


def run_episode(env, encoder, policy, device, max_steps=1000, render_mode='rgb_array'):
    """Run a single episode and collect frames"""

    frames = []
    rewards = []
    actions = []

    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps:
        # Render frame
        try:
            # Get the underlying minigrid env for rendering
            render_env = env
            while hasattr(render_env, 'env'):
                render_env = render_env.env

            if hasattr(render_env, 'get_full_render'):
                frame = render_env.get_full_render(highlight=True, tile_size=32)
            elif hasattr(render_env, 'render'):
                frame = render_env.render()
            else:
                frame = None

            if frame is not None:
                frames.append(frame)
        except Exception as e:
            if step == 0:
                print(f"Render warning: {e}")

        # Encode observation
        obs_tensor = preprocess_obs([obs])
        with torch.no_grad():
            state = encoder.encode(obs_tensor.to(device), return_one_hot=True)
            state = state.squeeze(0)

        # Get action
        action = get_action(policy, state, device, deterministic=True)
        actions.append(action)

        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        rewards.append(reward)
        total_reward += reward
        step += 1

    return frames, rewards, actions, total_reward, step


def create_video(frames, save_path, fps=5):
    """Create video/GIF from frames"""
    if not frames:
        print("No frames to save!")
        return

    if HAS_PIL:
        # Save as GIF
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            save_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"Saved GIF to: {save_path}")
    else:
        print("PIL not available, cannot save GIF")


def display_episode_matplotlib(frames, rewards, title="Episode Rollout", fps=5):
    """Display episode using matplotlib animation"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for display")
        return

    if not frames:
        print("No frames to display!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Setup frame display
    im = ax1.imshow(frames[0])
    ax1.axis('off')
    ax1.set_title(title)

    # Setup reward plot
    cumulative_rewards = np.cumsum(rewards) if rewards else [0]
    line, = ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlim(0, len(frames))
    ax2.set_ylim(min(0, min(cumulative_rewards) - 0.1), max(cumulative_rewards) + 0.1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Reward Over Time')
    ax2.grid(True, alpha=0.3)

    def init():
        im.set_data(frames[0])
        line.set_data([], [])
        return [im, line]

    def animate(i):
        im.set_data(frames[i])
        if i < len(cumulative_rewards):
            line.set_data(range(i + 1), cumulative_rewards[:i + 1])
        ax1.set_title(f'{title} - Step {i + 1}/{len(frames)}')
        return [im, line]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(frames), interval=1000 // fps, blit=True
    )

    plt.tight_layout()
    plt.show()

    return anim


def display_frames_interactive(frames, rewards, delay=0.2):
    """Display frames interactively one by one"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cumulative_reward = 0
    cumulative_rewards = []

    for i, frame in enumerate(frames):
        ax1.clear()
        ax1.imshow(frame)
        ax1.axis('off')
        ax1.set_title(f'Step {i + 1}/{len(frames)}')

        if i < len(rewards):
            cumulative_reward += rewards[i]
        cumulative_rewards.append(cumulative_reward)

        ax2.clear()
        ax2.plot(cumulative_rewards, 'b-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title(f'Total Reward: {cumulative_reward:.4f}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(delay)

    plt.ioff()
    plt.show()


def main():
    parser = make_argparser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--save_gifs', action='store_true',
                        help='Save episodes as GIF files')
    parser.add_argument('--output_dir', type=str, default='./rollout_videos',
                        help='Directory to save GIFs')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second for visualization')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between frames in interactive mode')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--policy_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--critic_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--rl_activation', default='relu')

    args = parser.parse_args()

    print("=" * 60)
    print("Model Rollout Visualization")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.n_episodes}")

    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"\nError: Model not found at {args.model_path}")
        print("\nLooking for models in ./models/:")
        if os.path.exists('./models'):
            for root, dirs, files in os.walk('./models'):
                for f in files:
                    if f.endswith('.pt') or f.endswith('.zip'):
                        print(f"  {os.path.join(root, f)}")
        return

    # Create environment
    print("\nCreating environment...")
    env = make_env(args.env_name, max_steps=args.env_max_steps)

    # Get sample observation
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])

    print(f"Observation shape: {sample_obs.shape[1:]}")
    print(f"Action space: {env.action_space}")

    # Load encoder
    print("\nLoading encoder...")
    encoder, _ = construct_ae_model(sample_obs.shape[1:], args, load=True)
    encoder = encoder.to(args.device)
    freeze_model(encoder)
    encoder.eval()
    print(f"Encoder type: {encoder.encoder_type}")

    # Load policy
    print("\nLoading policy...")
    policy, critic = load_model_free_checkpoint(args.model_path, encoder, args, args.device)
    print("Policy loaded successfully!")

    # Create output directory
    if args.save_gifs:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run episodes
    print("\n" + "=" * 60)
    print("Running Episodes")
    print("=" * 60)

    all_rewards = []
    all_lengths = []

    for ep in range(args.n_episodes):
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} ---")

        frames, rewards, actions, total_reward, steps = run_episode(
            env, encoder, policy, args.device, max_steps=args.max_steps
        )

        all_rewards.append(total_reward)
        all_lengths.append(steps)

        print(f"Total Reward: {total_reward:.4f}")
        print(f"Episode Length: {steps}")
        print(f"Frames captured: {len(frames)}")

        if frames:
            # Save GIF if requested
            if args.save_gifs:
                gif_path = os.path.join(args.output_dir, f"episode_{ep + 1}.gif")
                create_video(frames, gif_path, fps=args.fps)

            # Display episode
            if HAS_MATPLOTLIB:
                print(f"\nDisplaying episode {ep + 1}...")
                print("(Close the window to continue to next episode)")
                display_frames_interactive(frames, rewards, delay=args.delay)
        else:
            print("No frames captured for this episode")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean Reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"Mean Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")

    if args.save_gifs:
        print(f"\nGIFs saved to: {args.output_dir}/")

    env.close()
    print("\nDone!")


if __name__ == '__main__':
    main()