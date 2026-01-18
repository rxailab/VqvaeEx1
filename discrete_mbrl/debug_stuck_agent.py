#!/usr/bin/env python3
"""
Debug script to identify why the agent is stuck.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discrete_mbrl'))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from discrete_mbrl.env_helpers import make_env, preprocess_obs
from discrete_mbrl.model_construction import construct_ae_model
from discrete_mbrl.training_helpers import freeze_model, make_argparser
from discrete_mbrl.model_free.rl_utils import interpret_layer_sizes
from shared.models import mlp


def load_checkpoint(model_path, device):
    """Load checkpoint and print info"""
    # Fix for PyTorch 2.6+: explicitly set weights_only=False for checkpoints
    # containing numpy objects. Only use this for trusted checkpoint files.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("\n" + "=" * 60)
    print("CHECKPOINT INFO")
    print("=" * 60)
    print(f"Keys: {list(checkpoint.keys())}")
    print(f"Avg reward: {checkpoint.get('avg_reward', 'N/A')}")
    print(f"Step: {checkpoint.get('step', 'N/A')}")
    print(f"Model info: {checkpoint.get('model_info', 'N/A')}")

    if 'args' in checkpoint:
        print(f"\nSaved args:")
        for k, v in checkpoint['args'].items():
            print(f"  {k}: {v}")

    return checkpoint


def debug_environment(env_name):
    """Debug environment action space"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT DEBUG")
    print("=" * 60)

    env = make_env(env_name)
    print(f"Environment: {env_name}")
    print(f"Action space: {env.action_space}")
    print(f"Action space n: {env.action_space.n}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation shape: {env.observation_space.shape}")

    # Print action meanings for MiniGrid
    print("\nMiniGrid Actions:")
    print("  0: Turn left")
    print("  1: Turn right")
    print("  2: Move forward")
    print("  3: Pick up object")
    print("  4: Drop object")
    print("  5: Toggle/activate object")
    print("  6: Done (not used in most envs)")

    # Test a few steps with different actions
    print("\nTesting actions:")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    for action in range(min(env.action_space.n, 7)):
        env.reset()
        result = env.step(action)
        if len(result) == 5:
            _, reward, term, trunc, info = result
        else:
            _, reward, done, info = result
        print(f"  Action {action}: reward={reward:.4f}")

    env.close()
    return env.action_space.n


def debug_encoder(encoder, env_name, device):
    """Debug encoder output"""
    print("\n" + "=" * 60)
    print("ENCODER DEBUG")
    print("=" * 60)

    env = make_env(env_name)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"Raw observation shape: {obs.shape}")
    print(f"Raw observation dtype: {obs.dtype}")
    print(f"Raw observation range: [{obs.min():.4f}, {obs.max():.4f}]")

    obs_tensor = preprocess_obs([obs])
    print(f"\nPreprocessed observation shape: {obs_tensor.shape}")
    print(f"Preprocessed observation dtype: {obs_tensor.dtype}")

    with torch.no_grad():
        # Test different encoding methods
        encoded = encoder.encode(obs_tensor.to(device))
        print(f"\nDefault encode() output:")
        print(f"  Shape: {encoded.shape}")
        print(f"  Dtype: {encoded.dtype}")
        print(f"  Range: [{encoded.min():.4f}, {encoded.max():.4f}]")

        if hasattr(encoder, 'n_latent_embeds'):
            print(f"  n_latent_embeds: {encoder.n_latent_embeds}")
        if hasattr(encoder, 'n_embeddings'):
            print(f"  n_embeddings: {encoder.n_embeddings}")
        if hasattr(encoder, 'codebook_size'):
            print(f"  codebook_size: {encoder.codebook_size}")

        # Test with return_one_hot=True
        try:
            encoded_oh = encoder.encode(obs_tensor.to(device), return_one_hot=True)
            print(f"\nencode(return_one_hot=True) output:")
            print(f"  Shape: {encoded_oh.shape}")
            print(f"  Dtype: {encoded_oh.dtype}")
            print(f"  Range: [{encoded_oh.min():.4f}, {encoded_oh.max():.4f}]")

            # Check if it's actually one-hot
            if len(encoded_oh.shape) == 3:
                print(f"  Sum along dim 1: {encoded_oh.sum(dim=1)}")  # Should be all 1s if one-hot
        except Exception as e:
            print(f"\nencode(return_one_hot=True) failed: {e}")

        # Test with as_long=True
        try:
            encoded_long = encoder.encode(obs_tensor.to(device), as_long=True)
            print(f"\nencode(as_long=True) output:")
            print(f"  Shape: {encoded_long.shape}")
            print(f"  Dtype: {encoded_long.dtype}")
            print(f"  Values: {encoded_long}")
        except Exception as e:
            print(f"\nencode(as_long=True) failed: {e}")

    env.close()
    return encoded.shape


def debug_policy(checkpoint, encoder, env_name, device):
    """Debug policy network"""
    print("\n" + "=" * 60)
    print("POLICY DEBUG")
    print("=" * 60)

    # Get dimensions
    env = make_env(env_name)
    act_dim = env.action_space.n
    print(f"Action dimension: {act_dim}")

    # Check saved args
    saved_args = checkpoint.get('args', {})
    ae_model_type = saved_args.get('ae_model_type', 'vqvae')
    codebook_size = saved_args.get('codebook_size', 512)
    embedding_dim = saved_args.get('embedding_dim', 128)

    print(f"\nSaved model config:")
    print(f"  ae_model_type: {ae_model_type}")
    print(f"  codebook_size: {codebook_size}")
    print(f"  embedding_dim: {embedding_dim}")

    # Calculate input dimension
    if ae_model_type == 'vqvae':
        input_dim = embedding_dim * encoder.n_latent_embeds
        print(f"  Expected input_dim: {embedding_dim} * {encoder.n_latent_embeds} = {input_dim}")
    else:
        input_dim = encoder.latent_dim
        print(f"  Expected input_dim: {input_dim}")

    # Check policy state dict
    policy_state = checkpoint['policy_state_dict']
    print(f"\nPolicy state dict keys:")
    for k, v in policy_state.items():
        print(f"  {k}: {v.shape}")

    # Find first and last layer shapes
    first_layer = None
    last_layer = None
    for k, v in policy_state.items():
        if 'weight' in k:
            if first_layer is None:
                first_layer = (k, v.shape)
            last_layer = (k, v.shape)

    print(f"\nFirst layer: {first_layer}")
    print(f"Last layer: {last_layer}")

    if first_layer:
        expected_input = first_layer[1][1]
        print(f"\nPolicy expects input of size: {expected_input}")
        print(f"Encoder produces output of size: {input_dim}")
        if expected_input != input_dim:
            print("  ⚠️  MISMATCH! This could be the bug!")

    if last_layer:
        policy_output_dim = last_layer[1][0]
        print(f"\nPolicy output dimension: {policy_output_dim}")
        print(f"Environment action space: {act_dim}")
        if policy_output_dim != act_dim:
            print("  ⚠️  MISMATCH! Policy outputs wrong number of actions!")

    env.close()
    return input_dim, act_dim


def debug_forward_pass(checkpoint, encoder, env_name, device):
    """Debug the full forward pass"""
    print("\n" + "=" * 60)
    print("FORWARD PASS DEBUG")
    print("=" * 60)

    env = make_env(env_name)
    act_dim = env.action_space.n

    # Get saved args
    saved_args = checkpoint.get('args', {})
    ae_model_type = saved_args.get('ae_model_type', 'vqvae')
    codebook_size = saved_args.get('codebook_size', 512)
    embedding_dim = saved_args.get('embedding_dim', 128)
    policy_hidden = saved_args.get('policy_hidden', [256, 256])
    rl_activation = saved_args.get('rl_activation', 'relu')

    # Calculate input dimension
    if ae_model_type == 'vqvae':
        input_dim = embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    # Build policy with same config
    mlp_kwargs = {
        'activation': rl_activation,
        'discrete_input': ae_model_type == 'vqvae',
    }
    if ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = codebook_size
        mlp_kwargs['embed_dim'] = embedding_dim

    if isinstance(policy_hidden, str):
        policy_hidden = eval(policy_hidden)

    print(f"Building policy with:")
    print(f"  input_dim: {input_dim}")
    print(f"  hidden: {policy_hidden}")
    print(f"  output_dim: {act_dim}")
    print(f"  mlp_kwargs: {mlp_kwargs}")

    policy = mlp([input_dim] + list(policy_hidden) + [act_dim], **mlp_kwargs)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy = policy.to(device)
    policy.eval()

    print(f"\nPolicy structure:")
    print(policy)

    # Test forward pass
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    obs_tensor = preprocess_obs([obs])

    with torch.no_grad():
        # Encode
        state = encoder.encode(obs_tensor.to(device), return_one_hot=True)
        print(f"\nEncoded state shape: {state.shape}")
        print(f"Encoded state dtype: {state.dtype}")

        # Forward through policy
        try:
            logits = policy(state)
            print(f"\nPolicy output (logits):")
            print(f"  Shape: {logits.shape}")
            print(f"  Values: {logits}")

            probs = torch.softmax(logits, dim=-1)
            print(f"\nAction probabilities:")
            print(f"  {probs}")

            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            print(f"\nSelected action: {action}")

            # Check if always selecting same action
            print("\n--- Testing multiple steps ---")
            for step in range(10):
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, term, trunc, info = result
                    done = term or trunc
                else:
                    obs, reward, done, info = result

                obs_tensor = preprocess_obs([obs])
                state = encoder.encode(obs_tensor.to(device), return_one_hot=True)
                logits = policy(state)

                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

                print(f"  Step {step + 1}: action={action}, probs={probs.cpu().numpy().round(3)}, reward={reward:.4f}")

                if done:
                    print(f"  Episode done!")
                    break

        except Exception as e:
            print(f"\nForward pass failed: {e}")
            import traceback
            traceback.print_exc()

    env.close()


def main():
    parser = make_argparser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--policy_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--critic_hidden', nargs='*', default=[256, 256])
    parser.add_argument('--rl_activation', default='relu')
    args = parser.parse_args()

    print("=" * 60)
    print("DEBUGGING STUCK AGENT")
    print("=" * 60)

    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, args.device)

    # Debug environment
    n_actions = debug_environment(args.env_name)

    # Create and debug encoder
    env = make_env(args.env_name)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    sample_obs = preprocess_obs([obs])
    env.close()

    encoder, _ = construct_ae_model(sample_obs.shape[1:], args, load=True)
    encoder = encoder.to(args.device)
    freeze_model(encoder)
    encoder.eval()

    debug_encoder(encoder, args.env_name, args.device)

    # Debug policy
    debug_policy(checkpoint, encoder, args.env_name, args.device)

    # Debug forward pass
    debug_forward_pass(checkpoint, encoder, args.env_name, args.device)

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()