import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from data_logging import init_experiment, finish_experiment
from env_helpers import *
from training_helpers import *
from train_encoder import train_encoder
from train_transition_model import train_trans_model
from evaluate_model import eval_model
from train_rl_model import train_rl_model
from e2e_train import full_train


def main():
    """Main training and evaluation pipeline"""
    # Parse args

    args = get_args(apply_optimizations=True)

    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)
    print(args)

    # Setup GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    try:
        # Training pipeline
        if args.e2e_loss:
            print("Starting end-to-end training...")
            encoder_model, trans_model = full_train(args)
        else:
            print("Starting sequential training...")

            # Train encoder
            print("Step 1: Training encoder...")
            encoder_model = train_encoder(args)

            # Train transition model
            print("Step 2: Training transition model...")
            trans_model = train_trans_model(args, encoder_model)

            # Train RL model if requested
            if args.rl_train_steps > 0:
                print("Step 3: Training RL model...")
                train_rl_model(args, encoder_model, trans_model)

        # Evaluation
        print("Step 4: Evaluating models...")
        eval_model(args, encoder_model, trans_model)

        print("✅ Training and evaluation completed successfully!")

    except KeyboardInterrupt:
        print("❌ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up logging
        finish_experiment(args)


if __name__ == '__main__':
    main()