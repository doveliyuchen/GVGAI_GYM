import gym_gvgai as gvgai
import os
import numpy as np
import torch
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gym.wrappers import ResizeObservation

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param algorithm: (str) Algorithm name for saving path
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, algorithm: str, game: str, verbose=1, ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.algorithm = algorithm.lower()
        self.save_path = os.path.join(log_dir, f'best_{self.algorithm}_model_{game}')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

def create_model(algorithm, env, device):
    """
    Create a model based on the specified algorithm
    
    Args:
        algorithm: 'ppo' or 'dqn'
        env: The environment
        device: torch device (cpu or cuda)
    
    Returns:
        The created model
    """
    if algorithm.lower() == 'ppo':
        # PPO hyperparameters optimized for GVGAI environments
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=1,
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        print("Created PPO model with GPU support!")
        
    elif algorithm.lower() == 'dqn':
        # DQN hyperparameters
        model = DQN(
            "CnnPolicy", 
            env, 
            verbose=1, 
            device=device,
            buffer_size=1000000, 
            learning_starts=1000, 
            exploration_final_eps=0.1, 
            train_freq=4,
            learning_rate=1e-4,
            gamma=0.99,
            target_update_interval=1000
        )
        print("Created DQN model with GPU support!")
        
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'ppo' or 'dqn'")
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agent with DQN or PPO')
    parser.add_argument('--algorithm', '-a', type=str, default='ppo', choices=['ppo', 'dqn'],
                        help='Algorithm to use: ppo or dqn (default: ppo)')
    parser.add_argument('--timesteps', '-t', type=int, default=1000000,
                        help='Total timesteps for training (default: 1000000)')
    parser.add_argument('--env', '-e', type=str, default='gvgai-golddigger-lvl0-v0',
                        help='Environment name (default: gvgai-golddigger-lvl0-v0)')
    
    args = parser.parse_args()
    
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.timesteps}")
    
    # Create log dir based on algorithm
    log_dir = f"{args.algorithm.lower()}_gvgai_logs/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Create and wrap the environment
    env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(args.env), (84, 84))])
    env = VecMonitor(env, log_dir)
    env = VecTransposeImage(env)
    print("Environment created and wrapped successfully!")

    # Create model based on selected algorithm
    model = create_model(args.algorithm, env, device)
    print(f"Model device: {model.device}")

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=log_dir, 
        algorithm=args.algorithm
    )

    # Start training
    print(f"\nStarting {args.algorithm.upper()} training...")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    
    # Save final model
    final_model_path = os.path.join(log_dir, f'final_{args.algorithm.lower()}_model_{args.env}')
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}.zip")
    
    # Close environment
    env.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
