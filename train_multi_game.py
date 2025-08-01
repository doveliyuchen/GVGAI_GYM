import gym_gvgai as gvgai
import os
import numpy as np
import torch
import argparse
import concurrent.futures
import threading
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gym.wrappers import ResizeObservation

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, game: str, algorithm: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.game = game
        self.algorithm = algorithm.lower()
        self.save_path = os.path.join(log_dir, f'best_{self.algorithm}_model_{game}')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"[{self.game}] Timesteps: {self.num_timesteps}, Best reward: {self.best_mean_reward:.2f}, Current reward: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"[{self.game}] Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[{self.game}] Error in callback: {e}")

        return True

def get_available_levels(game_name):
    """Get all available levels for a game (both training and test levels)"""
    game_path = os.path.join(os.path.dirname(gvgai.__file__), 'envs', 'games', f"{game_name}_v0")
    if not os.path.isdir(game_path):
        return [0]
    
    levels = set()
    for f in os.listdir(game_path):
        if f.endswith('.txt'):
            # Extract level number from filename
            import re
            match = re.search(r'_lvl(\d+)\.txt$', f)
            if match:
                levels.add(int(match.group(1)))
    
    return sorted(list(levels)) if levels else [0]

def create_multi_level_env(game, levels, log_dir):
    """Create an environment that cycles through multiple levels"""
    def make_env():
        # Start with the first level
        env_name = f'gvgai-{game}-lvl{levels[0]}-v0'
        env = ResizeObservation(gvgai.make(env_name), (84, 84))
        return env
    
    env = DummyVecEnv([make_env])
    env = VecMonitor(env, log_dir)
    env = VecTransposeImage(env)
    return env

def train_single_game(game, algorithm, timesteps, device):
    """Train a model for a single game across all its levels"""
    print(f"\n=== Starting training for {game} ===")
    
    # Get all available levels for this game
    levels = get_available_levels(game)
    print(f"[{game}] Found {len(levels)} levels: {levels}")
    
    # Create log directory
    log_dir = f"multi_game_{algorithm.lower()}_logs/{game}_{algorithm.lower()}_logs/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[{game}] Log directory: {log_dir}")

    # Create environment for the first level
    env_name = f'gvgai-{game}-lvl{levels[0]}-v0'
    env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(env_name), (84, 84))])
    env = VecMonitor(env, log_dir)
    env = VecTransposeImage(env)
    print(f"[{game}] Environment created for {env_name}")

    # Create model
    if algorithm.lower() == 'ppo':
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
    elif algorithm.lower() == 'dqn':
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
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print(f"[{game}] Created {algorithm} model with device: {device}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=log_dir, 
        game=game,
        algorithm=algorithm
    )

    # Train on each level
    timesteps_per_level = timesteps // len(levels)
    print(f"[{game}] Training {timesteps_per_level} timesteps per level")
    
    for i, level in enumerate(levels):
        print(f"\n[{game}] Training on level {level} ({i+1}/{len(levels)})")
        
        # Create environment for this specific level
        env_name = f'gvgai-{game}-lvl{level}-v0'
        try:
            new_env = DummyVecEnv([lambda lvl=level: ResizeObservation(gvgai.make(f'gvgai-{game}-lvl{lvl}-v0'), (84, 84))])
            new_env = VecMonitor(new_env, log_dir)
            new_env = VecTransposeImage(new_env)
            
            # Set the new environment
            model.set_env(new_env)
            
            # Train on this level
            model.learn(total_timesteps=timesteps_per_level, callback=callback, reset_num_timesteps=False)
            
            new_env.close()
            print(f"[{game}] Completed training on level {level}")
            
        except Exception as e:
            print(f"[{game}] Error training on level {level}: {e}")
            continue

    # Save final model
    final_model_path = os.path.join(log_dir, f'final_{algorithm.lower()}_model_{game}')
    model.save(final_model_path)
    print(f"[{game}] Final model saved to: {final_model_path}.zip")
    
    env.close()
    print(f"[{game}] Training completed!")
    
    return f"[{game}] Training completed successfully"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agents for multiple games')
    parser.add_argument('--algorithm', '-a', type=str, default='dqn', choices=['ppo', 'dqn'],
                        help='Algorithm to use: ppo or dqn (default: dqn)')
    parser.add_argument('--timesteps', '-t', type=int, default=1000000,
                        help='Total timesteps for training per game (default: 1000000)')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Train games in parallel (default: sequential)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Total timesteps per game: {args.timesteps}")
    print(f"Parallel training: {args.parallel}")
    
    # Define the 6 games
    games = [ 'boulderdash', 'escape', 'sokoban', 'zelda']
    print(f"Games to train: {games}")
    
    if args.parallel:
        print("\n=== Starting parallel training ===")
        # Train games in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for game in games:
                future = executor.submit(train_single_game, game, args.algorithm, args.timesteps, device)
                futures.append(future)
            
            # Wait for all training to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed: {result}")
                except Exception as e:
                    print(f"Training failed with error: {e}")
    else:
        print("\n=== Starting sequential training ===")
        # Train games sequentially
        for game in games:
            try:
                result = train_single_game(game, args.algorithm, args.timesteps, device)
                print(f"Completed: {result}")
            except Exception as e:
                print(f"Training for {game} failed with error: {e}")
                continue
    
    print("\n=== All training completed! ===")
    print(f"Models saved in multi_game_{args.algorithm.lower()}_logs/ directories")

if __name__ == "__main__":
    main()
