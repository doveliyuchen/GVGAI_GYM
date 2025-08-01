import gym_gvgai as gvgai
import os
import numpy as np
import torch
import argparse
import concurrent.futures
import threading
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gym.wrappers import ResizeObservation

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward.
    """
    def __init__(self, check_freq: int, log_dir: str, game: str, level: int, algorithm: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.game = game
        self.level = level
        self.algorithm = algorithm.lower()
        self.save_path = os.path.join(log_dir, f'best_{self.algorithm}_model_{game}_lvl{level}')
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
                        print(f"[{self.game}-lvl{self.level}] Timesteps: {self.num_timesteps}, Best reward: {self.best_mean_reward:.2f}, Current reward: {mean_reward:.2f}")

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"[{self.game}-lvl{self.level}] Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[{self.game}-lvl{self.level}] Error in callback: {e}")

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

def train_single_game_level(game, level, algorithm, timesteps, device):
    """Train a model for a single game and level."""
    task_name = f"{game}-lvl{level}"
    print(f"\n=== Starting training for {task_name} ===")
    
    # Create log directory
    log_dir = f"multi_game_{algorithm.lower()}_logs/{game}_{algorithm.lower()}_logs/lvl_{level}/"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[{task_name}] Log directory: {log_dir}")

    # Create environment
    env_name = f'gvgai-{game}-lvl{level}-v0'
    try:
        env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(env_name), (84, 84))])
        env = VecMonitor(env, log_dir)
        env = VecTransposeImage(env)
        print(f"[{task_name}] Environment created for {env_name}")
    except Exception as e:
        print(f"[{task_name}] Error creating environment: {e}")
        return f"[{task_name}] Training failed due to environment creation error."

    # Create model
    if algorithm.lower() == 'ppo':
        model = PPO(
            "CnnPolicy", 
            env, 
            verbose=0,
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
            verbose=0, 
            device=device,
            buffer_size=100000,  # Reduced buffer size for single-level training
            learning_starts=1000, 
            exploration_final_eps=0.1, 
            train_freq=4,
            learning_rate=1e-4,
            gamma=0.99,
            target_update_interval=1000
        )
    elif algorithm.lower() == 'a2c':
        model = A2C(
            "CnnPolicy",
            env,
            verbose=0,
            device=device,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print(f"[{task_name}] Created {algorithm} model with device: {device}")

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=log_dir, 
        game=game,
        level=level,
        algorithm=algorithm,
        verbose=1
    )

    # Train the model
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
    except Exception as e:
        print(f"[{task_name}] Error during training: {e}")
        env.close()
        return f"[{task_name}] Training failed with an error."

    # Save final model
    final_model_path = os.path.join(log_dir, f'final_{algorithm.lower()}_model_{game}_lvl{level}')
    model.save(final_model_path)
    print(f"[{task_name}] Final model saved to: {final_model_path}.zip")
    
    env.close()
    print(f"[{task_name}] Training completed!")
    
    return f"[{task_name}] Training completed successfully"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agents for multiple games, one agent per level.')
    parser.add_argument('--algorithm', '-a', type=str, default='dqn', choices=['ppo', 'dqn', 'a2c'],
                        help='Algorithm to use: ppo, dqn, or a2c (default: dqn)')
    parser.add_argument('--timesteps', '-t', type=int, default=500000,
                        help='Total timesteps for training per game level (default: 200000)')
    parser.add_argument('--workers', '-w', type=int, default=2,
                        help='Number of parallel workers (default: 6)')
    parser.add_argument('--sequential', '-s', action='store_true',
                        help='Run training sequentially instead of in parallel')
    parser.add_argument('--games', '-g', nargs='+',
                        help='List of games to train (default: all games)')

    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Total timesteps per game-level: {args.timesteps}")
    print(f"Parallel training: {not args.sequential}")
    if not args.sequential:
        print(f"Max workers: {args.workers}")

    # Define the 6 games
    all_games = ['aliens', 'boulderdash', 'escape', 'realsokoban', 'sokoban', 'zelda']

    if args.games:
        games_to_train = args.games
    else:
        games_to_train = all_games
    
    # Exclude 'aliens' for DQN
    if args.algorithm.lower() == 'dqn':
        games_to_train = [game for game in games_to_train if game != 'aliens']
        
    print(f"Games to train: {games_to_train}")

    # Create a list of all training tasks (game, level)
    tasks = []
    for game in games_to_train:
        levels = get_available_levels(game)[:5]  # Get up to 5 levels
        for level in levels:
            tasks.append((game, level))
            
    print(f"\nTotal training tasks: {len(tasks)}")
    for game, level in tasks:
        print(f"  - {game}-lvl{level}")

    if not args.sequential:
        print(f"\n=== Starting parallel training with {args.workers} workers ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(train_single_game_level, game, level, args.algorithm, args.timesteps, device): (game, level) for game, level in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                game, level = futures[future]
                try:
                    result = future.result()
                    print(f"Completed: {result}")
                except Exception as e:
                    print(f"Training for {game}-lvl{level} failed with error: {e}")
    else:
        print("\n=== Starting sequential training ===")
        for game, level in tasks:
            try:
                result = train_single_game_level(game, level, args.algorithm, args.timesteps, device)
                print(f"Completed: {result}")
            except Exception as e:
                print(f"Training for {game}-lvl{level} failed with error: {e}")
                continue
    
    print("\n=== All training completed! ===")
    print(f"Models saved in multi_game_{args.algorithm.lower()}_logs/ directories")

if __name__ == "__main__":
    main()
