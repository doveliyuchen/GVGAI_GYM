import os
import json
import csv
import gym
from gym.wrappers import ResizeObservation
import gym_gvgai as gvgai
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from collections import Counter
import time
import re
import random

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param game: (str) Game name for saving path
    :param level: (int) Level number for saving path
    :param algorithm: (str) Algorithm name for saving path
    :param verbose: (int)
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

def get_all_games(exclude_games):
    games_path = os.path.join(os.path.dirname(gvgai.__file__), 'envs', 'games')
    all_games = [d.name for d in os.scandir(games_path) if d.is_dir()]
    return [game for game in all_games if game.split('_v0')[0] not in exclude_games]

def get_available_levels(game_name, is_test=False):
    game_path = os.path.join(os.path.dirname(gvgai.__file__), 'envs', 'games', f"{game_name}_v0")
    if not os.path.isdir(game_path):
        return [0]
    
    levels = set()
    for f in os.listdir(game_path):
        is_test_file = 'test' in f
        if (is_test and is_test_file) or (not is_test and not is_test_file):
            match = re.search(r'_lvl(\d+)\.txt$', f)
            if match:
                levels.add(int(match.group(1)))
    
    return sorted(list(levels)) if levels else [0]

def test_agent(model, game, test_levels, runs_per_level=5, max_steps_per_run=2000):
    wins = 0
    total_runs = 0
    for level in test_levels:
        print(f"Testing on {game}-lvl{level}...")
        total_runs += runs_per_level
        env_name = f'gvgai-{game}-lvl{level}-v0'
        try:
            env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(env_name), (84, 84))])
            env = VecTransposeImage(env)
            for _ in range(runs_per_level):
                obs = env.reset()
                done = False
                steps = 0
                while not done and steps < max_steps_per_run:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, info = env.step(action)
                    if info[0]['winner'] == "PLAYER_WINS":
                        wins += 1
                    steps += 1
            env.close()
        except Exception as e:
            print(f"Error during testing {env_name}: {e}")

    win_rate = wins / total_runs if total_runs > 0 else 0
    print(f"Test results for {game} on levels {test_levels}: Win rate {win_rate:.2%}")
    return win_rate


def train_agent(agent_type, game, train_level, test_levels, timesteps, output_dir, model=None):
    env_name = f'gvgai-{game}-lvl{train_level}-v0'
    
    # Create log directory for monitoring
    log_dir = os.path.join(output_dir, f"{game}_lvl{train_level}_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        env = DummyVecEnv([lambda: ResizeObservation(gvgai.make(env_name), (84, 84))])
        env = VecMonitor(env, log_dir)  # Add VecMonitor for reward tracking
        env = VecTransposeImage(env)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        return None, 0

    if model is None:
        if agent_type == 'DQN':
            model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000)
        elif agent_type == 'PPO':
            model = PPO("CnnPolicy", env, verbose=1)
        else:
            raise ValueError("Agent type must be 'DQN' or 'PPO'")
    else:
        model.set_env(env)

    # Create the callback: check every 1000 steps for best reward
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=log_dir, 
        game=game,
        level=train_level,
        algorithm=agent_type
    )

    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)

    win_rate = test_agent(model, game, test_levels)

    env.close()
    return model, win_rate


if __name__ == '__main__':
    EXCLUDED_GAMES = ['testgame1', 'testgame2', 'testgame3']
    AGENT_TYPE = 'DQN'  # Choose 'DQN' or 'PPO'
    TIMESTEPS_PER_LEVEL = 10000
    OUTPUT_DIR = "rl_agent_runs_output"
    MODEL_PATH = os.path.join(OUTPUT_DIR, f"{AGENT_TYPE}_general_model_checkpoint.zip")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_games = [g.split('_v0')[0] for g in get_all_games([])]
    available_games = [game for game in all_games if game not in EXCLUDED_GAMES and 'ghost' not in game]
    
    random.shuffle(available_games)

    model = None
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        if AGENT_TYPE == 'DQN':
            model = DQN.load(MODEL_PATH)
        elif AGENT_TYPE == 'PPO':  
            model = PPO.load(MODEL_PATH)

    for game in available_games:
        print(f"--- Continuing training with game: {game} ---")
        game_levels = get_available_levels(game)
        if not game_levels:
            print(f"No levels found for {game}. Skipping.")
            continue

        train_levels = get_available_levels(game, is_test=False)
        test_levels = get_available_levels(game, is_test=True)

        if not train_levels:
            print(f"No training levels found for {game}. Skipping.")
            continue
        
        if not test_levels:
            print(f"No test levels found for {game}. Using training levels for testing.")
            test_levels = train_levels

        # Check for action space compatibility before training
        env_name = f'gvgai-{game}-lvl{train_levels[0]}-v0'
        try:
            temp_env = gvgai.make(env_name)
            game_action_space = temp_env.action_space
            temp_env.close()
        except Exception as e:
            print(f"Error creating env for {game} to check action space, skipping. Error: {e}")
            continue

        # Check action space compatibility by comparing types and dimensions
        if model is not None:
            model_action_space = model.action_space
            spaces_compatible = False
            
            # Check if both are Discrete spaces with same n
            if (hasattr(model_action_space, 'n') and hasattr(game_action_space, 'n') and 
                type(model_action_space).__name__ == type(game_action_space).__name__):
                if model_action_space.n == game_action_space.n:
                    spaces_compatible = True
                    print(f"Action spaces compatible for {game}. Both are {type(game_action_space).__name__}({game_action_space.n})")
                else:
                    print(f"Action space size mismatch for {game}. Model has {type(model_action_space).__name__}({model_action_space.n}), game has {type(game_action_space).__name__}({game_action_space.n}). Skipping game.")
                    continue
            # Check if both are Box spaces with same shape
            elif (hasattr(model_action_space, 'shape') and hasattr(game_action_space, 'shape') and
                  type(model_action_space).__name__ == type(game_action_space).__name__):
                if model_action_space.shape == game_action_space.shape:
                    spaces_compatible = True
                    print(f"Action spaces compatible for {game}. Both are {type(game_action_space).__name__} with shape {game_action_space.shape}")
                else:
                    print(f"Action space shape mismatch for {game}. Model has {type(model_action_space).__name__} with shape {model_action_space.shape}, game has {type(game_action_space).__name__} with shape {game_action_space.shape}. Skipping game.")
                    continue
            else:
                print(f"Incompatible action space types for {game}. Model has {model_action_space}, game has {game_action_space}. Skipping game.")
                continue

        print(f"For game {game}: Training on levels {train_levels}, Testing on levels {test_levels}")

        for level in train_levels:
            print(f"Training on {game}-lvl{level}...")
            model, win_rate = train_agent(AGENT_TYPE, game, level, test_levels, TIMESTEPS_PER_LEVEL, OUTPUT_DIR, model=model)

            if model:
                print(f"Win rate after training on {game}-lvl{level}: {win_rate:.2%}")
                print(f"Saving model checkpoint to {MODEL_PATH}")
                model.save(MODEL_PATH)
                if win_rate > 0.8:
                    print(f"Win rate {win_rate:.2%} exceeds 50%. Stopping training for {game} and moving to the next game.")
                    break

    print(f"--- Finished training cycle. Final model saved at {MODEL_PATH} ---")
