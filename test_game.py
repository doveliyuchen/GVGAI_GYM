#!/usr/bin/env python3
"""
单个模型测试脚本 - 允许指定模型路径和游戏参数
Single Model Testing Script - Allows specifying model path and game parameters
"""

import os
import csv
import argparse
import gym
from gym.wrappers import ResizeObservation
import gym_gvgai as gvgai
import numpy as np
from stable_baselines3 import DQN, PPO
import cv2
from datetime import datetime

class RLAgent:
    def __init__(self, model_path, algorithm_type="auto"):
        """
        Initialize RL Agent with specified model path
        
        Args:
            model_path: Path to the trained model file (.zip)
            algorithm_type: Algorithm type ("DQN", "PPO", or "auto" to auto-detect)
        """
        self.model_path = model_path
        
        # Auto-detect algorithm type if not specified
        if algorithm_type.lower() == "auto":
            if "dqn" in model_path.lower():
                self.algorithm_type = "DQN"
            elif "ppo" in model_path.lower():
                self.algorithm_type = "PPO"
            else:
                # Default to DQN if can't detect
                print(f"Warning: Cannot auto-detect algorithm type from path '{model_path}', defaulting to DQN")
                self.algorithm_type = "DQN"
        else:
            self.algorithm_type = algorithm_type.upper()
        
        self.name = f"{self.algorithm_type}Agent"
        
        # Load model
        print(f"Loading {self.algorithm_type} model from: {model_path}")
        try:
            if self.algorithm_type == "DQN":
                self.model = DQN.load(model_path, device="cpu")
            elif self.algorithm_type == "PPO":
                self.model = PPO.load(model_path, device="cpu")
            else:
                raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
            
            print(f"✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
    def preprocess_observation(self, obs):
        """
        Preprocess observation to match training format:
        1. Resize to 84x84
        2. Convert to channels-first format (4, 84, 84)
        3. Normalize to [0, 1]
        """
        # Resize and convert to float32
        resized_obs = cv2.resize(obs, (84, 84)).astype(np.float32)
        
        # Normalize to [0, 1]
        resized_obs /= 255.0
        
        # Transpose to channels-first and add batch dimension
        processed_obs = np.transpose(resized_obs, (2, 0, 1))[np.newaxis, ...]
        
        return processed_obs
        
    def act(self, stateObs, actions):
        """
        Act function following GVGAI Agent pattern
        
        Args:
            stateObs: observation from the environment
            actions: list of available actions for this game
            
        Returns:
            action_id: index of the selected action
        """
        # Preprocess observation to match training format
        processed_obs = self.preprocess_observation(stateObs)
        
        # Get model prediction
        model_action, _states = self.model.predict(processed_obs, deterministic=True)
        
        # Ensure model_action is a scalar
        if isinstance(model_action, np.ndarray):
            model_action = model_action[0]
        
        # Map to available actions (use modulo to ensure valid action)
        action_id = int(model_action) % len(actions)
        return action_id

def test_model_on_game(agent, game, levels, episodes_per_level=5, max_steps_per_episode=2000, verbose=True):
    """
    Test a model on specified game and levels
    
    Args:
        agent: RLAgent instance
        game: Game name (e.g., "aliens", "zelda")
        levels: List of levels to test (e.g., [0, 1, 2])
        episodes_per_level: Number of episodes to run per level
        max_steps_per_episode: Maximum steps per episode
        verbose: Whether to print detailed progress
        
    Returns:
        results: List of result dictionaries
        summary: Summary statistics
    """
    results = []
    total_wins = 0
    total_episodes = 0
    
    if verbose:
        print(f"\n🎮 Testing {agent.algorithm_type} model on '{game}'")
        print(f"📊 Levels: {levels}, Episodes per level: {episodes_per_level}")
        print("=" * 60)
    
    for level in levels:
        level_wins = 0
        level_episodes = 0
        
        env_name = f'gvgai-{game}-lvl{level}-v0'
        
        if verbose:
            print(f"\n🎯 Level {level} ({env_name}):")
        
        try:
            # Create environment
            env = gvgai.make(env_name)
            
            for episode in range(episodes_per_level):
                obs = env.reset()
                done = False
                steps = 0
                total_reward = 0
                
                while not done and steps < max_steps_per_episode:
                    # Get available actions
                    available_actions = list(range(env.action_space.n))
                    
                    # Use agent to select action
                    action = agent.act(obs, available_actions)
                    
                    # Take step
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                
                # Check if won
                won = 1 if (done and info.get('winner') == "PLAYER_WINS") else 0
                score = info.get('score', total_reward)
                
                # Record result
                result = {
                    'algorithm': agent.algorithm_type,
                    'game': game,
                    'level': level,
                    'episode': episode + 1,
                    'won': won,
                    'score': score,
                    'steps': steps,
                    'model_path': agent.model_path
                }
                results.append(result)
                
                # Update counters
                if won:
                    level_wins += 1
                    total_wins += 1
                level_episodes += 1
                total_episodes += 1
                
                if verbose:
                    status = "🟢 WIN" if won else "🔴 LOSE"
                    print(f"  Episode {episode+1:2d}: {status} | Score: {score:8.1f} | Steps: {steps:4d}")
            
            env.close()
            
            # Level summary
            level_win_rate = level_wins / level_episodes if level_episodes > 0 else 0
            if verbose:
                print(f"  📈 Level {level} Summary: {level_wins}/{level_episodes} wins ({level_win_rate:.1%})")
                
        except Exception as e:
            print(f"  ❌ Error testing {env_name}: {e}")
            continue
    
    # Overall summary
    overall_win_rate = total_wins / total_episodes if total_episodes > 0 else 0
    summary = {
        'algorithm': agent.algorithm_type,
        'game': game,
        'levels_tested': levels,
        'total_episodes': total_episodes,
        'total_wins': total_wins,
        'win_rate': overall_win_rate,
        'model_path': agent.model_path
    }
    
    if verbose:
        print(f"\n🏆 Overall Results:")
        print(f"   Win Rate: {overall_win_rate:.1%} ({total_wins}/{total_episodes} wins)")
    
    return results, summary

def save_multigame_results(results, summary, output_prefix=None):
    """Save multi-game test results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_prefix is None:
        # Create a prefix from the list of games, truncating if too long
        games_str = "_".join(summary['games'])
        algorithm = summary['algorithm'].lower()
        output_prefix = f"test_{algorithm}_{games_str[:30]}_{timestamp}"
    
    # Save detailed results to CSV
    csv_file = f"{output_prefix}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = ['algorithm', 'game', 'level', 'episode', 'won', 'score', 'steps', 'model_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Save summary to text file
    txt_file = f"{output_prefix}_summary.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Multi-Game Model Test Summary\n")
        f.write(f"=============================\n\n")
        f.write(f"Model Path: {summary['model_path']}\n")
        f.write(f"Algorithm: {summary['algorithm']}\n")
        f.write(f"Games Tested: {', '.join(summary['games'])}\n")
        f.write(f"Levels Tested per Game: {summary['levels_tested']}\n\n")
        
        f.write(f"Overall Summary:\n")
        f.write(f"----------------\n")
        f.write(f"Total Episodes: {summary['total_episodes']}\n")
        f.write(f"Total Wins: {summary['total_wins']}\n")
        f.write(f"Overall Win Rate: {summary['win_rate']:.1%}\n\n")
        
        f.write("Per-Game Breakdown:\n")
        f.write("-------------------\n")
        for game_summary in summary['game_summaries']:
            game = game_summary['game']
            wins = game_summary['total_wins']
            total = game_summary['total_episodes']
            rate = game_summary['win_rate']
            f.write(f"Game: {game:<15} Win Rate: {rate:6.1%} ({wins}/{total} wins)\n")
            
            # Level breakdown for each game
            level_results = {}
            for result in [r for r in results if r['game'] == game]:
                level = result['level']
                if level not in level_results:
                    level_results[level] = {'wins': 0, 'total': 0}
                level_results[level]['total'] += 1
                if result['won']:
                    level_results[level]['wins'] += 1
            
            for level in sorted(level_results.keys()):
                wins = level_results[level]['wins']
                total = level_results[level]['total']
                rate = wins / total if total > 0 else 0
                f.write(f"  - Level {level}: {wins}/{total} wins ({rate:.1%})\n")
            f.write("\n")

    return csv_file, txt_file

def main():
    parser = argparse.ArgumentParser(description="Test a trained RL model on one or more GVGAI games")
    
    # Required arguments
    parser.add_argument("model_path", type=str, 
                       help="Path to the trained model file (.zip)")
    parser.add_argument("games", type=str, nargs='+',
                       help="Game name(s) to test (e.g., 'aliens', 'zelda', 'sokoban')")
    
    # Optional arguments
    parser.add_argument("--levels", type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help="Levels to test (default: 0 1 2 3 4)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes per level (default: 5)")
    parser.add_argument("--max-steps", type=int, default=2000,
                       help="Maximum steps per episode (default: 2000)")
    parser.add_argument("--algorithm", type=str, default="auto", choices=["auto", "DQN", "PPO"],
                       help="Algorithm type (default: auto-detect)")
    parser.add_argument("--output-prefix", type=str, default=None,
                       help="Output file prefix (default: auto-generated)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return 1
    
    # Print configuration
    if not args.quiet:
        print("🚀 GVGAI Model Testing")
        print("=" * 50)
        print(f"📁 Model: {args.model_path}")
        print(f"🎮 Games: {', '.join(args.games)}")
        print(f"🎯 Levels: {args.levels}")
        print(f"📊 Episodes per level: {args.episodes}")
        print(f"⏱️  Max steps per episode: {args.max_steps}")
        print(f"🤖 Algorithm: {args.algorithm}")
    
    try:
        # Create agent
        agent = RLAgent(args.model_path, args.algorithm)
        
        all_results = []
        all_summaries = []

        # Loop over all specified games
        for game in args.games:
            results, summary = test_model_on_game(
                agent=agent,
                game=game,
                levels=args.levels,
                episodes_per_level=args.episodes,
                max_steps_per_episode=args.max_steps,
                verbose=not args.quiet
            )
            if results:
                all_results.extend(results)
                all_summaries.append(summary)

        if not all_results:
            print("⚠️  No results generated from any game. Exiting.")
            return 1
            
        # Create overall summary
        total_wins = sum(s['total_wins'] for s in all_summaries)
        total_episodes = sum(s['total_episodes'] for s in all_summaries)
        overall_win_rate = total_wins / total_episodes if total_episodes > 0 else 0

        multi_game_summary = {
            'algorithm': agent.algorithm_type,
            'games': args.games,
            'levels_tested': args.levels,
            'total_episodes': total_episodes,
            'total_wins': total_wins,
            'win_rate': overall_win_rate,
            'model_path': agent.model_path,
            'game_summaries': all_summaries
        }
        
        # Save results
        csv_file, txt_file = save_multigame_results(all_results, multi_game_summary, args.output_prefix)
        print(f"\n💾 Results saved:")
        print(f"   📊 Detailed: {csv_file}")
        print(f"   📄 Summary: {txt_file}")
        
        # Print final summary
        if not args.quiet:
            print(f"\n🎯 Final Overall Results: {multi_game_summary['win_rate']:.1%} win rate ({multi_game_summary['total_wins']}/{multi_game_summary['total_episodes']} wins)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
