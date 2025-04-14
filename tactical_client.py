import os
import json
import time
from typing import List, Dict, Any, Optional

from vgdl_utils import get_game_env
from agent import Agent
from strategy_client import StrategicLLMAgent, Strategy
from llm.deepseek_client import DeepSeekClient

class SimpleTacticalAgent(Agent):
    """
    A simplified GVGAI agent combining strategic planning and tactical action selection using LLMs.
    """

    def initialize(self, env, vgdl_rules: str, lvl_layout: str, work_dir: str = ".", game_name: Optional[str] = None):
        """
        Initializes the tactical agent with environment, rules, and directory settings.
        """
        self.env = env
        self.vgdl_rules = vgdl_rules
        self.lvl_layout = lvl_layout

        # Extract game name from environment if not provided
        if game_name is None:
            if hasattr(env, 'spec') and env.spec is not None:
                game_name = env.spec.id.split('-')[1]
            else:
                game_name = "unknown_game"

        self.game_name = game_name
        self.work_dir = os.path.join(work_dir, f"{game_name}_{int(time.time())}")
        os.makedirs(self.work_dir, exist_ok=True)

        # State tracking
        self.step_count = 0
        self.total_reward = 0
        self.action_space = env.action_space
        self.action_meanings = env.unwrapped.get_action_meanings()

        self.reward_history = []
        self.action_history = []

        # Strategic reasoning
        self.strategic_agent = StrategicLLMAgent(
            work_dir=self.work_dir, 
            game_name=self.game_name
        )

        # Tactical decision-making LLM client
        self.tactical_llm = DeepSeekClient()

        # Translate VGDL rules to English
        self.game_rules = self.strategic_agent.vgdl_rules_to_eng(vgdl_rules)

        # Save rules
        rules_file = os.path.join(self.work_dir, f'{self.game_name}_rules.txt')
        with open(rules_file, 'w', encoding='utf-8') as f:
            f.write(self.game_rules)

        self.strategy_update_interval = 20
        self.last_strategy_update = 0
        self.current_strategy_data = None

        print(f"SimpleTacticalAgent initialized: {self.game_name}")
        print(f"Action space: {self.action_space}")
        print(f"Action meanings: {self.action_meanings}")
        print(f"Working directory: {self.work_dir}")

    def act(self, observation, reward, done, info):
        """
        Decide the next action based on current state and strategy.
        """
        self.step_count += 1
        self.total_reward += reward

        if self.step_count > 1:
            self.reward_history.append(reward)

        current_state = info['ascii']

        if (self.step_count - self.last_strategy_update >= self.strategy_update_interval) or done or self.current_strategy_data is None:
            self._update_strategy(current_state)

        action = self._get_tactical_action(current_state, observation, info)
        self.action_history.append(action)

        return action

    def _update_strategy(self, current_state):
        """
        Generate or update long-term strategy using LLM.
        """
        print(f"\n--- Updating long-term strategy (Step {self.step_count}) ---")

        action_space = [f"{i}: {action}" for i, action in enumerate(self.action_meanings)]

        recent_events = []
        for i, reward in enumerate(self.reward_history[-10:]):
            if reward != 0:
                recent_events.append(f"Step {self.step_count-len(self.reward_history[-10:])+i}: received reward {reward}")

        self.current_strategy_data = self.strategic_agent.generate_long_term_strategy(
            game_rules=self.game_rules,
            current_state=current_state,
            action_space=action_space,
            recent_events=recent_events,
            additional_context=f"Total reward so far: {self.total_reward}, Steps taken: {self.step_count}"
        )

        self.last_strategy_update = self.step_count

        if self.strategic_agent.current_strategy:
            strategy_name = self.strategic_agent.current_strategy.name
            print(f"Current long-term strategy: {strategy_name}")

        print("--- Strategy update complete ---\n")

    def _get_tactical_action(self, current_state, observation, info):
        """
        Use tactical LLM to select the next best action.
        """
        if not self.current_strategy_data:
            return self.action_space.sample()

        strategy_name = self.strategic_agent.current_strategy.name if self.strategic_agent.current_strategy else "Unknown"
        strategy_description = self.current_strategy_data.get("strategy_description", "")
        if not strategy_description and self.strategic_agent.current_strategy:
            strategy_details = self.strategic_agent.strategy_templates.get(self.strategic_agent.current_strategy, {})
            strategy_description = strategy_details.get("description", "")

        long_term_plan = self.current_strategy_data.get("long_term_plan", "")
        action_space_desc = "\n".join([f"{i}: {action}" for i, action in enumerate(self.action_meanings)])
        recent_rewards = ", ".join([str(r) for r in self.reward_history[-5:]]) or "None"

        if self.action_history:
            recent_actions = self.action_history[-5:]
            action_history_text = ", ".join([self.action_meanings[a] for a in recent_actions])
        else:
            action_history_text = "None"

        prompt = f"""
I'm playing a video game and need to decide the next action. Please select the best action based on the following information:

**Game State**
{current_state}

**Available Actions**
{action_space_desc}

**Current High-Level Strategy**
Strategy Name: {strategy_name}
Strategy Description: {strategy_description}

**Long-Term Plan**
{long_term_plan}...

**Current Status**
Current Step: {self.step_count}
Total Reward: {self.total_reward}
Recent Rewards: {recent_rewards}
Recent Actions: {action_history_text}

Please choose the most appropriate action based on the game state and strategy.
Only return the action's numeric ID. For example: 0
"""

        try:
            response = self.tactical_llm.query(prompt)
            action = self._parse_action_from_response(response)

            if action is None or action >= self.action_space.n:
                print(f"Invalid action: {action}, using random fallback")
                return self.action_space.sample()

            return action

        except Exception as e:
            print(f"Error querying tactical action: {e}")
            return self.action_space.sample()

    def _parse_action_from_response(self, response):
        """
        Parses numeric action ID from the LLM's response.
        """
        try:
            response = response.strip()
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                return int(numbers[0])
            for i, action_name in enumerate(self.action_meanings):
                if action_name.lower() in response.lower():
                    return i
            return None
        except:
            return None

    def cleanup(self):
        """
        Save gameplay history and summary.
        """
        history_file = os.path.join(self.work_dir, f'{self.game_name}_action_reward_history.json')
        try:
            history_data = {
                "steps": self.step_count,
                "total_reward": self.total_reward,
                "action_history": [int(a) for a in self.action_history],
                "action_meanings": [self.action_meanings[a] for a in self.action_history],
                "reward_history": self.reward_history
            }
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"Action-reward history saved to: {history_file}")
        except Exception as e:
            print(f"Failed to save history: {e}")

        summary_file = os.path.join(self.work_dir, f'{self.game_name}_summary.txt')
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Game: {self.game_name}\n")
                f.write(f"Total Steps: {self.step_count}\n")
                f.write(f"Total Reward: {self.total_reward}\n")
                if self.strategic_agent.current_strategy:
                    f.write(f"Final Strategy: {self.strategic_agent.current_strategy.name}\n")
            print(f"Game summary saved to: {summary_file}")
        except Exception as e:
            print(f"Failed to save summary: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GVGAI Simple Tactical Agent Demo')
    parser.add_argument('--game', type=str, default='gvgai-theshepherd-lvl0-v0',
                        help='GVGAI game ID (default: gvgai-theshepherd-lvl0-v0)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Maximum number of steps (default: 200)')
    parser.add_argument('--render', action='store_true',
                        help='Render the game (visualization)')
    parser.add_argument('--update-interval', type=int, default=20,
                        help='Strategy update interval (default: 20 steps)')
    parser.add_argument('--work-dir', type=str, default='runs',
                        help='Working directory (default: runs)')

    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    print(f"Initializing game: {args.game}")
    env, vgdl_rules, lvl_layout = get_game_env(args.game)

    agent = SimpleTacticalAgent()
    agent.initialize(
        env,
        vgdl_rules,
        lvl_layout,
        work_dir=args.work_dir,
        game_name=args.game.split('-')[1]
    )

    agent.strategy_update_interval = args.update_interval

    observation = env.reset()
    total_reward = 0
    done = False

    dummy_action = 0
    observation, reward, done, info = env.step(dummy_action)
    total_reward += reward

    step_count = 0
    try:
        while not done and step_count < args.steps:
            if args.render:
                env.render()

            action = agent.act(observation, reward, done, info)
            print(f"Step {step_count}: Action {action} ({agent.action_meanings[action]})")

            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            if reward != 0:
                print(f"Received reward: {reward}, Total: {total_reward}")

            if args.render:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("Game interrupted manually.")

    if done:
        agent.act(observation, reward, done, info)
        print(f"Game finished at step {step_count}, total reward: {total_reward}")

    agent.cleanup()
    env.close()

    print(f"Game summary: Steps = {step_count}, Total Reward = {total_reward}")
    return total_reward


if __name__ == "__main__":
    main()
                                                                        