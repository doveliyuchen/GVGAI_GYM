import os
import json
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import numpy as np

from conversation_manager import *
from vgdl_utils import *
from utilities import GifRecorder
from agent import Agent
# from llm.gemini_client import GeminiClient
from llm.deepseek_client import DeepSeekClient
# from llm.openai_client import OpenAIClient

class Strategy(Enum):
    Dont_Die = 0
    VerifyControls = 1
    WaitnSee = 2

    # TODO: Use for indexing the LLMs suggested Strategies
    # Overwrite and switch as necessary
    NewStrategyA = 3
    NewStrategyB = 4
    NewStrategyC = 5

    Exploit = 6

class StrategicLLMAgent:
    """An LLM-based agent that dynamically generates and manages long-term gameplay strategies."""

    def __init__(self, api_key: Optional[str] = None, work_dir: str = ".", game_name: str = "unknown_game"):
        """
        Initializes the agent and creates required folders and state variables.

        Args:
            api_key (Optional[str]): API key for the LLM client.
            work_dir (str): Directory to store strategy outputs.
            game_name (str): Name of the game for reference.
        """
        self.work_dir = work_dir
        self.game_name = game_name
        os.makedirs(work_dir, exist_ok=True)

        self.client = DeepSeekClient(api_key=api_key)

        # Initialize strategy tracking and templates
        self.strategy_history = []
        self.insights = []
        self.current_strategy = None
        self.strategy_templates = {}

        # Setup default strategy definitions
        self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """Initializes the default strategy templates."""
        self.strategy_templates = {
            Strategy.Dont_Die: {
                'description': 'Focus solely on avoiding immediate threats and staying alive.',
                'implementation': 'Prioritize moves that keep you away from enemies or hazards.',
                'insights': ""
            },
            Strategy.VerifyControls: {
                'description': 'Test different controls to understand game mechanics.',
                'implementation': 'Systematically try each possible action and observe the results.',
                'insights': ""
            },
            Strategy.WaitnSee: {
                'description': 'Observe the environment and how entities behave without taking major actions.',
                'implementation': 'Alternate between no-op actions and minimal movements to observe the game.',
                'insights': ""
            },
            Strategy.Exploit: {
                'description': 'Apply learned knowledge to maximize score/progress.',
                'implementation': 'Make aggressive moves toward objectives, apply learned patterns.',
                'insights': ""
            },
            Strategy.NewStrategyA: {
                'description': "Custom strategy slot A - can be modified",
                'implementation': "",
                'insights': ""
            },
            Strategy.NewStrategyB: {
                'description': "Custom strategy slot B - can be modified",
                'implementation': "",
                'insights': ""
            },
            Strategy.NewStrategyC: {
                'description': "Custom strategy slot C - can be modified",
                'implementation': "",
                'insights': ""
            }
        }

    def _format_template(self, title, body):
        """Formats a section with a title and indented body text."""
        tab = "\t"
        lines = body.split("\n")
        tabbed_lines = [tab + line for line in lines]
        return f"**{title}**\n" + "\n".join(tabbed_lines) + "\n"

    def _parse_json_response(self, response):
        """Parses a JSON object from the LLM response text."""
        text = response.strip()

        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if part.strip().startswith("json"):
                    text = part.replace("json", "", 1).strip()
                    break
                elif "{" in part and "}" in part:
                    text = part.strip()
                    break

        if "JSON:" in text:
            text = text.split("JSON:", 1)[1].strip()

        if not text.startswith("{") and "{" in text:
            text = text[text.find("{"):]
        if not text.endswith("}") and "}" in text:
            text = text[:text.rfind("}")+1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Original response: {response}")
            return {}

    def generate_long_term_strategy(self, 
                                    game_rules: str, 
                                    current_state: str, 
                                    action_space: List[str],
                                    recent_events: List[str] = None,
                                    additional_context: str = None) -> Dict:
        """
        Generates a long-term strategy based on the current game state, rules, and context.
        """
        action_space_desc = "\n".join([f"{i}: {action}" for i, action in enumerate(action_space)])
        recent_events_desc = "\n".join(recent_events) if recent_events else "None"

        current_strategy_desc = ""
        if self.current_strategy:
            strategy_info = self.strategy_templates.get(self.current_strategy, {})
            current_strategy_desc = f"{self.current_strategy.name}: {strategy_info.get('description', '')}"
        else:
            current_strategy_desc = "None"

        all_strategies_desc = []
        for strategy in Strategy:
            details = self.strategy_templates.get(strategy, {})
            desc = details.get('description', '')
            all_strategies_desc.append(f"- {strategy.name}: {desc}")

        all_strategies_text = "\n".join(all_strategies_desc)

        insights_text = "\n- ".join(self.insights) if self.insights else ""
        if insights_text and not insights_text.startswith("- "):
            insights_text = "- " + insights_text

        prompt = self._format_template("Preface", 
            "You are an AI game assistant tasked with designing a **long-term game strategy** (at least 10 steps ahead). Based on the current game state and rules, please provide specific long-term strategic guidance.")
        prompt += self._format_template("Game Rules", game_rules)
        prompt += self._format_template("Action Space", action_space_desc)
        prompt += self._format_template("Current Strategy", current_strategy_desc)
        prompt += self._format_template("Current State", current_state)
        prompt += self._format_template("Recent Events", recent_events_desc)
        prompt += self._format_template("Insights", insights_text)

        if additional_context:
            prompt += self._format_template("Additional Context", additional_context)

        prompt += """
Based on the game rules and current state, analyze the situation and recommend the **best long-term strategy** (at least 10 steps).

Available strategies:
""" + all_strategies_text + """

You may:
1. Continue using or switch to an existing strategy (use the **exact strategy name**).
2. Modify any strategy by providing a new description and implementation guide.

To define a new custom strategy, select a custom slot (NewStrategyA, B, or C) and provide its details.

Please respond in JSON format with your long-term strategic analysis:

```json
{
  "recommended_strategy": "<exact name of the strategy from the above list>",
  "modify_strategy": true/false,
  "strategy_description": "<new description if modifying the strategy>",
  "implementation_guidance": "<how to execute the strategy if modified>",
  "long_term_plan": "<detailed plan of at least 10 steps explaining how the strategy will unfold over future turns>",
  "anticipated_challenges": "<what challenges might arise while following this strategy>",
  "success_metrics": "<how to determine if the strategy is successful>",
  "fallback_strategy": "<backup strategy to use if the primary one fails>",
  "reasoning": "<rationale behind recommending this strategy>",
  "new_insights": "<any new observations or discoveries about the game>"
}
```
"""
        
        # 发送请求到LLM
        response = self.client.query(prompt)
        
        # 解析响应
        strategy_data = self._parse_json_response(response)
        
        # 更新当前策略
        if strategy_data and "recommended_strategy" in strategy_data:
            try:
                new_strategy = Strategy[strategy_data["recommended_strategy"]]
                
                # 检查是否需要修改策略
                if strategy_data.get("modify_strategy", False):
                    # 更新策略详情
                    if "strategy_description" in strategy_data:
                        self.strategy_templates[new_strategy]["description"] = strategy_data["strategy_description"]
                    
                    if "implementation_guidance" in strategy_data:
                        self.strategy_templates[new_strategy]["implementation"] = strategy_data["implementation_guidance"]
                    
                    if "long_term_plan" in strategy_data:
                        self.strategy_templates[new_strategy]["long_term_plan"] = strategy_data["long_term_plan"]
                
                # 记录策略变更
                self.current_strategy = new_strategy
                self.strategy_history.append({
                    "strategy": new_strategy.name,
                    "reasoning": strategy_data.get("reasoning", "无理由提供")
                })
                
                # 记录新洞察
                if "new_insights" in strategy_data and strategy_data["new_insights"]:
                    print("insight yet to implement")
                    ##self.add_insight(strategy_data["new_insights"])
                
            except KeyError:
                print(f"警告: 未知策略 {strategy_data['recommended_strategy']}")
        
        # 保存策略到文件
        self._save_strategy_data(strategy_data)
        
        return strategy_data

    def _save_strategy_data(self, strategy_data):
        """将策略数据保存到文件"""
        strategy_file = os.path.join(self.work_dir, f'{self.game_name}_latest_strategy.json')
        
        try:
            with open(strategy_file, 'w') as f:
                json.dump(strategy_data, f, indent=2)
        except Exception as e:
            print(f"保存策略数据时发生错误: {e}")

    def vgdl_rules_to_eng(self,game_rules):
        return retrieve_vgdl_translation(self.client.query(PromptFactory.vgdl_rules_to_eng(game_rules)))
    

if __name__ == "__main__":
    env, vgdl_rules, lvl_layout = get_game_env("gvgai-theshepherd-lvl0-v0")
    agent = StrategicLLMAgent(game_name="theshepherd")
    dummy_action = 0
    env.reset()
    observation, reward, done, info = env.step(dummy_action)
    game_rules = agent.vgdl_rules_to_eng(vgdl_rules)
    with open('./vgdl_translations/shepherd.txt', 'a', encoding='utf-8') as file:
        file.write(game_rules)
    current_state = info['ascii']
    action_meanings = env.unwrapped.get_action_meanings()
    action_space = [f"{i}: {action}" for i, action in enumerate(action_meanings)]
    strategy_data = agent.generate_long_term_strategy(
            game_rules=game_rules,
            current_state=current_state,
            action_space=action_space)
    print(strategy_data)
    