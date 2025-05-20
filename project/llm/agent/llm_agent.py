import os
import re
from typing import Optional, Any, Tuple, Literal
from llm.client import create_client_from_config
from llm.utils.agent_components import parse_action_from_response
from llm.utils.build_prompt import build_static_prompt, build_dynamic_prompt, PromptLogger, ReflectionManager
from llm.utils.config import get_profile_config
from llm.utils.game_analysis import generate_full_analysis_report

def rotate_ascii_left(ascii_map: str) -> str:
    lines = [list(line) for line in ascii_map.strip().splitlines()]
    if not lines:
        return ascii_map
    rotated = list(zip(*lines[::-1]))
    return "\n".join("".join(row) for row in rotated)

class LLMPlayer:
    def __init__(
        self,
        model_name: str,
        env: Any,
        vgdl_rules: str,
        initial_state: Optional[str] = None,
        extra_prompt: Optional[str] = '',
        mode: Literal["contextual", "zero-shot"] = "contextual",
        rotate_state: bool = False,
        expand_state: bool = False,
        log_dir: Optional[str] = None,
        max_action_history_len: Optional[int] = 3  # New parameter for history length
    ):
        self.model_name = model_name
        self.env = env
        self.vgdl_rules = vgdl_rules
        self.mode = mode
        self.extra_prompt = extra_prompt
        self.rotate_state = rotate_state
        self.expand_state = expand_state
        self.max_action_history_len = max_action_history_len # Store the max length
        self.llm_client = create_client_from_config(model_name)
        self.reflection_mgr = ReflectionManager()
        self.position_history = []
        self.sprite_map = {}
        self.total_reward = 0.0
        self.state_history = []
        self.action_history = []
        self.last_state = initial_state
        self.winner = None

        try:
            self.action_map = {
                i: env.unwrapped.get_action_meanings()[i]
                for i in range(env.action_space.n)
            }
        except AttributeError:
            self.action_map = {i: f"ACTION_{i}" for i in range(env.action_space.n)}

        game_name = getattr(env.unwrapped.spec, "id", "UnknownEnv")
        self.logger = PromptLogger(model_name=self.model_name, game_name=game_name, log_dir=log_dir)

        if self.mode == "contextual":
            self.llm_client.clear_history()
            static_prompt = build_static_prompt(
                vgdl_rules=self.vgdl_rules,
                action_map=self.action_map,
                optional_prompt=self.extra_prompt
            )


            self.llm_client.set_system_prompt(static_prompt)

    def select_action(
        self,
        current_state: str,
        current_position: Optional[str] = None,
        current_image_path: Optional[str] = None,
        last_image_path: Optional[str] = None,
        sprite_map: Optional[dict] = None,
        extra_prompt: Optional[str] = None,
        plan: Optional[str] = None
    ) -> int:
        # if self.rotate_state:
        #     rotated_state = rotate_ascii_left(current_state)
        #     current_state = f"=== Original State ===\n{current_state}\n\n=== Rotated State (Y-axis emphasis) ===\n{rotated_state}"

        self.state_history.append(current_state)
        self.position_history.append(current_position)

        config = get_profile_config(self.model_name)
        # Get the appropriate model name for tokenizer/logging, 
        # preferring "actual_model_name" for Portkey profiles, else "model" from the config.
        # self.model_name is the profile name (e.g., "portkey-4o-mini")
        model_for_tokenizer = config.get("actual_model_name", config.get("model"))
        if not model_for_tokenizer:
            # Fallback if neither "actual_model_name" nor "model" is in the profile,
            # though this should ideally be caught by config validation earlier.
            # Using self.model_name (profile name) as a last resort might not be ideal for tokenization.
            print(f"Warning: Neither 'actual_model_name' nor 'model' found in profile '{self.model_name}'. Using profile name for tokenizer.")
            model_for_tokenizer = self.model_name # Or a default like "gpt-3.5-turbo" if that's safer for tiktoken

        if self.mode == "zero-shot":
            self.llm_client.clear_history()
            prompt = build_static_prompt(
                vgdl_rules=self.vgdl_rules,
                action_map=self.action_map,
                optional_prompt=extra_prompt
            )
            prompt += "\n\n" + build_dynamic_prompt(
                current_ascii=current_state,
                current_image_path=current_image_path,
                avatar_position=current_position,
                action_map=self.action_map,
                sprite_mapping=sprite_map,
                rotate=self.rotate_state,
                expanded=self.expand_state,
                plan=plan
            )
        else:
            previous_state = self.state_history[-2] if len(self.state_history) >= 2 else None
            last_position = self.position_history[-2] if len(self.position_history) >= 2 else current_position
            
            # Truncate action history before passing to build_dynamic_prompt
            effective_action_history = self.action_history
            if self.max_action_history_len is not None and len(self.action_history) > self.max_action_history_len:
                effective_action_history = self.action_history[-self.max_action_history_len:]

            prompt = build_dynamic_prompt(
                current_ascii=current_state,
                last_ascii=previous_state,
                current_image_path=current_image_path,
                avatar_position=current_position,
                last_position=last_position,
                action_map=self.action_map,
                action_history=effective_action_history, # Use truncated history
                reflection_manager=self.reflection_mgr,
                logger=self.logger,
                llm_model_name=model_for_tokenizer, # Use the resolved model name
                sprite_mapping=sprite_map,
                rotate=self.rotate_state,
                expanded=self.expand_state,
                plan=plan
            )

        # print(prompt)

        response = self.llm_client.query(prompt, image_path=current_image_path)
        self.logger.log_response(response)
        action, _ = parse_action_from_response(response, self.action_map)
        self.action_history.append(action)
        self.last_state = current_state
        print(response)
        return action

    def update(self, action: int, reward: float, winner=None):
        self.total_reward += reward
        step = len(self.reflection_mgr.step_log)
        self.reflection_mgr.log_step(step=step, action=action, reward=reward)
        if winner is not None:
            self.winner = winner

    def save_logs(self):
        self.logger.save()
        self.llm_client.save_history(self.logger.game_name)

    def export_analysis(self, output_dir: str):
        generate_full_analysis_report(
            reflection_manager=self.reflection_mgr,
            states=self.state_history,
            output_dir=output_dir,
            winner=self.winner
        )

    def clear_history(self):
        self.llm_client.clear_history()

class LLMPlanner:
    def __init__(self, model_name: str, vgdl_rules: str):
        self.model_name = model_name
        self.llm_client = create_client_from_config(model_name)
        self.env = None
        self.action_map = {}
        self.vgdl = vgdl_rules
        self.state_history = []
        self.strategy_history = ''

    def initialize(self, env) -> None:
        self.env = env
        if hasattr(env.unwrapped, "get_action_meanings"):
            self.action_map = {
                i: env.unwrapped.get_action_meanings()[i]
                for i in range(env.action_space.n)
            }
        else:
            self.action_map = {
                i: f"ACTION_{i}"
                for i in range(env.action_space.n)
            }

    def clear_history(self):
        self.llm_client.clear_history()

    def query(self, image_path: Optional[str] = None,
              current_state: Optional[str] = '', action_history: Optional[int] = None,
              current_position: Optional[Tuple[int, int]] = None, sprite_mapping: Optional[dict] = None, prompt: Optional[str] = '') -> str:

        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(current_state)

        current_location=''

        base_prompt = (
            "Generate ABSTRACT OBJECTIVE-ORIENTED strategies using this framework:\n"
            "1. SYMBOL SEMANTICS: Use ONLY symbol names from the sprite mapping (e.g. 'door' not '%')\n"
            "2. MECHANICAL PURPOSE: Focus on how symbol types interact (e.g. 'keys open doors')\n"
            "3. ZONE PROGRESSION: Describe objectives by area features (e.g. 'eastern laser zone')\n"
            "4. FAILURE RECOVERY: If stuck, switch symbol type priorities with **Alert**\n\n"
            "Forbidden in responses:\n"
            "- Coordinates (x=.../y=...)\n"
            "- Directional commands (left/right/up/down)\n"
            "- Explicit positions (column/row)\n\n"
            "Required structure:\n"
            "1. CURRENT CAPABILITY: What symbol interactions are possible now?\n"
            "2. STRATEGIC CHOICE: Which symbol type best enables progression?\n"
            "3. EXECUTION PRINCIPLE: How should interactions be performed?\n"
            "   Example: 'Batch-process all nearby keys before approaching doors'"
        )
        last_strategy = self.strategy_history
        format_prompt = (
            "\nFormat response as: "
            "```**<symbol_type> strategy: <mechanism>** with/without **Alert**```\n"
            "Example: **door strategy: Collect keys to bypass gate** \n If you recieved a BAD feedback, you MUST revise your strategy."
        )

        sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()]
        sprite_mapping_prompt = ("=== Sprite Mapping ===\n" + "\n".join(sprite_lines))
        prev_state_prompt = '\n======Previous State=======\n' + prev_state
        current_state_prompt = '\n=======Current State========\n' + current_state
        action_history_text = '\n=====Action Sequence=====\n' + f'{action_history}\n'
        if current_position:
            current_location = (
                '\n======Current Location=======\n'
                f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
                "Coordinate system: X+ → Right, Y+ → Down\n"
                "Walls block movement\n"
            )
        evaluation = '\n======Evaluator Feedback=======\n' + prompt

        full_prompt = (
            sprite_mapping_prompt + '\n' +
            prev_state_prompt + '\n' +
            current_state_prompt +
            current_location +
            action_history_text + '\n' +
            base_prompt +'\n' +
            evaluation +'\n' +
            last_strategy +'\n' +
            format_prompt
        )
        self.strategy_history = ''

        response = self.llm_client.query(full_prompt, image_path=image_path)
        return response
        # matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        # for match in matches:
        #     self.strategy_history += match
        #     print(match.strip())
        # return match if matches else ""

    def save_logs(self):
        self.logger.save()
        self.llm_client.save_history(self.logger.game_name)


class LLMEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm_client = create_client_from_config(model_name)
        self.state_history = []
        self.reward_history = []

    def clear_history(self):
        self.llm_client.clear_history()

    def query(
        self,
        current_state: str,
        # last_state: Optional[str] = None,
        action_taken: Optional[int] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = False,
        current_position: Optional[Tuple[int, int]] = None,
        sprite_mapping: Optional[dict] = None,
        image_path: Optional[str] = None
    ) -> str:
        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(current_state)

        sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()] if sprite_mapping else []
        sprite_mapping_prompt = ("=== Sprite Mapping ===\n" + "\n".join(sprite_lines)) if sprite_lines else ""

        last_state_prompt = '\n======Previous State=======\n' + (prev_state or 'None')
        current_state_prompt = '\n=======Current State========\n' + current_state

        current_location_prompt = (
            '\n======Current Location=======\n'
            f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
            "Coordinate system: X+ → Right, Y+ → Down\n"
            "Walls block movement\n"
        ) if current_position else ""

        action_info = f"Action Taken: {action_taken}" if action_taken is not None else "Action Taken: None"
        reward_info = f"Reward Received: {reward}" if reward is not None else "Reward: None"
        self.reward_history.append(reward)
        done_info = f"Game Done: {done}"

        action_summary = (
            '\n=====Action Summary=====\n'
            f"{action_info}\n"
            f"{reward_info}\n"
            f"{done_info}"
        )

        # Base evaluation instruction
        base_prompt = (
                        "\nEvaluate the agent's last action with STRICT classification:\n"
            "First, decide if the action was GOOD or BAD based on:\n"
            "- EFFECTIVENESS: Did it progress toward winning?\n"
            "- RISK: Did it expose the agent to danger?\n"
            "- STRATEGIC FIT: Was it aligned with objectives?\n\n"
            "Rules:\n"
            "- You MUST classify as either GOOD or BAD.\n"
            "- Then briefly explain WHY such as blocking by a wall.\n"
            "- Format your entire response as:\n"
            f"```Evaluation: <GOOD or BAD> \nFeedback: <your reasoning and strategy> with a reward of {reward} from the environment```"
        )

        full_prompt = (
            sprite_mapping_prompt + '\n' +
            last_state_prompt + '\n' +
            current_state_prompt + '\n' +
            current_location_prompt + '\n' +
            action_summary + '\n' +
            base_prompt
        )

        response = self.llm_client.query(full_prompt, image_path=image_path)
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        for match in matches:
            print(match.strip())
        return matches[-1] if matches else ""

    def save_logs(self):
        # Placeholder for now
        pass
