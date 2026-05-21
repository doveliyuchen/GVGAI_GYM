import re
import time
from typing import Optional, Any, Tuple, Literal
from llm.client import create_client_from_config
from llm.utils.agent_components import parse_action_from_response
from llm.utils.build_prompt import build_static_prompt, build_dynamic_prompt, PromptLogger, ReflectionManager
from llm.utils.game_analysis import generate_full_analysis_report


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
        max_action_history_len: Optional[int] = 3,
        max_retries: int = 5,
        base_retry_wait: int = 60,
    ):
        self.model_name = model_name
        self.env = env
        self.vgdl_rules = vgdl_rules
        self.mode = mode
        self.extra_prompt = extra_prompt
        self.rotate_state = rotate_state
        self.expand_state = expand_state
        self.max_action_history_len = max_action_history_len
        self.max_retries = max_retries
        self.base_retry_wait = base_retry_wait
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
            static_prompt = build_static_prompt(
                vgdl_rules=self.vgdl_rules,
                action_map=self.action_map,
                optional_prompt=self.extra_prompt
            )
            self.llm_client.set_system_prompt(static_prompt)

    @staticmethod
    def _is_non_retriable_error(err_text: str) -> bool:
        """Return True for errors where waiting/retrying is usually pointless."""
        text = (err_text or "").lower()
        return (
            "http 400" in text
            or "http 401" in text
            or "http 403" in text
            or "http 404" in text
            or "invalid request" in text
            or "context length" in text
            or "maximum context" in text
        )

    def _default_action(self) -> Tuple[int, str]:
        action = 0
        return action, self.action_map.get(action, "ACTION_0")

    def _query_with_retry(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Query LLM with retry on API errors/exceptions and empty responses."""
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.query(prompt, image_path=image_path)
            except Exception as exc:
                err_text = f"Error: Exception during LLM query: {exc}"
                self.logger.log_response(err_text)
                print(f"[API ERROR] Attempt {attempt + 1}/{self.max_retries}: {err_text}")
                if self._is_non_retriable_error(str(exc)):
                    print("[API ERROR] Non-retriable request error, skip retries for this step.")
                    return ""
                if attempt < self.max_retries - 1:
                    wait = self.base_retry_wait * (2 ** attempt)
                    print(f"[RETRY] Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                return ""

            self.logger.log_response(response)

            if response and not response.strip().startswith("Error:") and response.strip():
                return response

            if response and response.strip().startswith("Error:"):
                print(f"[API ERROR] Attempt {attempt + 1}/{self.max_retries}: {response.strip()}")
                if self._is_non_retriable_error(response):
                    print("[API ERROR] Non-retriable response error, skip retries for this step.")
                    return ""
            else:
                print(f"[WARNING] Empty response from {self.model_name}, skipping step.")
                return ""

            if attempt < self.max_retries - 1:
                wait = self.base_retry_wait * (2 ** attempt)
                print(f"[RETRY] Waiting {wait}s...")
                time.sleep(wait)

        return ""

    def select_action(
        self,
        current_state: str,
        current_position: Optional[Tuple[int, int]] = None,
        current_image_path: Optional[str] = None,
        last_image_path: Optional[str] = None,
        sprite_map: Optional[dict] = None,
        extra_prompt: Optional[str] = None,
        plan: Optional[str] = None
    ) -> int:
        self.state_history.append(current_state)
        self.position_history.append(current_position)

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
                action_history=effective_action_history,
                reflection_manager=self.reflection_mgr,
                logger=self.logger,
                sprite_mapping=sprite_map,
                rotate=self.rotate_state,
                expanded=self.expand_state,
                plan=plan
            )

        action, action_name = self._default_action()
        response = self._query_with_retry(prompt, image_path=current_image_path)
        if response:
            action, action_name = parse_action_from_response(response, self.action_map)
            print(f"Response: {response}")
            print(f"Action: {action} ({action_name})")

        self.action_history.append(action)
        self.last_state = current_state
        return action

    def update(self, action: int, reward: float, winner=None):
        self.total_reward += reward
        step = len(self.reflection_mgr.step_log)
        self.reflection_mgr.log_step(step=step, action=action, reward=reward)
        if winner is not None:
            self.winner = winner

    def save_logs(self):
        self.logger.save()
        history_path = self.logger.base_log_dir / "chat_history.json"
        self.llm_client.save_history(filepath=str(history_path))

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
            self.action_map = {i: f"ACTION_{i}" for i in range(env.action_space.n)}

    def clear_history(self):
        self.llm_client.clear_history()

    def query(
        self,
        image_path: Optional[str] = None,
        current_state: Optional[str] = '',
        action_history: Optional[str] = None,
        current_position: Optional[Tuple[int, int]] = None,
        sprite_mapping: Optional[dict] = None,
        prompt: Optional[str] = ''
    ) -> str:
        state_text = current_state or ""
        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(state_text)

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
        format_prompt = (
            "\nFormat response as: "
            "```**<symbol_type> strategy: <mechanism>** with/without **Alert**```\n"
            "Example: **door strategy: Collect keys to bypass gate** \n If you recieved a BAD feedback, you MUST revise your strategy."
        )

        sprite_lines = [f"{k} -> '{v}'" for k, v in (sprite_mapping or {}).items()]
        feedback_text = prompt or ""
        action_hist_text = action_history or ""

        full_prompt = (
            "=== Sprite Mapping ===\n" + "\n".join(sprite_lines) + '\n' +
            '\n======Previous State=======\n' + prev_state + '\n' +
            '\n=======Current State========\n' + state_text +
            (
                '\n======Current Location=======\n'
                f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
                "Coordinate system: X+ → Right, Y+ → Down\n"
                "Walls block movement\n"
                if current_position else ''
            ) +
            '\n=====Action Sequence=====\n' + f'{action_hist_text}\n' + '\n' +
            base_prompt + '\n' +
            '\n======Evaluator Feedback=======\n' + feedback_text + '\n' +
            self.strategy_history + '\n' +
            format_prompt
        )
        self.strategy_history = ''

        return self.llm_client.query(full_prompt, image_path=image_path)


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
        action_taken: Optional[int] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = False,
        current_position: Optional[Tuple[int, int]] = None,
        sprite_mapping: Optional[dict] = None,
        image_path: Optional[str] = None
    ) -> str:
        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(current_state)
        self.reward_history.append(reward)

        sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()] if sprite_mapping else []
        sprite_mapping_prompt = ("=== Sprite Mapping ===\n" + "\n".join(sprite_lines)) if sprite_lines else ""

        current_location_prompt = (
            '\n======Current Location=======\n'
            f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
            "Coordinate system: X+ → Right, Y+ → Down\n"
            "Walls block movement\n"
        ) if current_position else ""

        action_info = f"Action Taken: {action_taken}" if action_taken is not None else "Action Taken: None"
        reward_info = f"Reward Received: {reward}" if reward is not None else "Reward: None"

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
            '\n======Previous State=======\n' + (prev_state or 'None') + '\n' +
            '\n=======Current State========\n' + current_state + '\n' +
            current_location_prompt + '\n' +
            '\n=====Action Summary=====\n' + action_info + '\n' + reward_info + '\n' + f"Game Done: {done}" + '\n' +
            base_prompt
        )

        response = self.llm_client.query(full_prompt, image_path=image_path)
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        for match in matches:
            print(match.strip())
        return matches[-1] if matches else ""
