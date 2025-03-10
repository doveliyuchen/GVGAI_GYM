import os
import gym
import gym_gvgai as gvgai
import numpy as np
import re
import pygame
import matplotlib.pyplot as plt
from llm.client import LLMClient
from collections import defaultdict, Counter
from typing import Iterable, Tuple, Optional
import imageio


def create_directory(base_dir='imgs'):
    """
    Create a directory, and generate a new directory name (e.g., imgs_1, imgs_2, etc.) if the directory already exists.

    Args:
        base_dir (str): The base directory name, default is 'imgs'.

    Returns:
        str: The final created directory path.
    """
    if os.path.exists(base_dir):
        # 如果目录已存在，生成新的目录名
        index = 1
        while True:
            new_dir = f"{base_dir}_{index}"
            if not os.path.exists(new_dir):
                base_dir = new_dir
                break
            index += 1

    # 创建目录
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


class show_state_gif():
    def __init__(self):
        self.frames = []
    def __call__(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, 'GIF', duration = 0.1)


def show_state(env, step, name, info, directory, vgdl_representation=None):
    """
    Render the environment state and save it as an image file.

    Args:
        env: Environment object used to render the image.
        step (int): Current step.
        name (str): Image name.
        info (str): Additional information to display in the title.
        directory (str): Target directory to save the image.
        vgdl_representation: Optional parameter to override the rendering logic.

    Returns:
        str: The file path of the saved image; returns None if saving fails.
    """
    # Render the image
    plt.figure(3)
    plt.clf()
    try:
        img = env.render(mode='rgb_array')
        plt.imshow(img)
    except Exception as e:
        if vgdl_representation:
            img = vgdl_to_image(vgdl_representation)
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "No image", fontsize=14, ha='center')


    plt.title(f"{name} | Step: {step} {info}")
    plt.axis("off")


    path = f'{directory}/{name}_{len(os.listdir(directory)) + 1}.png'
    plt.savefig(path)


    return path if os.path.exists(path) else None

def vgdl_to_image(vgdl_representation):
    pygame.init()
    font = pygame.font.Font(None, 24)
    surface = pygame.Surface((300, 300))
    surface.fill((0, 0, 0))
    for i, line in enumerate(vgdl_representation.split("\n")):
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (10, i * 20))
    return pygame.surfarray.array3d(surface)


class ReflectionManager:
    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_reflection(self, reflection: str):
        """Added a reflection to the history"""
        if reflection:
            self.history.append(reflection)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def get_formatted_history(self) -> str:
        """get the fomatted history"""
        return "\n".join(f"[History reflection{i + 1}] {r}"
                         for i, r in enumerate(self.history))


def parse_vgdl_level(vgdl_level):
    max_width = max(len(row) for row in vgdl_level)
    padded_level = [row.ljust(max_width, ".") for row in vgdl_level]

    avatar_pos = None
    for y, row in enumerate(padded_level):
        if 'A' in row:
            x = row.index('A')
            avatar_pos = (x, y)
            break

    return np.array([list(row) for row in padded_level]), avatar_pos 


class RewardSystem:
    def __init__(self):
        self.action_history = []
        self.reward_history = []
        self.action_efficacy = defaultdict(list)
        self.consecutive_zero_threshold = 3
        # self.window_size = window_size
        self.total_reward = 0.0

    def update(self, action: int, reward: float):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.action_efficacy[action].append(reward)
        self.total_reward += reward

        # if len(self.action_history) > self.window_size:
        #     removed_action = self.action_history.pop(0)
        #     self.reward_history.pop(0)
        #     if self.action_efficacy[removed_action]:
        #         self.action_efficacy[removed_action].pop(0)

    # disable the zero streak function
    def generate_guidance(self) -> str:
        # if self.get_zero_streak() >= self.consecutive_zero_threshold:
        #     return self._zero_reward_analysis()
        return self._performance_summary()

    def get_zero_streak(self) -> int:
        return next(
            (i for i, r in enumerate(reversed(self.reward_history)) if r != 0),
            len(self.reward_history))

    # def _zero_reward_analysis(self) -> str:
    #     recent_actions = self.action_history[-self.consecutive_zero_threshold:]
    #     action_stats = {a: (np.mean(self.action_efficacy[a]), len(self.action_efficacy[a]))
    #                     for a in set(recent_actions)}
    #     return f"""
    # Zero reward for {self.get_zero_streak()} consecutive times.
    # Recent action sequence: {recent_actions}
    # Performance analysis:
    # {chr(10).join(f'- Action {a}: Average reward {avg:.2f} (Attempt count {count})'
    #               for a, (avg, count) in action_stats.items())}
    # It is recommended to try new action combinations or check rule compliance. 
    # Note that there may be a certain delay in rewards. 
    # Combining multiple actions (changing positions) will increase the probability of obtaining a reward.

    # Avoid repetitive location
    # Get the reward
    # The final goal is win the game
    # """

    # def _performance_summary(self) -> str:
    #     top_actions = sorted([(a, np.mean(r)) for a, r in self.action_efficacy.items() if r],
    #                          key=lambda x: x[1], reverse=True)[:3]
    #     return f"""
    # Best action:
    # {chr(10).join(f'- action{a}: average reward{reward:.2f}' for a, reward in top_actions)}
    # """




def build_enhanced_prompt(vgdl_rules: str,
                          state: str,
                          last_state: str,
                          action_map: dict,
                          reward_system: RewardSystem,
                          reflection_mgr: ReflectionManager,
                          current_image_path: Optional[str] = None,
                          last_image_path: Optional[str] = None,
                          reflection = True, reward = False ) -> str:
    """Build English prompt with layout and reflection history"""

    last_action = None
    last_reward = None
    if reward_system.action_history:
        last_action = reward_system.action_history[-1]
        last_reward = reward_system.reward_history[-1]
    last_action_desc = action_map.get(last_action, "None") if last_action is not None else "None"
    last_reward_desc = action_map.get(last_reward, "None") if last_reward is not None else "None"

    formating = f'''
    You are controlling avatar A, try to win the game with actions. 
    Goal: Try to interact with the game by analyzing the game state and learn to play and win it.
    Respond in this format:
    Action: <action number>'''

    reflection_format =  None
    reflection_section = ""
    if reflection:
        reflection_format = ''' Reflection: ```<your strategy reflection>```  '''

        if reflection_mgr.history:
            reflection_section = f"\n=== Reflection History ===\n{reflection_mgr.get_formatted_history()}"

    base =f'''
    === Game Rules ===
    {vgdl_rules}


    === Last State ===
    You are "avatar"
    {last_state}
    Image of the last state is attached: {last_image_path}


    === Current State ===
    You are "avatar"
    {state}
    Image of the current state is attached: {current_image_path}

    === Last Action ===
    {last_action} ({last_action_desc})

    === Available Actions ===
    {chr(10).join(f'{k}: {v}' for k, v in action_map.items())}
    '''

    if reward:
        reward_prompt = f'''
         === Last Reward ===
         {last_reward} ({last_reward_desc})
        '''
    else:
        reward_prompt = ''

    guidance = f'''
    * Strategic Priority:

    State Awareness: Recognize the differences between states and identify the optimal action to transition to a desired state. This process operates within a fully defined Markov framework.
    Action Consistency: Be mindful that your current decision may negate the effects of the previous action. Aim to maintain consistency and avoid contradictory moves.
    Meaningful Decisions: Ensure that your actions are purposeful. For instance, moving against a wall is unproductive and should be avoided.
    Reflect on these guidelines and formulate your own strategic priorities, presenting them in a clear and structured format.
        '''

    return f"{formating}{reflection_format}{base}{reward_prompt}{reflection_section}\n{guidance}"


def query_llm(llm_client: LLMClient,
              vgdl_rules: str,
              current_state: str,
              last_state: str,
              action_map: dict,
              reward_system: RewardSystem,
              reflection_mgr: ReflectionManager,
              step: int,
              current_image_path: Optional[str] = None,
              last_image_path: Optional[str] = None,
              reflection = True,
              reward = False) -> Tuple[int, str]:
    prompt = build_enhanced_prompt(vgdl_rules, current_state, last_state, action_map,
                                   reward_system, reflection_mgr, current_image_path, last_image_path,reflection,reward)
    try:
        response = llm_client.query(prompt, image_path=current_image_path)

        action_match = re.search(r"Action:\s*(\d+)", response)
        reflection_match = re.search(r"Reflection:\s*```(.*?)```", response, re.DOTALL)

        action = int(action_match.group(1)) if action_match else 0
        reflection = reflection_match.group(1).strip() if reflection_match else ""

        action = action if action in action_map else 0

        print(f"\n=== Step {step} ===")
        print(f"Selected Action: {action} ({action_map.get(action, 'Unknown')})")
        if reflection:
            print(f"Strategy Reflection: {reflection[:700]}...")

        return action, reflection

    except Exception as e:
        print(f"LLM query error: {str(e)}")
        return 0, " "


def generate_report(system: RewardSystem, step: int, dir) -> str:
    print(f"\n=== Game analysis ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {system.total_reward}")
    print(f"Zero Streak: {system.get_zero_streak()}")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(system.reward_history)
    plt.title("Reward trend")

    plt.subplot(122)
    action_dist = Counter(system.action_history)
    plt.bar(action_dist.keys(), action_dist.values())
    plt.title("Action distribution")
    plt.savefig(dir+"game_analysis.png")


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games")
    llm_list = ["qwen","openai"]

    for game in os.listdir(full_path):
        env_name = "gvgai-"+game[:-3]+"-lvl0-v0"

        env = gvgai.make(env_name)
        state = env.reset()
        done = False

        # VGDL rule
        game_name = env.spec.id.replace("gvgai-", "").split("-")[0] + "_v0"

        game_dir = os.path.join(os.path.dirname(current_path), "gym_gvgai", "envs", "games", game_name)

        vgdl_rule_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                               if f.endswith(".txt") and "lvl" not in f), None)
        level_layout_file = next((os.path.join(game_dir, f) for f in os.listdir(game_dir)
                                  if f.endswith(".txt") and "lvl" in f), None)



        with open(vgdl_rule_file, "r") as f:
            vgdl_rules = f.read()
        with open(level_layout_file, "r") as f:
            level_layout = f.read()


        vgdl_grid, avatar_pos = parse_vgdl_level(level_layout)
        h, w = vgdl_grid.shape

        available_actions = list(range(env.action_space.n))
        try:
            action_mapping = {i: env.unwrapped.get_action_meanings()[i] for i in available_actions}
        except AttributeError:
            action_mapping = {i: f"Action {i}" for i in available_actions}


        for llm in llm_list:

            llm_client = LLMClient(llm)
            state = env.reset()
            done = False
            reflection_mgr = ReflectionManager()
            reward_system = RewardSystem()
            total_reward = 0
            step_count = 0
            info = None
            image_path = None
            img = show_state_gif()
            last_state = None
            game_state = vgdl_grid
            last_state_img = None
            game_state_img = None
            dir = create_directory("imgs/"+game_name)

            try:

                while not done:

                    action, reflection = query_llm(llm_client, vgdl_rules,game_state, last_state, action_mapping, reward_system,
                                                   reflection_mgr, step_count, game_state_img, last_state_img, reflection = False)
                    next_state, reward, done, info = env.step(action)
                    reward_system.update(action, reward)
                    last_state = game_state
                    game_state = info["ascii"]


                    total_reward += reward
                    print(f"Received Reward: {reward}")

                    last_state_img = game_state_img

                    game_state_img= show_state(env, step_count,
                               "enhanced_agent",
                               f"Reward: {reward} | Action: {action}",dir,
                               game_state)
                    img(env)
                    winner = info['winner']
                    step_count += 1


            finally:

                env.close()
                try:
                    img.save(dir+"_"+llm)
                except:
                    print("cannot save")
                with open("game_logs.txt", mode="a") as f:
                    f.write(f"game_name: {game_name}, step_count: {step_count}, winner: {winner}, api: {llm}\n")
                generate_report(reward_system, step_count,dir+"_"+llm)


