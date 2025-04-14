from abc import ABC, abstractmethod
from typing import Any, Dict
from vgdl_utils import *
import cv2
import numpy as np
from utilities import GifRecorder

class Agent(ABC):
    @abstractmethod
    def initialize(self, env: Any, vgdl_rules: str, lvl_layout: str) -> None:
        pass
    
    @abstractmethod
    def act(self, observation: Any, reward: float, done: bool, info: Dict) -> int:
        pass
    
    def cleanup(self) -> None:
        pass

class Random_CLI_Agent(Agent):
    def initialize(self, env: Any, vgdl_rules: str, lvl_layout: str) -> None:
        self.action_space = env.action_space
        self.vgdl_rules = vgdl_rules
        self.lvl_layout = lvl_layout
        self.step_count = 0
        print(f"Agent initialized with action space: {self.action_space}")
        
        # Print the level layout and rules for reference
        print("\nLevel Layout:")
        print(lvl_layout)
        print("\nVGDL Rules:")
        print(vgdl_rules)
    
    @staticmethod
    def simplify_vgdl_level_ascii(input_string):
        # Split the input by newlines to get rows
        rows = input_string.strip().split('\n')
        
        # Process each row
        processed_rows = []
        for row in rows:
            # Split by commas
            cells = row.split(',')
            
            processed_cells = []
            for cell in cells:
                if cell.strip() == '': processed_cells.append('_')
                else: processed_cells.append(cell[0])
            
            # Join the processed cells for this row
            processed_rows.append(''.join(processed_cells))
            
            # Join the processed rows with newlines
        return '\n'.join(processed_rows)

    def act(self, observation: Any, reward: float, done: bool, info: Dict) -> int:
        input()

        self.step_count += 1
        
        # Print ASCII representation if available
        assert 'ascii' in info
        print(f"\nStep {self.step_count} - ASCII Game State:")

        level_state_ascii = info['ascii']
        level_state_ascii = Random_CLI_Agent.simplify_vgdl_level_ascii(level_state_ascii)

        print(level_state_ascii)
        print(f"Reward: {reward}")
        
        # Select a random action
        action = self.action_space.sample()
        print(f"Taking action: {action}")
        
        return action

class UserControlled_GUI_Agent(Agent):
    def initialize(self, env: Any, vgdl_rules: str, lvl_layout: str) -> None:
        self.env = env
        self.vgdl_rules = vgdl_rules
        self.lvl_layout = lvl_layout
        self.action_space = env.action_space
        self.step_count = 0
        
        # Get action meanings for display
        self.action_meanings = env.unwrapped.get_action_meanings()
        self.num_actions = len(self.action_meanings)
        
        # Create OpenCV window
        self.window_name = "VGDL User Controlled Game"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Print controls information
        print("\nControls:")
        for i, meaning in enumerate(self.action_meanings):
            print(f"  {i}: {meaning}")
        
        # Set FPS
        self.fps = 30
        self.frame_time = 1.0 / self.fps
    
    def render(self, rgb_array, status_text):
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # Create a larger canvas with space below for status information
        h, w = bgr_array.shape[:2]
        status_height = 50  # Height of the status bar
        canvas = np.zeros((h + status_height, w, 3), dtype=np.uint8)
        
        # Copy the game frame to the top of the canvas
        canvas[:h, :] = bgr_array
        
        # Fill the status area with a dark gray background
        canvas[h:, :] = (50, 50, 50)  # Dark gray background for status bar
        
        # Display status information in the status area
        cv2.putText(canvas, status_text, (10, h + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return canvas

    def act(self, observation: Any, reward: float, done: bool, info: Dict) -> int:
        self.step_count += 1
        
        status_text = f"Step: {self.step_count} | Reward: {reward:.2f}"
        rgb_arr = self.env.render(mode='rgb_array')

        canvas = self.render(rgb_arr, status_text)
        
        # Show the combined image
        cv2.imshow(self.window_name, canvas)
        
        # Check for key press with minimal delay
        key = cv2.waitKey(1) & 0xFF
        
        if ord('0') <= key <= ord(str(self.num_actions-1)):
            return key - ord('0')
        
        # Return default action (NOOP) when no key is pressed
        return 0
    
    def cleanup(self):
        cv2.destroyAllWindows()

class GIF_Recording_GUI(UserControlled_GUI_Agent):
    def initialize(self, 
            env: Any, vgdl_rules: str, lvl_layout: str,
            gif_dir: str = os.getcwd(), gif_name: str = "game_recording") -> None:
        # Call the parent initialize method
        super().initialize(env, vgdl_rules, lvl_layout)
        
        # Initialize the recorder
        self.recorder = GifRecorder()

        self.gif_name = gif_name
        self.gif_dir = gif_dir

        print("Recording enabled - gameplay will be saved to", os.path.join(self.gif_dir, self.gif_name) + ".gif")

    def render(self, rgb_array, status_text):
        # Record the frame
        self.recorder.record(rgb_array)

        # Call the parent render method
        return super().render(rgb_array, status_text)
    
    def cleanup(self):
        # Save the GIF using the recorder
        self.recorder.save(self.gif_dir, self.gif_name)
        
        # Call the parent cleanup method
        super().cleanup()

if __name__ == "__main__":
    # Initialize the game environment
    env, vgdl_rules, lvl_layout = get_game_env('gvgai-aliens-lvl0-v0')
    
    agent = Random_CLI_Agent()
    agent.initialize(env, vgdl_rules, lvl_layout) # Initialize the agent
    
    total_reward = 0
    done = False

    # Get initial info with ASCII representation
    env.reset()
    dummy_action = 0
    observation, reward, done, info  = env.step(dummy_action)
    
    total_reward = total_reward + reward
    
    # Main game loop
    while not done:
        # Get action from agent
        action = agent.act(observation, reward, done, info)
        
        # Apply action to environment
        observation, reward, done, info = env.step(action)
        
        # Update total reward
        total_reward += reward

    # Give the agent one more chance to learn from the final state
    agent.act(observation, reward, done, info)
    print(f"Game finished with reward {total_reward}")
    
    # Call the cleanup method before closing the environment
    agent.cleanup()
    env.close() # Clean up