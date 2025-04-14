import os
import pygame
import matplotlib.pyplot as plt
import imageio
from collections import deque

class ReflectionManager:
    def __init__(self, max_history=3):
        self.history = deque(maxlen=max_history)

    def add_reflection(self, reflection: str):
        if reflection: self.history.append(reflection)

    def __str__(self) -> str:
        return "\n".join(f"[History Reflection {i + 1}] {r}" for i, r in enumerate(self.history))

class GifRecorder:
    def __init__(self):
        self.frames = []
        
    def record(self, rgb_arr):
        self.frames.append(rgb_arr)

    def save(self, dir_path, gif_name):
        os.makedirs(dir_path, exist_ok=True)
        gif_name = gif_name + '.gif'

        full_path = os.path.join(dir_path, gif_name)

        # Save to the full path
        imageio.mimsave(full_path, self.frames, 'GIF', duration=0.1)

def create_directory(dir_name='imgs'):
    # Create a directory, and generate a new directory name (e.g., imgs_1, imgs_2, etc.) if the directory already exists.
    # Returns the final created directory path.

    orig_base = dir_name
    index = 1
    while os.path.exists(dir_name):
        dir_name = f"{orig_base}_{index}"
        index += 1
    
    os.makedirs(dir_name)
    return dir_name

def save_snapshot(rgb_arr, plt_title, path):
    # Render the image
    plt.figure(3)
    plt.clf()
    plt.imshow(rgb_arr)

    plt.title(plt_title)
    plt.axis("off")

    plt.savefig(path)

    if not os.path.exists(path):
        raise Exception("Failed to save snapshot!")

    return path

def snapshot_env(env, step, snapshot_name, dir):
    # Create directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    
    rbg_arr = env.render(mode='rgb_array')
    plt_title = f"{snapshot_name} | Step: {step}"
    path = f'{dir}/{snapshot_name}_{len(os.listdir(dir)) + 1}.png'

    path_to_snap = save_snapshot(rbg_arr, plt_title, path)
    return path_to_snap
    
def render_text_as_image(ascii_repr):
    pygame.init()
    
    font = pygame.font.Font(None, 24)
    surface = pygame.Surface((300, 300))
    surface.fill((0, 0, 0))

    for i, line in enumerate(ascii_repr.split("\n")):
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (10, i * 20))
    
    return pygame.surfarray.array3d(surface)

def render_vgdl_ascii_as_img(vgdl_ascii_repr):
    render_text_as_image(vgdl_ascii_repr)