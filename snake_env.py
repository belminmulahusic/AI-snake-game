import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 40, 20

SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(GRID_HEIGHT * GRID_WIDTH,),  
            dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        #Erstmal fixe Apfel Position
        self.apple_queue = [(33, 5), (15, 10), (7, 3), (20, 15), (5, 13)]
        self.apple = None

    def spawn_apple(self):
        if self.apple_queue:
            self.apple = self.apple_queue.pop(0)
        else:
            self.apple = None
            self.done = True  

 