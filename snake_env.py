import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 50, 50

SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.apple = None

        self.reset()

    def spawn_apple(self):
        while True:
            self.apple = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            if self.apple not in self.snake:
                break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.steps = 0
        self.spawn_apple()
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        dir_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        new_direction = dir_map[action]

        # Kein Zurückgehen erlauben
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction and len(
            self.snake
        ) > 1:
            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        self.steps += 1

        # Kollision mit Wand oder sich selbst
        if self._is_collision(new_head):
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        dist_old = abs(self.apple[0] - head[0]) + abs(self.apple[1] - head[1])
        dist_new = abs(self.apple[0] - new_head[0]) + abs(self.apple[1] - new_head[1])

        reward = 0.005

        self.snake.insert(0, new_head)
        
        if new_head == self.apple:
            reward += 1.0
            self.spawn_apple()
        else:
            self.snake.pop()

            if dist_new < dist_old:
                reward += 0.1
            else:
                reward -= 0.05

        if self.apple is None:
            self.done = True
            reward += 1.0

        return self._get_obs(), reward, self.done, False, {}

    def _is_collision(self, new_head):
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        ):
            return True
        
        if new_head in self.snake:
            return True
        
        return False

    def _get_obs(self):
        head = self.snake[0]

        dir_vec = np.array([
            self.direction == (0, -1),
            self.direction == (1, 0),
            self.direction == (0, 1),
            self.direction == (-1, 0),
        ], dtype=np.uint8)

        up = (head[0], head[1] - 1)
        right = (head[0] + 1, head[1])
        down = (head[0], head[1] + 1)
        left = (head[0] - 1, head[1])
        
        danger_vec = np.array([
            self._is_collision(up),
            self._is_collision(right),
            self._is_collision(down),
            self._is_collision(left),
        ], dtype=np.uint8)

        if self.apple is not None:
            food_vec = np.array([
                self.apple[1] < head[1],
                self.apple[0] > head[0],
                self.apple[1] > head[1],
                self.apple[0] < head[0],
            ], dtype=np.uint8)
        else:
            food_vec = np.zeros(4, dtype=np.uint8)

        obs = np.concatenate((dir_vec, danger_vec, food_vec))
        return obs

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.window.fill((53, 53, 53))

        for x, y in self.snake:
            pygame.draw.rect(
                self.window,
                (120, 209, 142),
                pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=10,
            )

        if self.apple:
            ax, ay = self.apple
            pygame.draw.rect(
                self.window,
                (255, 80, 80),
                pygame.Rect(ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=15,
            )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
