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
    apple_positions = [(33, 5), (15, 10), (7, 3), (20, 15), (5, 13)]

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(GRID_HEIGHT * GRID_WIDTH,), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Erstmal fixe Apfel Position
        self.apple_queue = self.apple_positions.copy()
        self.apple = None

        self.reset()

    def spawn_apple(self):
        if self.apple_queue:
            self.apple = self.apple_queue.pop(0)
        else:
            self.apple = None
            self.done = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.apple_queue = self.apple_positions.copy()
        self.spawn_apple()
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

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
        if (
            new_head in self.snake
            or new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        ):
            self.done = True
            return self._get_obs(), 0, True, False, {}

        if new_head == self.apple:
            self.snake.insert(0, new_head)
            self.spawn_apple()
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

        if self.apple is None:
            self.done = True

        return self._get_obs(), 0, self.done, False, {}

    def _get_obs(self):
        obs = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for x, y in self.snake:
            obs[y, x] = 1
        if self.apple:
            obs[self.apple[1], self.apple[0]] = 2
        return obs.flatten()

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
