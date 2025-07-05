import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

CELL_SIZE = 15
GRID_WIDTH, GRID_HEIGHT = 40, 20

SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

BLACK = (30, 30, 30)
GREEN = (120, 209, 142)
RED = (255, 80, 80)
WHITE = (240, 240, 240)

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        self.apple = None 

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.int32),
            high=np.array([GRID_WIDTH - 1, GRID_HEIGHT - 1, GRID_WIDTH - 1, GRID_HEIGHT - 1], dtype=np.int32),
            shape=(4,),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(4)

        self.reset()

    def _get_obs(self):
        
        head = self.snake[0]
        apple_pos = self.apple if self.apple else (0, 0) # Fallback, falls kein Apfel
        return np.array([head[0], head[1], apple_pos[0], apple_pos[1]], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.steps = 0
        self.score = 0
        self.done = False

        self.apples_queue = [(33, 5), (15, 10), (7, 3), (20, 15), (5, 12)]
        self.current_apple = self.apples_queue.pop(0) if self.apples_queue else None
        print("Appel: ", self.current_apple)

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation

    def step(self, action):
        dir_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        new_direction = dir_map[action]


        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction and len(
            self.snake
        ) > 1:

            new_direction = self.direction
        else:
            self.direction = new_direction

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        self.steps += 1

        reward = 0.0

        if self._is_collision(new_head):
            self.done = True
            reward = -100.0

        else:
            self.snake.insert(0, new_head)

            if self.current_apple and new_head == self.current_apple:
                reward = +10.0
                self.score += 1
                if self.apples_queue:
                    self.current_apple = self.apples_queue.pop(0)
                else:
                    self.current_apple = None

            else:
                self.snake.pop()
                reward += 0.1

        if self.current_apple is None and not self.done:
            self.done = True
            reward += 100.0

        truncated = False

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation, reward, self.done, truncated

    def _is_collision(self, new_head):
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        ):
            return True

        if new_head in self.snake[1:]:
             return True

        return False

    def spawn_apple(self):
        if self.apples_queue:
            self.current_apple = self.apples_queue.pop(0)
        else:
            self.current_apple = None

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20)
            pygame.display.set_caption("Snake Game RL Environment") # Titel des Fensters setzen

        self.window.fill(BLACK) # Hintergrund füllen

        # Schlange zeichnen
        for i, segment in enumerate(self.snake):
            color = GREEN
            if i == 0: # Kopf der Schlange
                color = (0, 200, 80) # Etwas dunkleres Grün für den Kopf
            pygame.draw.rect(
                self.window,
                color,
                pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=10,
            )

        # Apfel zeichnen
        if self.current_apple:
            ax, ay = self.current_apple
            pygame.draw.rect(
                self.window,
                RED,
                pygame.Rect(ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=15,
            )

        # Punktestand anzeigen
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.window.blit(score_text, (10, 10))

        # Aktuelle Schritte anzeigen
        steps_text = self.font.render(f"Steps: {self.steps}", True, WHITE)
        self.window.blit(steps_text, (10, 40))


        pygame.display.flip() # Bildschirm aktualisieren
        self.clock.tick(self.metadata["render_fps"]) # FPS einhalten

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None

if __name__ == '__main__':
    env = SnakeEnv(render_mode='human')

    obs = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    episode_steps = 0

    print("Starte Snake RL Umgebung. Drücken Sie Pfeiltasten/WASD zur Steuerung oder warten Sie auf zufällige Aktionen.")

    while not terminated and not truncated:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = 1
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    action = 3
                
        if action is None:
            action = env.action_space.sample()

        obs, reward, terminated, truncated = env.step(action)
        total_reward += reward
        episode_steps += 1

    print(f"\nEpisode beendet nach {episode_steps} Schritten. Gesamtbelohnung: {total_reward:.2f}")
    env.close()