import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 40, 40

SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
OFFSET = 260
FONT_PATH = "assets/VCR_OSD_MONO_1.001.ttf"

OBSTACLE_COLOR = (100, 100, 100) 

OBSTACLE_PATTERNS = [
    # L-Formen
    [(0, 0), (1, 0), (0, 1)], 
    [(0, 0), (-1, 0), (0, 1)],
    [(0, 0), (1, 0), (0, -1)],
    [(0, 0), (-1, 0), (0, -1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (0, -1), (1, -1)],
    [(0, 0), (0, -1), (-1, -1)],
    [(0, 0), (1, 0), (0, 1), (0, 2)], 
    [(0, 0), (-1, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (0, -1), (0, -2)],
    [(0, 0), (-1, 0), (0, -1), (0, -2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (-1, 1), (-1, 2)],
    [(0, 0), (0, -1), (1, -1), (1, -2)],
    [(0, 0), (0, -1), (-1, -1), (-1, -2)],

    # Gerade Blöcke
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1)],
    
    # 2x2 Blöcke
    [(0, 0), (1, 0), (0, 1), (1, 1)],
]


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, num_obstacles=8):
        super(SnakeEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(57,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        self.apple = None
        self.obstacles = []
        self.num_obstacles = num_obstacles
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.steps = 0
        self.score = 0
        self.spawn_apple()
        self.generate_obstacles()
        self.done = False
        return self._get_obs(), {}


    def _get_all_obstacle_blocks(self):
        all_blocks = []
        for obstacle in self.obstacles:
            all_blocks.extend(obstacle)
        return all_blocks

    def spawn_apple(self):
        all_obstacle_blocks = self._get_all_obstacle_blocks()
        all_occupied = set(self.snake).union(all_obstacle_blocks)
        while True:
            self.apple = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            if self.apple not in all_occupied:
                break

    def generate_obstacles(self):
        self.obstacles = []
        all_existing_occupied = set(self.snake + [self.apple])
        
        while len(self.obstacles) < self.num_obstacles:
            
            pattern = random.choice(OBSTACLE_PATTERNS)
            
            start_x = random.randint(0, GRID_WIDTH - 1)
            start_y = random.randint(0, GRID_HEIGHT - 1)
            
            potential_obstacle = []
            is_valid_placement = True

            for rel_x, rel_y in pattern:
                abs_x, abs_y = start_x + rel_x, start_y + rel_y
                block_coord = (abs_x, abs_y)
                
                
                if (
                    abs_x < 0 or abs_x >= GRID_WIDTH or
                    abs_y < 0 or abs_y >= GRID_HEIGHT or
                    block_coord in all_existing_occupied
                ):
                    is_valid_placement = False
                    break
                potential_obstacle.append(block_coord)
            
            if is_valid_placement:
                self.obstacles.append(potential_obstacle)
                all_existing_occupied.update(potential_obstacle)


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
            self.score += 1
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
        for obstacle_blocks in self.obstacles:
            if new_head in obstacle_blocks:
                return True
        return False

    def _get_obs(self):
        head = self.snake[0]

        dir_vec = np.array([
            self.direction == (0, -1),
            self.direction == (1, 0),
            self.direction == (0, 1),
            self.direction == (-1, 0),
        ], dtype=np.float32)

#        up = (head[0], head[1] - 1)
#        right = (head[0] + 1, head[1])
#        down = (head[0], head[1] + 1)
#        left = (head[0] - 1, head[1])
#        
#        danger_vec = np.array([
#            self._is_collision(up),
#            self._is_collision(right),
#            self._is_collision(down),
#            self._is_collision(left),
#        ], dtype=np.float32)

        if self.apple is not None:
            food_vec = np.array([
                self.apple[1] < head[1],
                self.apple[0] > head[0],
                self.apple[1] > head[1],
                self.apple[0] < head[0],
            ], dtype=np.float32)
        else:
            food_vec = np.zeros(4, dtype=np.float32)
        snake_length_norm = len(self.snake) / (GRID_WIDTH * GRID_HEIGHT)
        snake_length_vec = np.array([snake_length_norm], dtype=np.float32)
        
        fov = []
        for dy in [-1, 0, 1, -2, 2, -3, 3]:
            for dx in [-1, 0, 1, -2, 2, -3, 3]:
                if dx == 0 and dy == 0:
                    continue 
                
                check_x, check_y = head[0] + dx, head[1] + dy
                
                is_dangerous = 0

                if self._is_collision((check_x, check_y)):
                    is_dangerous = 1
                else:
                    is_dangerous = 0

                fov.append(is_dangerous)

        fov_vec = np.array(fov, dtype=np.float32)
        
        obs = np.concatenate((dir_vec, food_vec, snake_length_vec, fov_vec))
        return obs

    def render(self, game_mode):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH + OFFSET, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.Font(FONT_PATH, 30)
            except FileNotFoundError:
                self.font = pygame.font.SysFont("arial", 30)

        self.window.fill((53, 53, 53))
        pygame.draw.rect(self.window, (64, 64, 64), pygame.Rect(0, 0, OFFSET, SCREEN_HEIGHT))
        pygame.draw.rect(self.window, (48, 48, 48), pygame.Rect(OFFSET-20, 0, 20, SCREEN_HEIGHT))

        for x, y in self.snake:
            pygame.draw.rect(
                self.window,
                (120, 209, 142),
                pygame.Rect(x * CELL_SIZE + OFFSET, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=10,
            )

        if self.apple:
            ax, ay = self.apple
            pygame.draw.rect(
                self.window,
                (255, 80, 80),
                pygame.Rect(ax * CELL_SIZE + OFFSET, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=15,
            )

        for obstacle_blocks in self.obstacles:
            for ox, oy in obstacle_blocks:
                pygame.draw.rect(
                    self.window,
                    OBSTACLE_COLOR,
                    pygame.Rect(ox * CELL_SIZE + OFFSET, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    border_radius=5,
                )
                
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (42, 370))

        if game_mode != "none":
            lines = [
                "Press 'SPACE'",
                " to enable ",
                f" {game_mode}-mode"
            ]
            for i, line in enumerate(lines):
                info_msg = self.font.render(line, True, (255, 255, 255))
                self.window.blit(info_msg, (7, 20 + i * 30))

        if self.done:
            final_score_text = self.font.render(f"Final Score: {self.score}", True, (255, 255, 0))
            text_rect = final_score_text.get_rect(center=((SCREEN_WIDTH // 2) + OFFSET, SCREEN_HEIGHT // 2))

            padding = 10
            background_rect = pygame.Rect(
                text_rect.x - padding,
                text_rect.y - padding + 2,
                text_rect.width + 2 * padding,
                text_rect.height + 2 * padding
            )
            
            pygame.draw.rect(self.window, (5, 5, 5), background_rect, border_radius=10)
            self.window.blit(final_score_text, text_rect)

            pygame.display.flip()
            pygame.time.wait(5000)
            return

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
