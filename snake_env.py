##################################################################
#                                                                #
#   Die snake_env.py ist das Herzstück unseres Projekts.         #
#   Hier werden alle wichtigen Aspekte für das Reinforcement     #
#   Learning bereitgestellt.                                     #
#                                                                #
##################################################################

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import collections

# Spielfeldgrößen und Anzeigeeinstellungen
CELL_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = 30, 30
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
OFFSET = 260
FONT_PATH = "assets/VCR_OSD_MONO_1.001.ttf"
OBSTACLE_COLOR = (100, 100, 100) 

# Vordefinierte Hindernismuster (verschiedene Blockformen)
OBSTACLE_PATTERNS = [

    # Gerade Blöcke 2x
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1)],

    # Gerade Blöcke 4x
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # Horizontal
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # Vertikal

    # Gerade Blöcke 8x
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],  # Horizontal
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],  # Vertikal

    # 2x2 Block
    [(0, 0), (1, 0), (0, 1), (1, 1)],

    # 8x2 Blöcke
    [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)],  # Horizontal
    [(0, y) for y in range(8)] + [(1, y) for y in range(8)],  # Vertikal
]

# Bewegungsrichtungen
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]



# SnakeEnv ist eine Snake-Umgebung für Reinforcement Learning, basierend auf der
# Gymnasium API. Des Weiteren wird zum Rendering pygame benutzt.
class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 8}


    # Initialisiert die Snake Umgebung
    # Parameter:
    #   render_mode (str): 'human' oder None, je nach gewünschtem Render-Verhalten
    #   num_obstacles (int): Anzahl der zu generierenden Hindernisse
    #   render_fps (int): Framerate des Renderings bei der Verwendung von Pygame
    def __init__(self, render_mode=None, num_obstacles=8, render_fps=8):
        super(SnakeEnv, self).__init__()

        # Definiert den Aktionsraum für die Gymnasium Umgebung
        self.action_space = spaces.Discrete(3)
        # Definiert den Beobachtungsraum für die Gymnasium Umgebung
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(182,), dtype=np.float32
        )

        self.metadata["render_fps"] = render_fps
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        self.apple = None
        self.obstacles = []
        self.num_obstacles = num_obstacles
        self.action_history = collections.deque(maxlen=3)
        self.apple_img = None
        self.snake_body_img = None
        self.snake_head_images = {}
        self.reset()


    # Setzt die Umgebung zurück und initialisiert die benötigten Variablen
    # Parameter:
    #   seed (int): Optionaler Zufalls-Seed
    #   options (dict): Optionales Dictionary für zusätzliche Optionen
    # Return:
    #   Tuple: Beobachtungsvektor, leeres Info-Dictionary
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5), (4, 5), (3, 5)]
        self.direction = (1, 0)
        self.steps = 0
        self.steps_since_apple = 0
        self.score = 0

        self.action_history.clear()
        for _ in range(self.action_history.maxlen):
            self.action_history.append(-1)
            
        self.generate_obstacles()
        self.spawn_apple()
        self.done = False
        return self._get_obs(), {}


    # Gibt eine Liste aller Hindernis-Koordinaten im Spielfeld zurück
    # Return:
    #   List: Alle Koordinaten aller Hindernis-Blöcke
    def _get_all_obstacle_blocks(self):
        all_blocks = []
        for obstacle in self.obstacles:
            all_blocks.extend(obstacle)
        return all_blocks


    # Spawnt den Apfel an einer freien Position auf dem Spielfeld
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


    # Generiert Hindernisse auf dem Spielfeld mithilfe zufälliger Muster
    def generate_obstacles(self):
            self.obstacles = []
            forbidden_zone = set()
            spawn_distance= 4

            # Verbietet Spawn-Zonen an den Rändern (Weil das zu Bugs und unerreichbaren Äpfeln geführt hat)
            for i in range(GRID_WIDTH):
                for j in range(spawn_distance):
                    forbidden_zone.add((i, j))
                    forbidden_zone.add((i, GRID_HEIGHT - 1 - j))

            for i in range(GRID_HEIGHT):
                for j in range(spawn_distance):
                    forbidden_zone.add((j, i))
                    forbidden_zone.add((GRID_WIDTH - 1 - j, i))

            # Verbietet Spawn-Zonen um die aktuelle Schlange
            for sx, sy in self.snake:
                for dx in range(-spawn_distance, spawn_distance + 1):
                    for dy in range(-spawn_distance, spawn_distance + 1):
                        forbidden_zone.add((sx + dx, sy + dy))

            attempts = 0
            max_attempts = 2000

            # Platziert Hindernismuster zufällig im Spielfeld
            while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
                pattern = random.choice(OBSTACLE_PATTERNS)
                
                start_x = random.randint(0, GRID_WIDTH - 1)
                start_y = random.randint(0, GRID_HEIGHT - 1)
                
                potential_obstacle = []
                is_valid_placement = True

                for rel_x, rel_y in pattern:
                    abs_x, abs_y = start_x + rel_x, start_y + rel_y
                    block_coord = (abs_x, abs_y)
                    
                    if block_coord in forbidden_zone:
                        is_valid_placement = False
                        break
                    potential_obstacle.append(block_coord)
                
                if is_valid_placement:
                    self.obstacles.append(potential_obstacle)

                    # Neue Sperrzone um das Hindernis erzeugen
                    for ox, oy in potential_obstacle:
                        for dx in range(-spawn_distance, spawn_distance + 1):
                            for dy in range(-spawn_distance, spawn_distance + 1):
                                forbidden_zone.add((ox + dx, oy + dy))
                    attempts = 0
                else:
                    attempts += 1


    # Führt einen Schritt in der Umgebung aus, basierend auf der übergebenen Aktion
    # Parameter:
    #   action (int): Die gewählte Aktion (0 = geradeaus, 1 = links drehen, 2 = rechts drehen)
    # Return:
    #   Tuple:
    #     - Beobachtungsvektor nach dem Schritt
    #     - Belohnung (float)
    #     - done (bool): Ob die Episode beendet ist
    #     - truncated (bool): Ob die Episode durch Zeitlimit abgebrochen wurde (aufgrund von Loops)
    #     - info (dict): Zusatzinformationen
    def step(self, action):

        direction_index = DIRECTIONS.index(self.direction)
        if action == 0:
            new_direction_index = direction_index
        elif action == 1:
            new_direction_index = (direction_index - 1) % 4
        elif action == 2:
            new_direction_index = (direction_index + 1) % 4
        self.direction = DIRECTIONS[new_direction_index]
        self.action_history.append(action)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        self.steps += 1

        # Kollision mit Wand oder sich selbst
        if self._is_collision(new_head):
            self.done = True
            return self._get_obs(), -1.0, True, False, {}

        # Distanz zur Nahrung vor und nach dem Schritt
        dist_old = abs(self.apple[0] - head[0]) + abs(self.apple[1] - head[1])
        dist_new = abs(self.apple[0] - new_head[0]) + abs(self.apple[1] - new_head[1])

        # Überlebensbonus
        reward = 0.005

        self.snake.insert(0, new_head)
        
        self.steps_since_apple += 1
        
        if new_head == self.apple:
            # Belohnung fürs Essen
            reward += 1.0
            self.score += 1
            self.steps_since_apple = 0
            self.spawn_apple()
        else:
            self.snake.pop()

            if dist_new < dist_old:
                # Belohnung wenn Bewegung Richtung Apfel
                reward += 0.1
            else:
                # Strafe wenn Bewegung weg vom Apfel
                reward -= 0.05

        if self.apple is None:
            self.done = True
            reward += 1.0

        # Maßnahme zur Vermeidung von Loops (Der Agent soll bestraft und die Episode beendet werden)
        if self.steps_since_apple > 500:
            return self._get_obs(), -1.0, False, True, {}

        return self._get_obs(), reward, self.done, False, {}



    # Überprüft, ob die neue Kopfposition mit Hindernis, Wand oder Körper kollidiert
    # Parameter:
    #   new_head (Tuple[int, int]): Die neue Position des Schlangenkopfs
    # Return:
    #   bool: True, wenn eine Kollision bevorsteht, sonst False
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


    # Gibt den Beobachtungsvektor des aktuellen Zustands zurück
    # Der Beobachtungsvektor wird in der Ausarbeitung genauer beschrieben
    # Return:
    #   np.ndarray: Der kombinierte Beobachtungsvektor als float32
    def _get_obs(self):
        head = self.snake[0]

        # Die aktuelle Richtung der Schlange
        dir_vec = np.array([
            self.direction == (0, -1),
            self.direction == (1, 0),
            self.direction == (0, 1),
            self.direction == (-1, 0),
        ], dtype=np.float32)

        # Die Richtung in der sich der Apfel befindet
        if self.apple is not None:
            food_vec = np.array([
                self.apple[1] < head[1],
                self.apple[0] > head[0],
                self.apple[1] > head[1],
                self.apple[0] < head[0],
            ], dtype=np.float32)
        else:
            food_vec = np.zeros(4, dtype=np.float32)
            
        # Länge der Schlange
        snake_length_norm = len(self.snake) / (GRID_WIDTH * GRID_HEIGHT)
        snake_length_vec = np.array([snake_length_norm], dtype=np.float32)
        
        
        # Sichtfeld der Schlange
        fov = []
        for dy in [-1, 0, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]:
            for dx in [-1, 0, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]:
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
        
        # Normalisierter Abstand zum Apfel
        apple_dist_vec = np.zeros(2, dtype=np.float32)
        if self.apple:
            apple_dist_vec = np.array([
                (self.apple[0] - head[0]) / GRID_WIDTH,
                (self.apple[1] - head[1]) / GRID_HEIGHT
            ], dtype=np.float32)

        # Vergangene Aktionen
        action_history_vec = np.array(list(self.action_history), dtype=np.float32) / (self.action_space.n -1)
        
        obs = np.concatenate((dir_vec, food_vec, snake_length_vec, fov_vec, apple_dist_vec, action_history_vec))
        return obs.astype(np.float32)


    # Zeigt das Spiel mit pygame an. Nur aktiv, wenn render_mode auf "human" gesetzt ist.
    #
    # Parameter:
    #   game_mode (str): Verändert den Text in der UI im Spielermodus (Je nachdem ob der Spieler oder die KI spielt)
    def render(self, game_mode):
        if self.render_mode != "human":
            return

        # Initialisierung von pygame und den Assets
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH + OFFSET, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.Font(FONT_PATH, 30)
            except FileNotFoundError:
                self.font = pygame.font.SysFont("arial", 30)

            try:
                apple_unscaled = pygame.image.load("assets/apple.png").convert_alpha()
                body_unscaled = pygame.image.load("assets/snake_body.png").convert_alpha()
                head_unscaled = pygame.image.load("assets/snake_head.png").convert_alpha()

                self.apple_img = pygame.transform.scale(apple_unscaled, (CELL_SIZE, CELL_SIZE))
                self.snake_body_img = pygame.transform.scale(body_unscaled, (CELL_SIZE, CELL_SIZE))
                head_original = pygame.transform.scale(head_unscaled, (CELL_SIZE, CELL_SIZE))

                self.snake_head_images = {
                    (0, -1): head_original,
                    (1, 0): pygame.transform.rotate(head_original, -90),
                    (0, 1): pygame.transform.rotate(head_original, 180),
                    (-1, 0): pygame.transform.rotate(head_original, 90)
                }
            except pygame.error as e:
                print(f"Fehler beim Laden der Bilder: {e}")
                self.close()
                return

         # Hintergrund & UI zeichnen
        self.window.fill((53, 53, 53))
        pygame.draw.rect(self.window, (64, 64, 64), pygame.Rect(0, 0, OFFSET, SCREEN_HEIGHT))
        pygame.draw.rect(self.window, (48, 48, 48), pygame.Rect(OFFSET-20, 0, 20, SCREEN_HEIGHT))

        # Schlange zeichnen
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * CELL_SIZE + OFFSET, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if i == 0:
                head_img = self.snake_head_images.get(self.direction)
                if head_img:
                    self.window.blit(head_img, rect)
            else:
                if self.snake_body_img:
                    self.window.blit(self.snake_body_img, rect)

        # Apfel zeichnen
        if self.apple and self.apple_img:
            ax, ay = self.apple
            rect = pygame.Rect(ax * CELL_SIZE + OFFSET, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            self.window.blit(self.apple_img, rect)

        # Hindernisse zeichnen
        for obstacle_blocks in self.obstacles:
            for ox, oy in obstacle_blocks:
                pygame.draw.rect(
                    self.window,
                    OBSTACLE_COLOR,
                    pygame.Rect(ox * CELL_SIZE + OFFSET, oy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    border_radius=0,
                )
        
        # Score anzeigen
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (42, 370))

        # UI Hinweis anzeigen
        if game_mode != "none":
            lines = [
                "Press 'SPACE'",
                " to enable ",
                f" {game_mode}-mode"
            ]
            for i, line in enumerate(lines):
                info_msg = self.font.render(line, True, (255, 255, 255))
                self.window.blit(info_msg, (7, 20 + i * 30))

        # Bei Spielende: Finaler Score anzeigen
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


    # Schließt das Fenster und beendet pygame
    def close(self):
        if self.window:
            pygame.quit()
            self.window = None