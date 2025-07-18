##################################################################
#                                                                #
#   (Veraltet) Dies ist die erste Version des Spiels.            #
#   Die main.py wurde nur Aufgrund von Aufgabe 1 erstellt        #
#                                                                #
##################################################################

import pygame
import sys

pygame.init()
pygame.display.set_caption("Snake Game")

CELL_SIZE = 50
GRID_WIDTH, GRID_HEIGHT = 40, 20
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

BLACK = (30, 30, 30)
GREEN = (0, 255, 100)
RED = (255, 80, 80)
WHITE = (240, 240, 240)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('consolas', 24)

snake = [(5, 5), (4, 5), (3, 5)]
direction = (1, 0)
score = 0

# Fixe Apfel Positionen (Aufgabe 1)
apples_queue = [(33, 5), (15, 10), (7, 3), (20, 15), (5, 12)]
current_apple = apples_queue.pop(0)



# Zeichnet eine einzelne Zelle an der Position mit der Farbe
# Parameter:
#   pos (tuple): (x, y) Koordinate der Zelle
#   color (tuple): RGB-Farbwert
def draw_cell(pos, color):
    x, y = pos
    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, rect, border_radius=12)


# Beendet das Spiel und zeigt eine Nachricht an
# Parameter:
#   message (str): Text für die NAchricht
def game_end(message):
    screen.fill(BLACK)
    msg = font.render(message, True, WHITE)
    screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, SCREEN_HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(2500)
    pygame.quit()
    sys.exit()


# Hauptspiel-Schleife
running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)
    direction_changed = False

    # Steuerung
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and not direction_changed:
            key = event.key
            if (key == pygame.K_UP or key == pygame.K_w) and direction != (0, 1):
                direction = (0, -1)
                direction_changed = True
            elif (key == pygame.K_DOWN or key == pygame.K_s) and direction != (0, -1):
                direction = (0, 1)
                direction_changed = True
            elif (key == pygame.K_LEFT or key == pygame.K_a) and direction != (1, 0):
                direction = (-1, 0)
                direction_changed = True
            elif (key == pygame.K_RIGHT or key == pygame.K_d) and direction != (-1, 0):
                direction = (1, 0)
                direction_changed = True

    new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

    # Kollision wird geprüft
    if (
        new_head in snake
        or new_head[0] < 0 or new_head[0] >= GRID_WIDTH
        or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT
    ):
        game_end(f"Game Over! Punkte: {score}")

    if new_head == current_apple:
        score += 1
        if apples_queue:
            current_apple = apples_queue.pop(0)
        else:
            current_apple = None
    else:
        snake.pop()

    snake.insert(0, new_head)

    # Gewinnbedingung
    if current_apple is None:
        game_end(f"Gewonnen! Alle Äpfel gegessen. Punkte: {score}")

    # Schlange und Apfel zeichnen
    for segment in snake:
        draw_cell(segment, GREEN)

    if current_apple:
        draw_cell(current_apple, RED)

    # Punktestand anzeigen
    text = font.render(f"Punkte: {score}", True, WHITE)
    screen.blit(text, (10, 10))

    pygame.display.flip()
