##################################################################
#                                                                #
#   Die game.py ist die zentrale Komponente unserer Anwendung.   #
#   Hier kann man Snake spielen und sich auch den trainierten    #
#   Agenten ankucken.                                            #
#                                                                #
##################################################################


import pygame
from stable_baselines3 import DQN
from snake_env import SnakeEnv

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# Button-Styling
BUTTON_WIDTH = 400
BUTTON_HEIGHT = 80
BUTTON_COLOR = (84, 173, 62)
HOVER_COLOR = (113, 205, 90)
CLICK_COLOR = (70, 152, 50)
TEXT_COLOR = (255, 255, 255)

# Assets
FONT_PATH = "assets/VCR_OSD_MONO_1.001.ttf"
BACKGROUND_IMAGE_PATH = "assets/background.png"


# Startet das Spiel im Spielermodus mit Tastatursteuerung
def play():
    # Lädt ein Dummy-Modell, falls der Spieler auf den AI-Modus umschalten möchte
    env = SnakeEnv(render_mode="human")
    model = DQN.load("dqn_snake_model")
    obs, info = env.reset()
    done = False
    
    use_ai = False

    while not done:
        action_to_take = 0
        direction_changed_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    use_ai = not use_ai
                
                # Manuelle Steuerung nur
                if not use_ai and not direction_changed_this_frame:
                    current_direction = env.direction
                    
                    if event.key == pygame.K_w:
                        if current_direction != (0, 1):
                            env.direction = (0, -1)
                            direction_changed_this_frame = True
                    elif event.key == pygame.K_s:
                        if current_direction != (0, -1):
                            env.direction = (0, 1)
                            direction_changed_this_frame = True
                    elif event.key == pygame.K_a:
                        if current_direction != (1, 0):
                            env.direction = (-1, 0)
                            direction_changed_this_frame = True
                    elif event.key == pygame.K_d:
                        if current_direction != (-1, 0):
                            env.direction = (1, 0)
                            direction_changed_this_frame = True
        
        # Wenn AI aktiviert ist, entscheidet das Modell
        if use_ai:
            action, _states = model.predict(obs, deterministic=True)
            action_to_take = int(action)

        # Führt den Schritt aus
        obs, reward, done, truncated, info = env.step(action_to_take)
        env.render(game_mode="User" if not use_ai else " AI")

    env.close()
    return "menu"


# Startet das Spiel im automatischen KI Modus
def test_model(model_path="dqn_snake_model"):
    env = SnakeEnv(render_mode="human", render_fps=30)
    model = DQN.load(model_path)

    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, done, truncated, info = env.step(action)
        env.render(game_mode="none")

    print(f"Spiel beendet. End-Score: {env.score}")
    env.close()


# Initialisiert Pygame, Schriftart und Hintergrund
def init_pygame():
    if not pygame.get_init():
        pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")

    try:
        font = pygame.font.Font(FONT_PATH, 50)
    except FileNotFoundError:
        font = pygame.font.SysFont("arial", 50)

    background_image = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    return screen, font, background_image


# Zeigt das Hauptmenü an: Hier kann zwischen zwei Modi gewählt werden
def main_menu():
    screen, font, background_image = init_pygame()

    # Buttons erstellen
    play_button_rect = pygame.Rect(
        (SCREEN_WIDTH - BUTTON_WIDTH) // 2,
        (SCREEN_HEIGHT // 3) + 30,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
    )
    ai_button_rect = pygame.Rect(
        (SCREEN_WIDTH - BUTTON_WIDTH) // 2,
        (SCREEN_HEIGHT // 3 + BUTTON_HEIGHT + 20) + 30,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
    )

    play_button_clicked = False
    ai_button_clicked = False
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_button_rect.collidepoint(event.pos):
                    play_button_clicked = True
                    game_result = play()

                    if not pygame.display.get_init():
                        screen, font, background_image = init_pygame()

                    if game_result == "quit":
                        running = False

                elif ai_button_rect.collidepoint(event.pos):
                    ai_button_clicked = True
                    test_model()

                    if not pygame.display.get_init():
                        screen, font, background_image = init_pygame()

            if event.type == pygame.MOUSEBUTTONUP:
                play_button_clicked = False
                ai_button_clicked = False

        if background_image:
            screen.blit(background_image, (0, 0))
        else:
            screen.fill((0, 0, 0))

        # Play-Button zeichnen
        play_button_color = CLICK_COLOR if play_button_clicked else (
            HOVER_COLOR if play_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        )
        pygame.draw.rect(screen, play_button_color, play_button_rect, border_radius=10)
        play_text = font.render("Play", True, TEXT_COLOR)
        screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))

        # AI-Button zeichnen
        ai_button_color = CLICK_COLOR if ai_button_clicked else (
            HOVER_COLOR if ai_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        )
        pygame.draw.rect(screen, ai_button_color, ai_button_rect, border_radius=10)
        ai_text = font.render("Run AI Model", True, TEXT_COLOR)
        screen.blit(ai_text, ai_text.get_rect(center=ai_button_rect.center))

        pygame.display.flip()

    pygame.quit()


# Startet das Hauptmenü beim Ausführen des Scripts
if __name__ == "__main__":
    main_menu()