import pygame
from stable_baselines3 import DQN
from snake_env import SnakeEnv

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

BUTTON_WIDTH = 400
BUTTON_HEIGHT = 80
BUTTON_COLOR = (84, 173, 62)
HOVER_COLOR = (113, 205, 90)
CLICK_COLOR = (70, 152, 50)
TEXT_COLOR = (255, 255, 255)
FONT_PATH = "assets/VCR_OSD_MONO_1.001.ttf"
BACKGROUND_IMAGE_PATH = "assets/background.png"


def play():
    env = SnakeEnv(render_mode="human")
    model = DQN.load("dqn_snake_model")
    obs, info = env.reset()
    done = False
    current_action = 1
    use_ai = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    use_ai = not use_ai
                if not use_ai:
                    action = None
                    if event.key in (pygame.K_UP, pygame.K_w):
                        action = 0
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        action = 1
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        action = 2
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        action = 3

                    if action is not None:
                        current_action = action

        if use_ai:
            action, _states = model.predict(obs, deterministic=True)
            current_action = int(action)

        obs, reward, done, truncated, info = env.step(current_action)

        env.render(game_mode="User" if use_ai else " AI")

    env.close()



def test_model(model_path="dqn_snake_model"):
    env = SnakeEnv(render_mode="human")
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


def main_menu():
    screen, font, background_image = init_pygame()

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

        play_button_color = CLICK_COLOR if play_button_clicked else (
            HOVER_COLOR if play_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        )
        pygame.draw.rect(screen, play_button_color, play_button_rect, border_radius=10)
        play_text = font.render("Play", True, TEXT_COLOR)
        screen.blit(play_text, play_text.get_rect(center=play_button_rect.center))

        ai_button_color = CLICK_COLOR if ai_button_clicked else (
            HOVER_COLOR if ai_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        )
        pygame.draw.rect(screen, ai_button_color, ai_button_rect, border_radius=10)
        ai_text = font.render("Run AI Model", True, TEXT_COLOR)
        screen.blit(ai_text, ai_text.get_rect(center=ai_button_rect.center))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main_menu()
