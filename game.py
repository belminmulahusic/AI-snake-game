import pygame

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


def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")
    
    try:
        font = pygame.font.Font(FONT_PATH, 50)
    except FileNotFoundError:
        font = pygame.font.SysFont("arial", 50)

    background_image = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))


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
                elif ai_button_rect.collidepoint(event.pos):
                    ai_button_clicked = True

            if event.type == pygame.MOUSEBUTTONUP:
                play_button_clicked = False
                ai_button_clicked = False

        if background_image:
            screen.blit(background_image, (0, 0))
        else:
            screen.fill(0, 0, 0)

        play_button_current_color = BUTTON_COLOR
        if play_button_clicked:
            play_button_current_color = CLICK_COLOR
        elif play_button_rect.collidepoint(mouse_pos):
            play_button_current_color = HOVER_COLOR
        pygame.draw.rect(screen, play_button_current_color, play_button_rect, border_radius=10)
        play_text = font.render("Play", True, TEXT_COLOR)
        play_text_rect = play_text.get_rect(center=play_button_rect.center)
        screen.blit(play_text, play_text_rect)

        ai_button_current_color = BUTTON_COLOR
        if ai_button_clicked:
            ai_button_current_color = CLICK_COLOR
        elif ai_button_rect.collidepoint(mouse_pos):
            ai_button_current_color = HOVER_COLOR
        pygame.draw.rect(screen, ai_button_current_color, ai_button_rect, border_radius=10)
        ai_text = font.render("Run AI Model", True, TEXT_COLOR)
        ai_text_rect = ai_text.get_rect(center=ai_button_rect.center)
        screen.blit(ai_text, ai_text_rect)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main_menu()