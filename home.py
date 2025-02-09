import pygame
import math
import sys
import subprocess
import csv
import os

pygame.init()

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (60, 60, 60)
TEXT_COLOR = (255, 255, 255)
FONT_SIZE = 60  # Reduced from 80
BUTTON_FONT_SIZE = 30  # Reduced from 36
NUM_RAYS = 360
WOOD_COLOR = (255,77,4)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projector Projection Machine")
font = pygame.font.Font(None, FONT_SIZE)
button_font = pygame.font.Font(None, BUTTON_FONT_SIZE)
leader_font = pygame.font.SysFont("Arial", 24, bold=True)  # Smaller and bold
small_font = pygame.font.Font(None, 30)

recording = False


def draw_text(text, font, color, surface, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    surface.blit(text_surface, text_rect)


def get_top_scores():
    scores = []
    if os.path.exists("scores.csv"):
        with open("scores.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        name = row[0]
                        score = int(row[1])
                        scores.append((name, score))
                    except:
                        pass
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]


class Ray:
    def __init__(self, x1, y1, dirX, dirY):
        self.x1 = x1
        self.y1 = y1
        self.dirX = dirX
        self.dirY = dirY

    def collide(self, wall):
        wx1, wy1, wx2, wy2 = wall.get_coordinates()
        rx3, ry3 = self.x1, self.y1
        rx4, ry4 = self.x1 + self.dirX, self.y1 + self.dirY

        n = (wx1 - rx3) * (ry3 - ry4) - (wy1 - ry3) * (rx3 - rx4)
        d = (wx1 - wx2) * (ry3 - ry4) - (wy1 - wy2) * (rx3 - rx4)

        if d == 0:
            return False

        t = n / d
        u = ((wx2 - wx1) * (wy1 - ry3) - (wy2 - wy1) * (wx1 - rx3)) / d

        if 0 < t < 1 and u > 0:
            px = wx1 + t * (wx2 - wx1)
            py = wy1 + t * (wy2 - wy1)
            return (px, py)
        return False


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def get_coordinates(self):
        return self.x1, self.y1, self.x2, self.y2

    def show(self, surface):
        pygame.draw.line(surface, WOOD_COLOR, (self.x1, self.y1), (self.x2, self.y2), 5)


class Light:
    def __init__(self, x1, y1, num_rays):
        self.x1, self.y1 = x1, y1
        self.rays = [
            Ray(self.x1, self.y1, math.cos(math.radians(i)), math.sin(math.radians(i)))
            for i in range(0, 360, int(360 / num_rays))
        ]

    def show(self, surface, walls):
        for ray in self.rays:
            ray.x1, ray.y1 = self.x1, self.y1
            closest = float("inf")
            closest_point = None
            for wall in walls:
                intersection = ray.collide(wall)
                if intersection:
                    distance = math.sqrt(
                        (ray.x1 - intersection[0]) ** 2 + (ray.y1 - intersection[1]) ** 2
                    )
                    if distance < closest:
                        closest = distance
                        closest_point = intersection
            if closest_point:
                pygame.draw.line(surface, (255,77,4), (ray.x1, ray.y1), closest_point)


class Button:
    def __init__(self, text, x, y, width, height, color=BUTTON_COLOR):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = BUTTON_HOVER_COLOR

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        current_color = self.color if not self.rect.collidepoint(mouse_pos) else self.hover_color
        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)
        text_surface = button_font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def click(self):
        mouse_pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(mouse_pos)


# Revised projector-shaped walls for side view
walls = [
    # Main body (rectangle)
    Wall(200, 200, 200, 400),
    Wall(200, 400, 600, 400),
    Wall(600, 400, 600, 200),
    Wall(600, 200, 200, 200),
    # Lens (front)
    Wall(600, 250, 650, 250),
    Wall(650, 250, 650, 350),
    Wall(650, 350, 600, 350),
    # Base
    Wall(250, 400, 250, 450),
    Wall(250, 450, 550, 450),
    Wall(550, 450, 550, 400),
]

light = Light(500, 500, NUM_RAYS)


def draw_leader_board(surface):
    top_scores = get_top_scores()
    panel_width, panel_height = 300, 180
    panel_x = 250
    panel_y = 210

    # Create semi-transparent background
    panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    pygame.draw.rect(panel_surface, (0, 0, 0, 200), panel_surface.get_rect(), border_radius=12)
    pygame.draw.rect(panel_surface, (255, 255, 255, 100), panel_surface.get_rect(), 2, border_radius=12)
    surface.blit(panel_surface, (panel_x, panel_y))

    # Draw content
    draw_text("HIGH SCORES", leader_font, TEXT_COLOR, surface, panel_x + panel_width // 2, panel_y + 25)
    pygame.draw.line(surface, (255, 255, 255), (panel_x + 20, panel_y + 50), (panel_x + panel_width - 20, panel_y + 50),
                     1)

    y_offset = panel_y + 70
    for idx, (name, score) in enumerate(top_scores):
        entry = f"{idx + 1}. {name[:10]:<10} {score:>5}"
        text_surface = leader_font.render(entry, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(panel_x + panel_width // 2, y_offset))
        surface.blit(text_surface, text_rect)
        y_offset += 35


start_button = Button("Start", WIDTH // 2 - 75, HEIGHT - 120, 150, 50)

running = True
while running:
    screen.fill(BACKGROUND_COLOR)

    draw_text("Projector Projection", font, TEXT_COLOR, screen, WIDTH // 2, 50)

    for wall in walls:
        wall.show(screen)

    light.x1, light.y1 = pygame.mouse.get_pos()
    light.show(screen, walls)

    start_button.draw(screen)
    draw_leader_board(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.click():
                pygame.quit()
                subprocess.run(["python", "main.py"])
                sys.exit()

    pygame.display.flip()

pygame.quit()
sys.exit()