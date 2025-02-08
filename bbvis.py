import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Bounding Box Visualization")

# Bounding box data (x, y, width, height)
boxes = [
    (357.3016910363732, 438.0, 109.6983089636268, 42.0),
    (210.59374987147748, 391.0, 56.40625011898915, 89.0),
    (153.7499673366547, 275.0, 44.25003266334622, 42.00000000005778),
    (236.0, 215.0, 42.0, 41.876929552333294),
    (315.0197758823638, 152.04170963404925, 81.26941717064146, 77.95829036595076)
]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill background with white
    screen.fill((255, 255, 255))

    # Draw bounding boxes
    for box in boxes:
        # Convert coordinates to integers
        x, y, w, h = [int(round(coord)) for coord in box]
        pygame.draw.rect(screen, (255, 0, 0), (x, y, w, h), 2)

    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()