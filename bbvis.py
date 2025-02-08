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
boxes = [(358.0, 438.0, 105.0, 42.0),
 (207.0, 391.0, 55.0, 88.99999999999999),
 (150.25000005960464, 275.0014267228544,
  45.24961083382368, 42.01548025670674),
 (233.0, 215.0, 42.00097668170929, 41.999999999999545),
 (313.0, 157.0, 77.00000000023283, 76.0)]



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