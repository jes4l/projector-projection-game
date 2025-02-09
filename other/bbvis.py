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
boxes =  [(358.0, 439.0, 106.0, 41.0), (207.00039025543956, 392.0, 54.99960974456044, 88.0), (150.3103411656808, 275.0000022865832, 44.69043875271128, 42.99999771341682), (233.99999999998835, 215.5336958424663, 41.00000000001163, 41.466304157533735), (313.0, 157.0, 77.49551637671563, 76.0)]



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