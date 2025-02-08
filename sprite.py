import cv2
import numpy as np


class Sprite:
    def __init__(self, image_path, initial_position=(100, 100)):
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Could not load sprite image from {image_path}")

        self.position = list(initial_position)
        self.speed = 5
        self.direction = [0, 0]
        self.moving = False
        self.active = True

    def start_movement(self):
        self.moving = True

    def stop_movement(self):
        self.moving = False

    def move_left(self):
        self.direction = [-1, 0]

    def move_right(self):
        self.direction = [1, 0]

    def move_up(self):
        self.direction = [0, -1]

    def move_down(self):
        self.direction = [0, 1]

    def update(self, frame_width, frame_height):
        if self.moving:
            new_x = self.position[0] + self.direction[0] * self.speed
            new_y = self.position[1] + self.direction[1] * self.speed

            self.position[0] = np.clip(new_x, 0, frame_width - self.image.shape[1])
            self.position[1] = np.clip(new_y, 0, frame_height - self.image.shape[0])

    def draw(self, frame):
        if not self.active:
            return

        y1 = int(self.position[1])
        y2 = y1 + self.image.shape[0]
        x1 = int(self.position[0])
        x2 = x1 + self.image.shape[1]

        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            return

        alpha_channel = self.image[:, :, 3] / 255.0
        inverse_alpha = 1.0 - alpha_channel

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (
                    alpha_channel * self.image[:, :, c] +
                    inverse_alpha * frame[y1:y2, x1:x2, c]
            )