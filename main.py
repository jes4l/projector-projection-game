import sys
import cv2
import numpy as np
import random
import serial
import threading
import csv
import pygame
from PyQt5.QtCore import Qt, QTimer, QEvent, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget, QHBoxLayout
)

pygame.mixer.init()

class CameraWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)
        self.edgesMode = False
        self.prevBoxes = []
        self.autoAdjustFrames = 100
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False
        self.lastFrame = None
        self.capturedCoords = []
        self.bboxAcc = []
        self.bboxCount = 0
        self.freezeBoxes = False
        self.spawned = False
        self.sprite = None
        self.x_offset = 0
        self.y_offset = 0
        self.ground_y = 0
        self.move_step = 5
        self.is_jumping = False
        self.jump_velocity = 0
        self.gravity = 1
        self.key_state = {"a": False, "d": False, " ": False}
        self.savedSpritePos = None
        self.score = 0
        self.targetSprite = None
        self.targetX = 0
        self.targetY = 0
        self.target_sprite_width = 0
        self.target_sprite_height = 0

        heart_sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0003.png"
        self.heartSprite = cv2.imread(heart_sprite_path, cv2.IMREAD_UNCHANGED)
        if self.heartSprite is None:
            print(f"Error: Could not load heart sprite from {heart_sprite_path}")

        self.jump_sound = pygame.mixer.Sound(r"C:\Users\LLR User\Desktop\your-childhood-game\bounding-box.mp3")
        self.coin_sound = pygame.mixer.Sound(r"C:\Users\LLR User\Desktop\your-childhood-game\coin.mp3")
        self.scream_sound = pygame.mixer.Sound(r"C:\Users\LLR User\Desktop\your-childhood-game\scream.mp3")
        pygame.mixer.music.load(r"C:\Users\LLR User\Desktop\your-childhood-game\background-music.mp3")
        pygame.mixer.music.play(-1)

        self.ambientVisible = False
        self.ambient_particles = None
        self.coin_particles = []
        self.heart_particles = []
        self.platform_glow_triggered = []
        self.platform_glow_color = []
        self.platform_particles = []
        self.justLanded = False

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("background-color: black;")
        layout.addWidget(self.videoLabel)

        self.controlBarOverlay = QWidget(self.videoLabel)
        self.controlBarOverlay.setStyleSheet("background-color: rgba(50, 50, 50, 230); border: none;")
        self.controlBarOverlay.setVisible(False)

        controlLayout = QHBoxLayout(self.controlBarOverlay)
        controlLayout.setContentsMargins(10, 5, 10, 5)
        controlLayout.setSpacing(10)
        self.thresholdButton = QPushButton("Canny Thresholds")
        self.edgesViewButton = QPushButton("Edge Detection")
        self.spawnButton = QPushButton("Spawn")
        for btn in (self.thresholdButton, self.edgesViewButton, self.spawnButton):
            btn.setFixedSize(110, 30)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setStyleSheet("font-size:10pt; color: white; background-color: transparent; border: none;")
        controlLayout.addWidget(self.thresholdButton)
        controlLayout.addWidget(self.edgesViewButton)
        controlLayout.addWidget(self.spawnButton)
        controlLayout.addStretch(1)

        self.sliderOverlay = QWidget(self.videoLabel)
        self.sliderOverlay.setStyleSheet("background-color: rgba(40, 40, 40, 230); border: none;")
        self.sliderOverlay.setVisible(False)
        self.sliderOverlay.resize(300, 200)
        sliderLayout = QVBoxLayout(self.sliderOverlay)
        sliderLayout.setContentsMargins(10, 10, 10, 10)
        sliderLayout.setSpacing(10)
        self.lowerLabel = QLabel("Canny Lower Threshold: 50")
        self.lowerLabel.setStyleSheet("color: white;")
        self.lowerSlider = QSlider(Qt.Horizontal)
        self.lowerSlider.setRange(0, 500)
        self.lowerSlider.setValue(50)
        self.upperLabel = QLabel("Canny Upper Threshold: 150")
        self.upperLabel.setStyleSheet("color: white;")
        self.upperSlider = QSlider(Qt.Horizontal)
        self.upperSlider.setRange(0, 500)
        self.upperSlider.setValue(150)
        self.lowerSlider.valueChanged.connect(self.updateLowerLabel)
        self.upperSlider.valueChanged.connect(self.updateUpperLabel)
        slider_style = """
            QSlider::groove:horizontal {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff0000, stop:1 #00ff00);
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #eeeeee;
                border: none;
                width: 16px;
                margin: -3px 0;
                border-radius: 8px;
            }
        """
        self.lowerSlider.setStyleSheet(slider_style)
        self.upperSlider.setStyleSheet(slider_style)
        sliderLayout.addWidget(self.lowerLabel)
        sliderLayout.addWidget(self.lowerSlider)
        sliderLayout.addWidget(self.upperLabel)
        sliderLayout.addWidget(self.upperSlider)
        self.autoCalibrateButton = QPushButton("Auto Calibrate")
        self.autoCalibrateButton.setStyleSheet("font-size:10pt; color: white; background-color: #444; border: none; padding: 5px;")
        self.autoCalibrateButton.setAutoDefault(False)
        self.autoCalibrateButton.setDefault(False)
        self.autoCalibrateButton.clicked.connect(self.manualCalibrate)
        sliderLayout.addWidget(self.autoCalibrateButton)

        self.thresholdButton.clicked.connect(self.toggleSliderOverlay)
        self.edgesViewButton.clicked.connect(self.toggleEdgesMode)
        self.spawnButton.clicked.connect(self.spawnPlayer)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open camera")
            sys.exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

        self.installEventFilter(self)
 
        enemy_sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0004.png"
        self.enemySprite = cv2.imread(enemy_sprite_path, cv2.IMREAD_UNCHANGED)
        if self.enemySprite is None:
            print(f"Error: Could not load enemy sprite from {enemy_sprite_path}")
        else:
            self.enemySprite_height, self.enemySprite_width = self.enemySprite.shape[:2]
        self.enemies = []
        self.enemy_speed = 2
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for i in range(3):
            enemy = {}
            enemy['x'] = random.randint(0, max(0, frame_width - (self.enemySprite_width if self.enemySprite is not None else 50)))
            enemy['y'] = frame_height - (self.enemySprite_height if self.enemySprite is not None else 50)
            enemy['direction'] = random.choice([-1, 1])
            enemy['speed'] = self.enemy_speed
            self.enemies.append(enemy)
        self.lives = 3
        self.invulnerable = False
        self.red_tint = False

    def spawnPlayer(self):
        if not self.spawned:
            self.spawned = True
            self.score = 0
            self.reset_gameplay()
            self.controlBarOverlay.setVisible(False)
            self.sliderOverlay.setVisible(False)
            self.ambientVisible = True
            self.ambient_particles = None
            target_sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0002.png"
            self.targetSprite = cv2.imread(target_sprite_path, cv2.IMREAD_UNCHANGED)
            if self.targetSprite is None:
                print(f"Error: Could not load target sprite from {target_sprite_path}")
            else:
                self.target_sprite_height, self.target_sprite_width = self.targetSprite.shape[:2]
                self.generateTarget()
            print("Player spawned!")

    def reset_gameplay(self):
        sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0001.png"
        self.sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if self.sprite is None:
            print(f"Error: Could not load sprite from {sprite_path}")
            return
        self.sprite_height, self.sprite_width = self.sprite.shape[:2]
        if self.savedSpritePos is not None:
            self.x_offset, self.y_offset = self.savedSpritePos
        else:
            self.x_offset = 10
            self.y_offset = 0
        self.is_jumping = False
        self.jump_velocity = 0
        self.gravity = 1
        self.move_step = 5
        self.key_state = {"a": False, "d": False, " ": False}
        self.justLanded = False

    def hideSprite(self):
        if self.spawned:
            self.savedSpritePos = (self.x_offset, self.y_offset)
            self.spawned = False
            self.sprite = None
            print("Sprite hidden for calibration, saved position:", self.savedSpritePos)
            self.ambientVisible = False

    def respawnSprite(self, pos):
        self.savedSpritePos = None
        self.spawned = True
        self.reset_gameplay()
        self.x_offset, self.y_offset = pos
        print("Sprite respawned at", pos)

    def generateTarget(self):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_attempts = 10
        valid = False
        attempt = 0
        new_x = 0
        new_y = 0
        if self.heartSprite is not None:
            heart_h, heart_w = self.heartSprite.shape[:2]
            hearts_x = 10
            hearts_y = 10
            hearts_width = heart_w * 3 + 2 * 5
            hearts_bottom = hearts_y + heart_h
            min_target_y = hearts_bottom + 30
        else:
            min_target_y = 0
        while attempt < max_attempts and not valid:
            new_x = random.randint(0, max(0, frame_width - self.target_sprite_width))
            max_target_y = max(0, (frame_height // 4) - self.target_sprite_height)
            new_y = random.randint(min_target_y, max_target_y) if min_target_y < max_target_y else random.randint(0, max_target_y)
            valid = True
            for (bx, by, bw, bh) in self.capturedCoords:
                if self.rectangles_intersect(new_x, new_y, self.target_sprite_width, self.target_sprite_height,
                                             bx, by, bw, bh):
                    valid = False
                    break
            if valid and self.heartSprite is not None:
                if self.rectangles_intersect(new_x, new_y, self.target_sprite_width, self.target_sprite_height,
                                             hearts_x, hearts_y, hearts_width, heart_h):
                    valid = False
            attempt += 1
        self.targetX = new_x
        self.targetY = new_y

    def rectangles_intersect(self, x1, y1, w1, h1, x2, y2, w2, h2):
        return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)

    def updateLowerLabel(self, value):
        self.lowerLabel.setText(f"Canny Lower Threshold: {value}")
        if self.spawned:
            self.hideSprite()
        if self.freezeBoxes:
            self.reset_bbox_accumulation()

    def updateUpperLabel(self, value):
        self.upperLabel.setText(f"Canny Upper Threshold: {value}")
        if self.spawned:
            self.hideSprite()
        if self.freezeBoxes:
            self.reset_bbox_accumulation()

    def reset_bbox_accumulation(self):
        self.bboxAcc = []
        self.bboxCount = 0
        self.freezeBoxes = False
        self.platform_glow_triggered = []
        self.platform_glow_color = []
        self.platform_particles = []
        self.coin_particles = []
        self.heart_particles = []
        self.ambientVisible = False

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_H:
                visible = not self.controlBarOverlay.isVisible()
                self.controlBarOverlay.setVisible(visible)
                if not visible:
                    self.sliderOverlay.setVisible(False)
                if visible:
                    self.controlBarOverlay.raise_()
                    self.updateControlBarPosition()
                return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        if self.spawned:
            if event.key() == Qt.Key_A:
                self.key_state["a"] = True
            elif event.key() == Qt.Key_D:
                self.key_state["d"] = True
            elif event.key() == Qt.Key_Space:
                self.key_state[" "] = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.spawned:
            if event.key() == Qt.Key_A:
                self.key_state["a"] = False
            elif event.key() == Qt.Key_D:
                self.key_state["d"] = False
            elif event.key() == Qt.Key_Space:
                self.key_state[" "] = False
        super().keyReleaseEvent(event)

    def toggleSliderOverlay(self):
        visible = not self.sliderOverlay.isVisible()
        self.sliderOverlay.setVisible(visible)
        if visible:
            self.sliderOverlay.raise_()
            self.updateSliderOverlayPosition()

    def toggleEdgesMode(self):
        self.edgesMode = not self.edgesMode
        self.edgesViewButton.setText("Overlay View" if self.edgesMode else "Edge Detection")

    def manualCalibrate(self):
        if self.spawned:
            self.hideSprite()
        self.autoCalibrateButton.setText("Autocalibrating")
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False
        print("Manual calibration initiated.")
        self.ambientVisible = False
        self.reset_bbox_accumulation()

    def onCalibrationComplete(self):
        self.autoCalibrateButton.setText("Auto Calibrate")
        if self.savedSpritePos is not None:
            self.respawnSprite(self.savedSpritePos)
        self.reset_bbox_accumulation()
    def reset_invulnerability(self):
        self.invulnerable = False

    def reset_red_tint(self):
        self.red_tint = False

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.ambientVisible:
            if self.ambient_particles is None:
                self.ambient_particles = []
                for _ in range(50):
                    particle = {
                        'x': random.uniform(0, frame.shape[1]),
                        'y': random.uniform(0, frame.shape[0]),
                        'vy': random.uniform(0.2, 0.5),
                        'radius': random.randint(2, 4),
                        'color': random.choice([(230,230,255), (240,240,255), (255,250,250), (245,245,255)])
                    }
                    self.ambient_particles.append(particle)
            for particle in self.ambient_particles:
                particle['y'] += particle['vy']
                if particle['y'] > frame.shape[0]:
                    particle['y'] = 0
                    particle['x'] = random.uniform(0, frame.shape[1])
                cv2.circle(frame, (int(particle['x']), int(particle['y'])), particle['radius'], particle['color'], thickness=-1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)

        if not self.autoThresholdCalibrated:
            median_val = np.median(blurred)
            scale_factor = 0.26
            calibrated_val = median_val * scale_factor
            self.medianSum += calibrated_val
            self.medianCount += 1
            if self.medianCount >= self.autoAdjustFrames:
                avg_calibrated = self.medianSum / self.medianCount
                sigma = 0.36
                lower_auto = int(max(0, (1.0 - sigma) * avg_calibrated))
                upper_auto = int(min(255, (1.0 + sigma) * avg_calibrated))
                self.lowerSlider.setValue(lower_auto)
                self.upperSlider.setValue(upper_auto)
                self.autoThresholdCalibrated = True
                self.autoCalibrateButton.setText("Autocalibrated")
                print(f"Auto-calibrated thresholds: lower={lower_auto}, upper={upper_auto}")
                QTimer.singleShot(2000, self.onCalibrationComplete)

        lower_val = self.lowerSlider.value()
        upper_val = self.upperSlider.value()
        edges = cv2.Canny(blurred, lower_val, upper_val)
        inverted = cv2.bitwise_not(blurred)
        inverted_edges = cv2.Canny(inverted, lower_val, upper_val)
        combined_edges = cv2.bitwise_or(edges, inverted_edges)
        kernel = np.ones((7, 7), np.uint8)
        closed_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilate_kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(closed_edges, dilate_kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        newBoxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > frame_area * 0.9:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            newBoxes.append((x, y, w, h))
        smoothedBoxes = self.smoothBoxes(newBoxes, self.prevBoxes, alpha=0.5, distance_threshold=20)
        self.prevBoxes = smoothedBoxes

        if not self.spawned:
            if not self.freezeBoxes:
                if smoothedBoxes:
                    self.bboxAcc.append(smoothedBoxes)
                    self.bboxCount += 1
                if self.bboxCount >= 10:
                    averaged = self.average_bounding_boxes(self.bboxAcc)
                    self.capturedCoords = averaged
                    self.freezeBoxes = True
                    print("Averaged bounding boxes over 10 frames:", self.capturedCoords)
                    self.platform_glow_triggered = [False] * len(self.capturedCoords)
                    self.platform_glow_color = [None] * len(self.capturedCoords)
                    self.platform_particles = [[] for _ in range(len(self.capturedCoords))]
            else:
                smoothedBoxes = self.capturedCoords
        else:
            smoothedBoxes = []

        if not self.spawned:
            for box in smoothedBoxes:
                x, y, w, h = box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), thickness=2)

        if self.spawned:
            ground_y = frame.shape[0] - self.sprite_height
            self.ground_y = ground_y

            old_x = self.x_offset
            if self.key_state.get("a", False):
                self.x_offset -= self.move_step
            if self.key_state.get("d", False):
                self.x_offset += self.move_step
            self.x_offset = max(0, min(self.x_offset, frame.shape[1] - self.sprite_width))
            for (plat_x, plat_y, plat_w, plat_h) in self.capturedCoords:
                if self.rectangles_intersect(self.x_offset, self.y_offset, self.sprite_width, self.sprite_height,
                                             plat_x, plat_y, plat_w, plat_h):
                    if abs((self.y_offset + self.sprite_height) - plat_y) >= 5:
                        self.x_offset = old_x
                        break

            if self.y_offset == 0 and not self.is_jumping:
                self.y_offset = ground_y

            if not self.is_jumping:
                supported = False
                tolerance = 5
                for (plat_x, plat_y, plat_w, plat_h) in self.capturedCoords:
                    if self.x_offset + self.sprite_width > plat_x and self.x_offset < plat_x + plat_w:
                        if abs((self.y_offset + self.sprite_height) - plat_y) < tolerance:
                            supported = True
                            break
                if abs(self.y_offset - ground_y) < tolerance:
                    supported = True
                if not supported:
                    self.is_jumping = True
                    self.jump_velocity = 0

            if self.key_state.get(" ", False) and not self.is_jumping:
                self.is_jumping = True
                self.jump_velocity = -15
                self.key_state[" "] = False
                self.justLanded = False

            if self.is_jumping:
                prev_y_offset = self.y_offset
                self.y_offset += self.jump_velocity
                self.jump_velocity += self.gravity
                if self.jump_velocity > 0:
                    landed = False
                    for (plat_x, plat_y, plat_w, plat_h) in self.capturedCoords:
                        if self.x_offset + self.sprite_width > plat_x and self.x_offset < plat_x + plat_w:
                            if (prev_y_offset + self.sprite_height <= plat_y) and (self.y_offset + self.sprite_height >= plat_y):
                                self.y_offset = int(plat_y - self.sprite_height)
                                if not self.justLanded:
                                    pygame.mixer.Sound.play(self.jump_sound)
                                    self.justLanded = True
                                self.is_jumping = False
                                self.jump_velocity = 0
                                landed = True
                                break
                    if not landed and self.y_offset >= ground_y:
                        self.y_offset = ground_y
                        if self.is_jumping and not self.justLanded:
                            pygame.mixer.Sound.play(self.jump_sound)
                            self.justLanded = True
                        self.is_jumping = False
                        self.jump_velocity = 0
                if self.y_offset < 0:
                    self.y_offset = 0
                    if self.jump_velocity < 0:
                        self.jump_velocity = 0

            if self.sprite is not None:
                if self.sprite.shape[2] == 4:
                    sprite_bgr = self.sprite[:, :, :3]
                    if self.red_tint:
                        tinted_sprite = np.zeros_like(sprite_bgr)
                        tinted_sprite[:, :] = (0, 0, 255)
                        sprite_bgr = tinted_sprite
                    alpha_channel = self.sprite[:, :, 3] / 255.0
                    alpha = np.dstack([alpha_channel] * 3)
                    roi = frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width]
                    blended = (alpha * sprite_bgr.astype(float) + (1 - alpha) * roi.astype(float)).astype(np.uint8)
                    frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width] = blended
                else:
                    if self.red_tint:
                        frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width] = np.full((self.sprite_height, self.sprite_width, 3), (0, 0, 255), dtype=np.uint8)
                    else:
                        frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width] = self.sprite

            if self.targetSprite is not None:
                tx, ty = self.targetX, self.targetY
                if self.targetSprite.shape[2] == 4:
                    sprite2_bgr = self.targetSprite[:, :, :3]
                    alpha_channel = self.targetSprite[:, :, 3] / 255.0
                    alpha2 = np.dstack([alpha_channel] * 3)
                    roi2 = frame[ty:ty+self.target_sprite_height, tx:tx+self.target_sprite_width]
                    blended2 = (alpha2 * sprite2_bgr.astype(float) + (1 - alpha2) * roi2.astype(float)).astype(np.uint8)
                    frame[ty:ty+self.target_sprite_height, tx:tx+self.target_sprite_width] = blended2
                else:
                    frame[ty:ty+self.target_sprite_height, tx:tx+self.target_sprite_width] = self.targetSprite

                if (self.x_offset < self.targetX + self.target_sprite_width and
                    self.x_offset + self.sprite_width > self.targetX and
                    self.y_offset < self.targetY + self.target_sprite_height and
                    self.y_offset + self.sprite_height > self.targetY):
                    self.score += 1
                    print("Score increased to", self.score)
                    pygame.mixer.Sound.play(self.coin_sound)
                    for _ in range(10):
                        particle = {
                            'x': self.targetX + self.target_sprite_width / 2,
                            'y': self.targetY + self.target_sprite_height / 2,
                            'vx': random.uniform(-2, 2),
                            'vy': random.uniform(-2, 0),
                            'life': random.randint(30, 45),
                            'max_life': 45,
                            'radius': random.randint(1, 3)
                        }
                        self.coin_particles.append(particle)
                    self.generateTarget()

        for particle in self.coin_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
        self.coin_particles = [p for p in self.coin_particles if p['life'] > 0]
        for particle in self.coin_particles:
            cx = int(particle['x'])
            cy = int(particle['y'])
            cv2.circle(frame, (cx, cy), particle['radius'], (0,215,255), thickness=-1)

        disp = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR) if self.edgesMode else frame

        if self.heartSprite is not None:
            heart_h, heart_w = self.heartSprite.shape[:2]
            margin = 5
            start_x = 10
            start_y = 10
            for i in range(self.lives):
                x = start_x + i * (heart_w + margin)
                y = start_y
                if self.heartSprite.shape[2] == 4:
                    heart_bgr = self.heartSprite[:, :, :3]
                    alpha_channel = self.heartSprite[:, :, 3] / 255.0
                    alpha = np.dstack([alpha_channel] * 3)
                    roi = disp[y:y+heart_h, x:x+heart_w]
                    blended = (alpha * heart_bgr.astype(float) + (1 - alpha) * roi.astype(float)).astype(np.uint8)
                    disp[y:y+heart_h, x:x+heart_w] = blended
                else:
                    disp[y:y+heart_h, x:x+heart_w] = self.heartSprite
            if hasattr(self, 'prev_lives'):
                if self.lives < self.prev_lives:
                    lost_index = self.lives
                    if self.heartSprite is not None:
                        heart_w = self.heartSprite.shape[1]
                        heart_h = self.heartSprite.shape[0]
                    else:
                        heart_w, heart_h = 30, 30
                    heart_x = 10 + lost_index * (heart_w + margin)
                    heart_y = 10
                    for _ in range(10):
                        particle = {
                            'x': heart_x + heart_w/2,
                            'y': heart_y + heart_h/2,
                            'vx': random.uniform(-2, 2),
                            'vy': random.uniform(-2, 0),
                            'life': random.randint(30, 45),
                            'max_life': 45,
                            'radius': random.randint(1, 3)
                        }
                        self.heart_particles.append(particle)
            self.prev_lives = self.lives

            text_x = 10
            text_y = start_y + heart_h + 30
            cv2.putText(disp, f"Score: {self.score}", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(disp, f"Score: {self.score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        new_heart_particles = []
        for particle in self.heart_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            if particle['life'] > 0:
                new_heart_particles.append(particle)
                cx = int(particle['x'])
                cy = int(particle['y'])
                cv2.circle(disp, (cx, cy), particle['radius'], (0,215,255), thickness=-1)
        self.heart_particles = new_heart_particles

        if self.autoThresholdCalibrated and self.spawned:
            frame_width = disp.shape[1]
            for enemy in self.enemies:
                enemy_width = self.enemySprite_width if self.enemySprite is not None else 50
                enemy_height = self.enemySprite_height if self.enemySprite is not None else 50
                in_platform = False
                for platform in self.capturedCoords:
                    if self.rectangles_intersect(enemy['x'], enemy['y'], enemy_width, enemy_height,
                                                 platform[0], platform[1], platform[2], platform[3]):
                        in_platform = True
                        break
                if in_platform:
                    valid_spawn = False
                    attempts = 0
                    while not valid_spawn and attempts < 10:
                        candidate_x = random.randint(0, frame_width - enemy_width)
                        candidate_y = enemy['y']
                        valid_spawn = True
                        for platform in self.capturedCoords:
                            if self.rectangles_intersect(candidate_x, candidate_y, enemy_width, enemy_height,
                                                         platform[0], platform[1], platform[2], platform[3]):
                                valid_spawn = False
                                break
                        if valid_spawn:
                            enemy['x'] = candidate_x
                        attempts += 1

                predicted_x = enemy['x'] + enemy['direction'] * enemy['speed']
                collision = False
                for platform in self.capturedCoords:
                    if self.rectangles_intersect(predicted_x, enemy['y'], enemy_width, enemy_height,
                                                 platform[0], platform[1], platform[2], platform[3]):
                        collision = True
                        break
                if collision:
                    enemy['direction'] = -enemy['direction']
                    predicted_x = enemy['x'] + enemy['direction'] * enemy['speed']
                enemy['x'] = predicted_x

                if enemy['x'] <= 0:
                    enemy['x'] = 0
                    enemy['direction'] = 1
                elif enemy['x'] + enemy_width >= frame_width:
                    enemy['x'] = frame_width - enemy_width
                    enemy['direction'] = -1

                ex, ey = int(enemy['x']), int(enemy['y'])
                if self.enemySprite is not None:
                    if self.enemySprite.shape[2] == 4:
                        enemy_bgr = self.enemySprite[:, :, :3]
                        alpha_channel = self.enemySprite[:, :, 3] / 255.0
                        alpha_enemy = np.dstack([alpha_channel] * 3)
                        if ey + enemy_height <= disp.shape[0] and ex + enemy_width <= disp.shape[1]:
                            roi_enemy = disp[ey:ey+enemy_height, ex:ex+enemy_width]
                            blended_enemy = (alpha_enemy * enemy_bgr.astype(float) + (1 - alpha_enemy) * roi_enemy.astype(float)).astype(np.uint8)
                            disp[ey:ey+enemy_height, ex:ex+enemy_width] = blended_enemy
                    else:
                        if ey + enemy_height <= disp.shape[0] and ex + enemy_width <= disp.shape[1]:
                            disp[ey:ey+enemy_height, ex:ex+enemy_width] = self.enemySprite

                if not self.invulnerable and self.sprite is not None:
                    player_x, player_y = self.x_offset, self.y_offset
                    player_w, player_h = self.sprite_width, self.sprite_height
                    if not (player_x + player_w <= enemy['x'] or player_x >= enemy['x'] + enemy_width or
                            player_y + player_h <= enemy['y'] or player_y >= enemy['y'] + enemy_height):
                        self.lives -= 1
                        print("Player hit by enemy! Lives remaining:", self.lives)
                        pygame.mixer.Sound.play(self.scream_sound)
                        # NEW: Record the score in a CSV file when lives reach 0.
                        if self.lives <= 0:
                            with open("scores.csv", "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(["player", self.score])
                        heart_index = self.lives
                        if self.heartSprite is not None:
                            heart_w = self.heartSprite.shape[1]
                            heart_h = self.heartSprite.shape[0]
                        else:
                            heart_w, heart_h = 30, 30
                        heart_x = 10 + heart_index * (heart_w + 5)
                        heart_y = 10
                        for _ in range(10):
                            particle = {
                                'x': heart_x + heart_w/2,
                                'y': heart_y + heart_h/2,
                                'vx': random.uniform(-2, 2),
                                'vy': random.uniform(-2, 0),
                                'life': random.randint(30, 45),
                                'max_life': 45,
                                'radius': random.randint(1, 3)
                            }
                            self.heart_particles.append(particle)
                        self.invulnerable = True
                        self.red_tint = True
                        QTimer.singleShot(1000, self.reset_invulnerability)
                        QTimer.singleShot(1000, self.reset_red_tint)
                        enemy['x'] = random.randint(0, frame_width - enemy_width)
                        enemy['direction'] = random.choice([-1, 1])
                        if self.lives <= 0:
                            print("No lives left! Game Over.")
                            self.cap.release()
                            import subprocess
                            subprocess.Popen(["python", "start.py"])
                            QApplication.quit()

        new_heart_particles = []
        for particle in self.heart_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            if particle['life'] > 0:
                new_heart_particles.append(particle)
                cx = int(particle['x'])
                cy = int(particle['y'])
                cv2.circle(disp, (cx, cy), particle['radius'], (0,215,255), thickness=-1)
        self.heart_particles = new_heart_particles

        if self.freezeBoxes and hasattr(self, 'platform_glow_triggered') and self.spawned and self.sprite is not None:
            tolerance = 5
            player_bottom = self.y_offset + self.sprite_height
            for i, plat in enumerate(self.capturedCoords):
                if (self.x_offset + self.sprite_width > plat[0] and self.x_offset < plat[0] + plat[2] and
                    abs(player_bottom - plat[1]) < tolerance):
                    if not self.platform_glow_triggered[i]:
                        self.platform_glow_triggered[i] = True
                        self.platform_glow_color[i] = random.choice([(255, 0, 0), (128, 0, 128), (0, 0, 255), (0, 165, 255)])
                    for _ in range(3):
                        particle = {
                            'x': random.uniform(plat[0], plat[0] + plat[2]),
                            'y': plat[1],
                            'vx': random.uniform(-1, 1),
                            'vy': random.uniform(-2, -0.5),
                            'life': random.randint(20, 40),
                            'max_life': 40,
                            'radius': random.randint(1, 3)
                        }
                        self.platform_particles[i].append(particle)
                new_particles = []
                for particle in self.platform_particles[i]:
                    particle['x'] += particle['vx']
                    particle['y'] += particle['vy']
                    particle['life'] -= 1
                    if particle['life'] > 0:
                        new_particles.append(particle)
                        cx = int(particle['x'])
                        cy = int(particle['y'])
                        cv2.circle(disp, (cx, cy), particle['radius'], self.platform_glow_color[i], thickness=-1)
                self.platform_particles[i] = new_particles

        self.lastFrame = disp
        height, width, channel = disp.shape
        bytesPerLine = 3 * width
        qImg = QImage(disp.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        self.videoLabel.setPixmap(pixmap.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.controlBarOverlay.isVisible():
            self.updateControlBarPosition()
        if self.sliderOverlay.isVisible():
            self.updateSliderOverlayPosition()

    def average_bounding_boxes(self, bbox_list):
        if not bbox_list:
            return []
        reference_boxes = bbox_list[0]
        averaged_boxes = []
        for ref_box in reference_boxes:
            sum_x, sum_y, sum_w, sum_h = ref_box
            count = 1
            ref_center = (ref_box[0] + ref_box[2] / 2, ref_box[1] + ref_box[3] / 2)
            for boxes in bbox_list[1:]:
                best_match = None
                best_dist = float('inf')
                for box in boxes:
                    center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                    dist = np.hypot(center[0] - ref_center[0], center[1] - ref_center[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_match = box
                if best_match is not None and best_dist < 20:
                    sum_x += best_match[0]
                    sum_y += best_match[1]
                    sum_w += best_match[2]
                    sum_h += best_match[3]
                    count += 1
            averaged_box = (sum_x / count, sum_y / count, sum_w / count, sum_h / count)
            averaged_boxes.append(averaged_box)
        return averaged_boxes

    def smoothBoxes(self, newBoxes, prevBoxes, alpha=0.5, distance_threshold=20):
        smoothed = []
        used_prev = [False] * len(prevBoxes)
        for nb in newBoxes:
            x, y, w, h = nb
            center_new = (x + w / 2, y + h / 2)
            best_index = -1
            best_distance = float('inf')
            for i, pb in enumerate(prevBoxes):
                px, py, pw, ph = pb
                center_prev = (px + pw / 2, py + ph / 2)
                dist = np.hypot(center_new[0] - center_prev[0], center_new[1] - center_prev[1])
                if dist < best_distance and dist < distance_threshold and not used_prev[i]:
                    best_distance = dist
                    best_index = i
            if best_index != -1:
                pb = prevBoxes[best_index]
                new_smoothed = (
                    alpha * x + (1 - alpha) * pb[0],
                    alpha * y + (1 - alpha) * pb[1],
                    alpha * w + (1 - alpha) * pb[2],
                    alpha * h + (1 - alpha) * pb[3]
                )
                smoothed.append(new_smoothed)
                used_prev[best_index] = True
            else:
                smoothed.append(nb)
        return smoothed

    def updateControlBarPosition(self):
        margin = 0
        width = self.videoLabel.width()
        height = 40
        self.controlBarOverlay.setGeometry(margin, margin, width, height)

    def updateSliderOverlayPosition(self):
        global_btn_pos = self.thresholdButton.mapToGlobal(self.thresholdButton.rect().bottomLeft())
        local_pos = self.videoLabel.mapFromGlobal(global_btn_pos)
        x = local_pos.x()
        y = local_pos.y() + 5
        self.sliderOverlay.setGeometry(x, y, self.sliderOverlay.width(), self.sliderOverlay.height())

    def resizeEvent(self, event):
        if self.videoLabel.pixmap():
            self.videoLabel.setPixmap(self.videoLabel.pixmap().scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.controlBarOverlay.isVisible():
            self.updateControlBarPosition()
        if self.sliderOverlay.isVisible():
            self.updateSliderOverlayPosition()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def updateKeyState(self, key, state):
        if key in self.key_state:
            self.key_state[key] = state
            print(f"Key state for '{key}' updated to {state}")

class SerialReaderThread(QThread):
    key_signal = pyqtSignal(str, bool)

    def __init__(self, port="COM6", baudrate=115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self._running = True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Serial port {self.port} opened at {self.baudrate} baud.")
        except Exception as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

    def run(self):
        if not self.ser:
            return
        while self._running:
            data = self.ser.read(1)
            if not data:
                continue
            try:
                char = data.decode('utf-8', errors='ignore')
            except Exception as e:
                print("Decoding error:", e)
                continue
            if char == 'a':
                self.key_signal.emit('a', True)
                threading.Timer(0.3, lambda: self.key_signal.emit('a', False)).start()
            elif char == 'd':
                self.key_signal.emit('d', True)
                threading.Timer(0.3, lambda: self.key_signal.emit('d', False)).start()
            elif char == ' ':
                self.key_signal.emit(' ', True)
            elif char == '_':
                self.key_signal.emit(' ', False)
        if self.ser:
            self.ser.close()

    def stop(self):
        self._running = False
        self.wait()

from PyQt5.QtCore import QObject

class AutoCalibrateKeyFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_C:
                obj.manualCalibrate()
                return True
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWidget()
    serial_thread = SerialReaderThread("COM6", 115200)
    serial_thread.key_signal.connect(window.updateKeyState)
    serial_thread.start()
    autoCalibFilter = AutoCalibrateKeyFilter()
    window.installEventFilter(autoCalibFilter)

    window.showMaximized()
    exit_code = app.exec_()
    serial_thread.stop()
    sys.exit(exit_code)
