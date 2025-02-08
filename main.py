import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget, QHBoxLayout
)

class CameraWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game")
        # Ensure the window gets key events.
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Use a modern font for the entire application.
        app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)

        # --------------------------
        # Computer Vision Variables
        # --------------------------
        self.edgesMode = False               # Toggle raw edge mask vs. overlay view.
        self.prevBoxes = []                  # For smoothing bounding boxes.
        self.autoAdjustFrames = 100          # Number of frames for auto-calibration.
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False # Flag for auto calibration.
        self.lastFrame = None                # Holds the last processed frame.
        self.capturedCoords = []             # Holds averaged bounding boxes.

        # For accumulating bounding boxes over several frames.
        self.bboxAcc = []        # List of lists of bounding boxes per frame.
        self.bboxCount = 0       # How many frames accumulated.
        self.freezeBoxes = False # When True, freeze bounding boxes (use averaged ones).

        # --------------------------
        # Spawn / Gameplay Variables
        # --------------------------
        self.spawned = False     # Will be set True when the Spawn button is clicked.
        self.sprite = None       # The player sprite image.
        self.x_offset = 0        # Sprite horizontal position.
        self.y_offset = 0        # Sprite vertical position.
        self.ground_y = 0        # Calculated ground level.
        self.move_step = 5       # Pixels moved per frame horizontally.
        self.is_jumping = False  # Jump state.
        self.jump_velocity = 0   # Current jump vertical velocity.
        self.gravity = 1         # Gravity per frame.
        # Dictionary to hold current key states.
        self.key_state = {"a": False, "d": False, "space": False}
        # When calibration is triggered while the sprite is visible, its position is saved.
        self.savedSpritePos = None

        # --------------------------
        # UI Setup
        # --------------------------
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("background-color: black;")
        layout.addWidget(self.videoLabel)

        # --- Control Bar Overlay ---
        self.controlBarOverlay = QWidget(self.videoLabel)
        self.controlBarOverlay.setStyleSheet("background-color: rgba(50, 50, 50, 230); border: none;")
        self.controlBarOverlay.setVisible(False)

        controlLayout = QHBoxLayout(self.controlBarOverlay)
        controlLayout.setContentsMargins(10, 5, 10, 5)
        controlLayout.setSpacing(10)
        self.thresholdButton = QPushButton("Canny Thresholds")
        self.edgesViewButton = QPushButton("Edge Detection")
        self.spawnButton = QPushButton("Spawn")
        # Disable auto-default so that keys like SPACE won’t trigger buttons.
        for btn in (self.thresholdButton, self.edgesViewButton, self.spawnButton):
            btn.setFixedSize(110, 30)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setStyleSheet("font-size:10pt; color: white; background-color: transparent; border: none;")
        controlLayout.addWidget(self.thresholdButton)
        controlLayout.addWidget(self.edgesViewButton)
        controlLayout.addWidget(self.spawnButton)
        controlLayout.addStretch(1)

        # --- Threshold Slider Overlay ---
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
        # Connect slider changes to update labels and trigger sprite hiding.
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

        # --- Connect UI Buttons ---
        self.thresholdButton.clicked.connect(self.toggleSliderOverlay)
        self.edgesViewButton.clicked.connect(self.toggleEdgesMode)
        self.spawnButton.clicked.connect(self.spawnPlayer)

        # --------------------------
        # OpenCV Camera Capture
        # --------------------------
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Could not open camera")
            sys.exit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # roughly 30 fps

        self.installEventFilter(self)

    # --------------------------
    # Sprite & Gameplay Methods
    # --------------------------
    def spawnPlayer(self):
        """Called when Spawn is clicked. Initializes gameplay if not already spawned."""
        if not self.spawned:
            self.spawned = True
            self.reset_gameplay()
            print("Player spawned!")

    def reset_gameplay(self):
        """Load the sprite and initialize its movement variables.
           If the sprite was hidden during calibration, its saved position is used."""
        sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0001.png"
        self.sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if self.sprite is None:
            print(f"Error: Could not load sprite from {sprite_path}")
            return
        # Use the original sprite size without scaling.
        self.sprite_height, self.sprite_width = self.sprite.shape[:2]
        # Set initial position; if we have a saved position, use it.
        if self.savedSpritePos is not None:
            self.x_offset, self.y_offset = self.savedSpritePos
        else:
            self.x_offset = 10
            self.y_offset = 0
        self.is_jumping = False
        self.jump_velocity = 0
        self.gravity = 1
        self.move_step = 5
        self.key_state = {"a": False, "d": False, "space": False}

    def hideSprite(self):
        """If the sprite is visible, save its current position and hide it immediately."""
        if self.spawned:
            self.savedSpritePos = (self.x_offset, self.y_offset)
            self.spawned = False
            self.sprite = None
            print("Sprite hidden for calibration, saved position:", self.savedSpritePos)

    def respawnSprite(self, pos):
        """Respawn the sprite at the given position."""
        self.savedSpritePos = None
        self.spawned = True
        self.reset_gameplay()  # This loads the sprite and resets variables.
        # Override the position with the saved position.
        self.x_offset, self.y_offset = pos
        print("Sprite respawned at", pos)

    # --------------------------
    # UI Event Handlers
    # --------------------------
    def updateLowerLabel(self, value):
        self.lowerLabel.setText(f"Canny Lower Threshold: {value}")
        # When thresholds change, if a sprite is active, hide it.
        if self.spawned:
            self.hideSprite()
        if self.freezeBoxes:
            self.reset_bbox_accumulation()

    def updateUpperLabel(self, value):
        self.upperLabel.setText(f"Canny Upper Threshold: {value}")
        # When thresholds change, if a sprite is active, hide it.
        if self.spawned:
            self.hideSprite()
        if self.freezeBoxes:
            self.reset_bbox_accumulation()

    def reset_bbox_accumulation(self):
        """Reset the bounding box accumulator so that averaging will restart."""
        self.bboxAcc = []
        self.bboxCount = 0
        self.freezeBoxes = False

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            # Toggle the control bar overlay using the H key.
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
                self.key_state["space"] = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self.spawned:
            if event.key() == Qt.Key_A:
                self.key_state["a"] = False
            elif event.key() == Qt.Key_D:
                self.key_state["d"] = False
            elif event.key() == Qt.Key_Space:
                self.key_state["space"] = False
        super().keyReleaseEvent(event)

    def toggleSliderOverlay(self):
        visible = not self.sliderOverlay.isVisible()
        self.sliderOverlay.setVisible(visible)
        if visible:
            self.sliderOverlay.raise_()
            self.updateSliderOverlayPosition()

    def toggleEdgesMode(self):
        self.edgesMode = not self.edgesMode
        if self.edgesMode:
            self.edgesViewButton.setText("Overlay View")
        else:
            self.edgesViewButton.setText("Edge Detection")

    def manualCalibrate(self):
        """Reset calibration counters and start auto-calibration.
           Also hide the sprite (if present) and reset bounding box averaging."""
        if self.spawned:
            self.hideSprite()
        self.autoCalibrateButton.setText("Autocalibrating")
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False
        print("Manual calibration initiated.")
        self.reset_bbox_accumulation()

    def onCalibrationComplete(self):
        """Called after auto-calibration delay; resets button text and respawns sprite if needed."""
        self.autoCalibrateButton.setText("Auto Calibrate")
        if self.savedSpritePos is not None:
            self.respawnSprite(self.savedSpritePos)
        self.reset_bbox_accumulation()

    # --------------------------
    # Frame Processing
    # --------------------------
    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process for edge detection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)

        # --- Auto-adjust thresholds (if not already calibrated) ---
        if not self.autoThresholdCalibrated:
            median_val = np.median(blurred)
            scale_factor = 0.26
            calibrated_val = median_val * scale_factor
            self.medianSum += calibrated_val
            self.medianCount += 1
            if self.medianCount >= self.autoAdjustFrames:
                avg_calibrated = self.medianSum / self.medianCount
                sigma = 0.36  # yields thresholds near 25 (lower) and 53 (upper)
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

        # --- Bounding Box Averaging (only if sprite is not spawned) ---
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
            else:
                smoothedBoxes = self.capturedCoords
        else:
            # In spawn mode, ignore bounding boxes.
            smoothedBoxes = []

        # Draw bounding boxes (only if not spawned)
        if not self.spawned:
            for box in smoothedBoxes:
                x, y, w, h = box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), thickness=2)

        # --------------
        # Gameplay Code
        # --------------
        if self.spawned:
            # Calculate ground level based on current frame height.
            ground_y = frame.shape[0] - self.sprite_height
            self.ground_y = ground_y
            # Horizontal movement:
            if self.key_state.get("a", False):
                self.x_offset -= self.move_step
            if self.key_state.get("d", False):
                self.x_offset += self.move_step
            self.x_offset = max(0, min(self.x_offset, frame.shape[1] - self.sprite_width))
            # Jumping:
            if not self.is_jumping:
                self.y_offset = ground_y
            if self.key_state.get("space", False) and not self.is_jumping:
                self.is_jumping = True
                self.jump_velocity = -15
                self.key_state["space"] = False  # consume the key
            if self.is_jumping:
                self.y_offset += self.jump_velocity
                self.jump_velocity += self.gravity
                if self.y_offset >= ground_y:
                    self.y_offset = ground_y
                    self.is_jumping = False
                    self.jump_velocity = 0
            # Overlay the sprite onto the frame.
            if self.sprite is not None:
                if self.sprite.shape[2] == 4:
                    sprite_bgr = self.sprite[:, :, :3]
                    alpha_channel = self.sprite[:, :, 3] / 255.0
                    alpha = np.dstack([alpha_channel] * 3)
                    roi = frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width]
                    blended = (alpha * sprite_bgr.astype(float) + (1 - alpha) * roi.astype(float)).astype(np.uint8)
                    frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width] = blended
                else:
                    frame[self.y_offset:self.y_offset+self.sprite_height, self.x_offset:self.x_offset+self.sprite_width] = self.sprite

        # --------------------
        # Display the Frame
        # --------------------
        disp = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR) if self.edgesMode else frame
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
        """
        Given a list of lists of bounding boxes (each box is a tuple (x, y, w, h)),
        compute an average bounding box for each object using the first frame as reference.
        """
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
        """
        Smooth new bounding boxes by comparing with previous boxes.
        For each new box, if a previous box's center is within the distance_threshold,
        update its coordinates using exponential smoothing; otherwise, use the new box.
        """
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.showMaximized()
    sys.exit(app.exec_())
