import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget, QDesktopWidget, QHBoxLayout
)


class CameraWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game")

        # Use a modern font for the entire application.
        app_font = QFont("Segoe UI", 10)
        self.setFont(app_font)

        # Dynamically set the window size (80% of available screen size)
        screen = QDesktopWidget().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        # Toggle between raw edge mask view and bounding boxes view.
        self.edgesMode = False

        # For smoothing bounding boxes between frames.
        self.prevBoxes = []

        # Auto-calibration parameters.
        self.autoAdjustFrames = 100  # number of frames to collect for calibration
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False  # once calibrated, stop auto-adjusting

        # --- Central Widget and Video Display ---
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setStyleSheet("background-color: black;")
        layout.addWidget(self.videoLabel)

        # --- Control Bar Overlay (with buttons) ---
        self.controlBarOverlay = QWidget(self.videoLabel)
        self.controlBarOverlay.setStyleSheet("background-color: rgba(50, 50, 50, 230); border: none;")
        self.controlBarOverlay.setVisible(False)

        controlLayout = QHBoxLayout(self.controlBarOverlay)
        controlLayout.setContentsMargins(10, 5, 10, 5)
        controlLayout.setSpacing(10)
        self.thresholdButton = QPushButton("Canny Thresholds")
        self.edgesViewButton = QPushButton("Edge Detection")
        for btn in (self.thresholdButton, self.edgesViewButton):
            btn.setFixedSize(110, 30)
            btn.setStyleSheet("font-size:10pt; color: white; background-color: transparent; border: none;")
        controlLayout.addWidget(self.thresholdButton)
        controlLayout.addWidget(self.edgesViewButton)
        controlLayout.addStretch(1)

        # --- Threshold Slider Overlay ---
        self.sliderOverlay = QWidget(self.videoLabel)
        self.sliderOverlay.setStyleSheet("background-color: rgba(40, 40, 40, 230); border: none;")
        self.sliderOverlay.setVisible(False)
        self.sliderOverlay.resize(300, 200)  # taller to accommodate the new button

        sliderLayout = QVBoxLayout(self.sliderOverlay)
        sliderLayout.setContentsMargins(10, 10, 10, 10)
        sliderLayout.setSpacing(10)
        self.lowerLabel = QLabel("Canny Lower Threshold:")
        self.lowerLabel.setStyleSheet("color: white;")
        self.lowerSlider = QSlider(Qt.Horizontal)
        self.lowerSlider.setRange(0, 500)
        self.lowerSlider.setValue(50)
        self.upperLabel = QLabel("Canny Upper Threshold:")
        self.upperLabel.setStyleSheet("color: white;")
        self.upperSlider = QSlider(Qt.Horizontal)
        self.upperSlider.setRange(0, 500)
        self.upperSlider.setValue(150)

        # Vibrant slider style (red-to-green gradient)
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

        # --- Auto Calibrate Button with feedback ---
        self.autoCalibrateButton = QPushButton("Auto Calibrate")
        self.autoCalibrateButton.setStyleSheet("font-size:10pt; color: white; background-color: #444; border: none; padding: 5px;")
        self.autoCalibrateButton.clicked.connect(self.manualCalibrate)
        sliderLayout.addWidget(self.autoCalibrateButton)

        # --- Connect Buttons ---
        self.thresholdButton.clicked.connect(self.toggleSliderOverlay)
        self.edgesViewButton.clicked.connect(self.toggleEdgesMode)

        # --- OpenCV Camera Capture ---
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Could not open camera")
            sys.exit()

        # --- Timer for Updating Frames ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # roughly 30ms per frame (~30fps)

        # Install event filter for key presses ("h" to toggle overlays).
        self.installEventFilter(self)

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
           Also provide visual feedback on the button."""
        self.autoCalibrateButton.setText("Autocalibrating")
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False
        print("Manual calibration initiated.")

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_equalized = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise.
        blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)

        # --- Auto-adjust thresholds on startup or when manually triggered ---
        if not self.autoThresholdCalibrated:
            median_val = np.median(blurred)
            self.medianSum += median_val
            self.medianCount += 1
            if self.medianCount >= self.autoAdjustFrames:
                avg_median = self.medianSum / self.medianCount
                sigma = 0.33
                lower_auto = int(max(0, (1.0 - sigma) * avg_median))
                upper_auto = int(min(255, (1.0 + sigma) * avg_median))
                self.lowerSlider.setValue(lower_auto)
                self.upperSlider.setValue(upper_auto)
                self.autoThresholdCalibrated = True
                self.autoCalibrateButton.setText("Autocalibrated")
                print(f"Auto-calibrated thresholds: lower={lower_auto}, upper={upper_auto}")
                QTimer.singleShot(2000, lambda: self.autoCalibrateButton.setText("Auto Calibrate"))

        # Get thresholds from sliders.
        lower_val = self.lowerSlider.value()
        upper_val = self.upperSlider.value()

        # --- Dual Edge Detection ---
        edges = cv2.Canny(blurred, lower_val, upper_val)
        inverted = cv2.bitwise_not(blurred)
        inverted_edges = cv2.Canny(inverted, lower_val, upper_val)
        combined_edges = cv2.bitwise_or(edges, inverted_edges)

        # Apply morphological closing to connect broken edges.
        kernel = np.ones((7, 7), np.uint8)
        closed_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Dilate the mask slightly to join nearby edge segments.
        dilate_kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(closed_edges, dilate_kernel, iterations=1)

        # Find contours in the dilated mask.
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        newBoxes = []
        frame_area = frame.shape[0] * frame.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out noise (very small areas) and avoid large false detections (e.g. the whole background).
            if area < 100 or area > frame_area * 0.9:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            newBoxes.append((x, y, w, h))

        # Smooth bounding boxes with previous detections to reduce jitter.
        smoothedBoxes = self.smoothBoxes(newBoxes, self.prevBoxes, alpha=0.5, distance_threshold=20)
        self.prevBoxes = smoothedBoxes

        # Draw bounding boxes on the frame.
        for box in smoothedBoxes:
            x, y, w, h = box
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), thickness=2)

        # Toggle view: raw closed edge mask (for debugging) or the bounding box overlay.
        if self.edgesMode:
            disp = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
        else:
            disp = frame

        # Convert processed frame to QImage and display.
        height, width, channel = disp.shape
        bytesPerLine = 3 * width
        qImg = QImage(disp.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Update overlay positions if visible.
        if self.controlBarOverlay.isVisible():
            self.updateControlBarPosition()
        if self.sliderOverlay.isVisible():
            self.updateSliderOverlayPosition()

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
        margin = 0  # span edge-to-edge
        width = self.videoLabel.width()
        height = 40  # fixed height for the control bar
        self.controlBarOverlay.setGeometry(margin, margin, width, height)

    def updateSliderOverlayPosition(self):
        global_btn_pos = self.thresholdButton.mapToGlobal(self.thresholdButton.rect().bottomLeft())
        local_pos = self.videoLabel.mapFromGlobal(global_btn_pos)
        x = local_pos.x()
        y = local_pos.y() + 5
        self.sliderOverlay.setGeometry(x, y, self.sliderOverlay.width(), self.sliderOverlay.height())

    def resizeEvent(self, event):
        if self.videoLabel.pixmap():
            self.videoLabel.setPixmap(self.videoLabel.pixmap().scaled(
                self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
    window.show()
    sys.exit(app.exec_())
