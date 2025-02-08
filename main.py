import sys
import cv2
import numpy as np
from collections import deque
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

        self.edgesMode = False

        # For one-shot calibration: use these counters.
        self.autoAdjustFrames = 100  # number of frames to collect for calibration
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False  # becomes True when calibration completes

        # Flag to control auto calibration. When False, manual slider adjustments stick.
        self.auto_calibration_enabled = True

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
        # Original buttons remain unchanged.
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
        self.sliderOverlay.resize(300, 200)  # tall enough for two sliders and one button

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

        # Connect slider signals so that manual adjustments disable auto calibration.
        self.lowerSlider.valueChanged.connect(self.disableAutoCalibration)
        self.upperSlider.valueChanged.connect(self.disableAutoCalibration)

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

        # --- Auto Calibrate Button (in slider overlay) ---
        self.autoCalibrateButton = QPushButton("Auto Calibrate")
        self.autoCalibrateButton.setStyleSheet("font-size:10pt; color: white; background-color: #444; border: none; padding: 5px;")
        self.autoCalibrateButton.clicked.connect(self.resetCalibration)
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

        # Set up CLAHE for adaptive histogram equalization.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def disableAutoCalibration(self, value):
        """If the user adjusts either slider, disable auto calibration."""
        if self.auto_calibration_enabled:
            self.auto_calibration_enabled = False
            print("Auto calibration disabled due to manual slider adjustment.")

    def resetCalibration(self):
        """When the Auto Calibrate button is pressed:
           - Clear previous calibration counters
           - Enable auto calibration
           - (The next 100 frames will update thresholds automatically)
        """
        self.medianSum = 0.0
        self.medianCount = 0
        self.autoThresholdCalibrated = False
        self.auto_calibration_enabled = True
        self.autoCalibrateButton.setText("Auto Calibrating")
        print("Auto calibration initiated.")
        # After 2 seconds, revert the button text.
        QTimer.singleShot(2000, lambda: self.autoCalibrateButton.setText("Auto Calibrate"))

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            # Pressing 'h' toggles the control bar overlay.
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
            self.edgesViewButton.setText("Bounding Boxes")
        else:
            self.edgesViewButton.setText("Edge Detection")

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to normalize the image under varying lighting.
        norm_gray = self.clahe.apply(gray)

        # Smooth the image.
        blurred = cv2.GaussianBlur(norm_gray, (5, 5), 0)

        # --- Auto Calibration (if enabled and not yet calibrated) ---
        if self.auto_calibration_enabled and not self.autoThresholdCalibrated:
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
                print(f"Auto calibrated thresholds: lower={lower_auto}, upper={upper_auto}")
        # Get thresholds from sliders (manual adjustments override auto updates).
        lower_val = self.lowerSlider.value()
        upper_val = self.upperSlider.value()

        # Apply Canny edge detection.
        edges = cv2.Canny(blurred, lower_val, upper_val)

        # Morphological closing to enhance edges.
        kernel = np.ones((7, 7), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.edgesMode:
            # If in edge mode, display the edge image.
            disp = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
        else:
            # Find contours and draw smooth outlines.
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter out small contours.
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]
            for cnt in filtered_contours:
                arc_len = cv2.arcLength(cnt, True)
                epsilon = 0.02 * arc_len  # Adjust factor (2% of arc length) as needed
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
            disp = frame

        # Convert the processed frame to QImage and display.
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
