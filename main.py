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

        self.edgesMode = False

        # --- Central Widget and Video Display ---
        # The central widget holds only the video feed.
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
        # This overlay spans the entire width of the video feed and now has a darker grey background.
        self.controlBarOverlay = QWidget(self.videoLabel)
        self.controlBarOverlay.setStyleSheet(
            "background-color: rgba(50, 50, 50, 230); border: none;")
        self.controlBarOverlay.setVisible(False)

        controlLayout = QHBoxLayout(self.controlBarOverlay)
        controlLayout.setContentsMargins(10, 5, 10, 5)
        controlLayout.setSpacing(10)
        self.thresholdButton = QPushButton("Canny Thresholds")
        self.edgesViewButton = QPushButton("Edge Detection")
        for btn in (self.thresholdButton, self.edgesViewButton):
            btn.setFixedSize(110, 30)
            # Force button text to white and use no border.
            btn.setStyleSheet("font-size:10pt; color: white; background-color: transparent; border: none;")
        controlLayout.addWidget(self.thresholdButton)
        controlLayout.addWidget(self.edgesViewButton)
        controlLayout.addStretch(1)

        # --- Threshold Slider Overlay ---
        # This overlay (which appears when the threshold button is clicked)
        # now uses a darker grey background.
        self.sliderOverlay = QWidget(self.videoLabel)
        self.sliderOverlay.setStyleSheet(
            "background-color: rgba(40, 40, 40, 230); border: none;")
        self.sliderOverlay.setVisible(False)
        self.sliderOverlay.resize(300, 140)

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

        # Custom stylesheet for vibrant slider appearance:
        slider_style = """
            QSlider::groove:horizontal {
                border: none;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
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
        self.timer.start(30)  # roughly 30 ms per frame (~30fps)

        # Install event filter so we can catch key presses in the main window.
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        """Catch key press events to toggle the control bar overlay."""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_H:
                # Toggle the entire control bar overlay on key press "h"
                visible = not self.controlBarOverlay.isVisible()
                self.controlBarOverlay.setVisible(visible)
                # Also hide the slider overlay if hiding the control bar.
                if not visible:
                    self.sliderOverlay.setVisible(False)
                if visible:
                    self.controlBarOverlay.raise_()
                    self.updateControlBarPosition()
                return True
        return super().eventFilter(source, event)

    def toggleSliderOverlay(self):
        """Show or hide the threshold slider overlay."""
        visible = not self.sliderOverlay.isVisible()
        self.sliderOverlay.setVisible(visible)
        if visible:
            self.sliderOverlay.raise_()
            self.updateSliderOverlayPosition()

    def toggleEdgesMode(self):
        """Toggle between bounding boxes view and edges view."""
        self.edgesMode = not self.edgesMode
        if self.edgesMode:
            self.edgesViewButton.setText("Bounding Boxes")
        else:
            self.edgesViewButton.setText("Edge Detection")

    def updateFrame(self):
        """Capture a frame, process it, and update the display."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Preprocessing: grayscale and blur.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Get thresholds from sliders.
        lower_val = self.lowerSlider.value()
        upper_val = self.upperSlider.value()

        # Canny edge detection.
        edges = cv2.Canny(blurred, lower_val, upper_val)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if self.edgesMode:
            # In edges view, show the processed edges.
            disp = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
        else:
            # In bounding boxes view, find contours and draw rectangles on the original frame.
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:  # filter small contours
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            disp = frame

        # Convert image from OpenCV (BGR) to QImage.
        height, width, channel = disp.shape
        bytesPerLine = 3 * width
        qImg = QImage(disp.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)

        # Scale the pixmap to fit the videoLabel while keeping the aspect ratio.
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Update overlay positions if they are visible.
        if self.controlBarOverlay.isVisible():
            self.updateControlBarPosition()
        if self.sliderOverlay.isVisible():
            self.updateSliderOverlayPosition()

    def updateControlBarPosition(self):
        """Position the control bar overlay at the top of the video feed, spanning edge-to-edge."""
        margin = 0  # no margin so it spans edge to edge
        width = self.videoLabel.width()
        height = 40  # fixed height for the control bar
        self.controlBarOverlay.setGeometry(margin, margin, width, height)

    def updateSliderOverlayPosition(self):
        """Position the slider overlay below the threshold button (which is inside the control bar)."""
        # Map the threshold button's bottom left to videoLabel coordinates.
        global_btn_pos = self.thresholdButton.mapToGlobal(self.thresholdButton.rect().bottomLeft())
        local_pos = self.videoLabel.mapFromGlobal(global_btn_pos)
        x = local_pos.x()
        y = local_pos.y() + 5  # add a small margin
        self.sliderOverlay.setGeometry(x, y, self.sliderOverlay.width(), self.sliderOverlay.height())

    def resizeEvent(self, event):
        """Ensure that the video feed is rescaled and overlays are repositioned when the window is resized."""
        if self.videoLabel.pixmap():
            self.videoLabel.setPixmap(self.videoLabel.pixmap().scaled(
                self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.controlBarOverlay.isVisible():
            self.updateControlBarPosition()
        if self.sliderOverlay.isVisible():
            self.updateSliderOverlayPosition()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Release the camera upon closing the window."""
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())