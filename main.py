import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(1)

# Create trackbars for adjusting Canny thresholds (optional)
def nothing(x):
    pass

cv2.namedWindow("Settings")
cv2.createTrackbar("Threshold1", "Settings", 50, 500, nothing)
cv2.createTrackbar("Threshold2", "Settings", 150, 500, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get Canny thresholds from trackbars (or hardcode)
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    edges = cv2.Canny(blurred, thresh1, thresh2)

    # Close gaps between edges (morphological closing)
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    cv2.imshow("Edges", edges)
    cv2.imshow("Bounding Boxes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
