import cv2
import numpy as np
import keyboard  # For smooth continuous key detection

# ---------------------------------------------------
# Bounding box data for platforms (x, y, width, height)
boxes = [
    (358.0, 439.0, 106.0, 41.0),
    (207.00039025543956, 392.0, 54.99960974456044, 88.0),
    (150.3103411656808, 275.0000022865832, 44.69043875271128, 42.99999771341682),
    (233.99999999998835, 215.5336958424663, 41.00000000001163, 41.466304157533735),
    (313.0, 157.0, 77.49551637671563, 76.0)
]

# ---------------------------------------------------
# Setup for smooth keyboard input via a key_state dict:
# ---------------------------------------------------
key_state = {"a": False, "d": False, "space": False}

def on_key_event(e):
    if e.name in key_state:
        if e.event_type == 'down':
            key_state[e.name] = True
        elif e.event_type == 'up':
            key_state[e.name] = False

keyboard.hook(on_key_event)

# ---------------------------------------------------
# Load and scale the sprite:
# ---------------------------------------------------
sprite_path = r"C:\Users\LLR User\Desktop\your-childhood-game\Sprite-0001.png"
sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
if sprite is None:
    print(f"Error: Could not load sprite from {sprite_path}")
    exit(1)

# Scale the sprite (adjust scale factor if needed)
new_width = int(sprite.shape[1])
new_height = int(sprite.shape[0])
sprite = cv2.resize(sprite, (new_width, new_height), interpolation=cv2.INTER_AREA)
sprite_height, sprite_width = sprite.shape[:2]

# ---------------------------------------------------
# Open the camera and set a lower resolution to reduce lag:
# ---------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Grab an initial frame to determine frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the camera.")
    exit(1)
frame_height, frame_width = frame.shape[:2]

# Define the ground level (when the sprite is on the bottom of the frame)
ground_y = frame_height - sprite_height

# ---------------------------------------------------
# Initial sprite position:
# ---------------------------------------------------
x_offset = 10  # Start near the left edge.
# Start on the ground:
y_offset = ground_y

# ---------------------------------------------------
# Movement and Jump variables:
# ---------------------------------------------------
move_step = 5        # Horizontal movement per frame.
is_jumping = False   # Jump state flag.
jump_velocity = 0    # Current vertical velocity.
gravity = 1          # Gravity effect per frame.

print("Use A and D to move horizontally.")
print("Press SPACE to jump. Press 'q' (in the OpenCV window) to quit.")

# ---------------------------------------------------
# Main loop:
# ---------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    # Update frame dimensions in case they change:
    frame_height, frame_width = frame.shape[:2]
    ground_y = frame_height - sprite_height

    # ---------------------------------------------------
    # Handle Horizontal Movement:
    # ---------------------------------------------------
    if key_state["a"]:
        x_offset -= move_step
    if key_state["d"]:
        x_offset += move_step

    # Constrain horizontal position within the frame boundaries.
    x_offset = max(0, min(x_offset, frame_width - sprite_width))

    # ---------------------------------------------------
    # Check if the sprite is still supported by a platform or the ground.
    # (This lets the sprite “fall” if it walks off a platform.)
    # ---------------------------------------------------
    if not is_jumping:
        supported = False
        tolerance = 5  # pixels of tolerance when checking alignment
        # Check for platform support:
        for (plat_x, plat_y, plat_w, plat_h) in boxes:
            # Check horizontal overlap
            if x_offset + sprite_width > plat_x and x_offset < plat_x + plat_w:
                # If the sprite's bottom is close enough to the platform's top
                if abs((y_offset + sprite_height) - plat_y) < tolerance:
                    supported = True
                    break
        # Also, if the sprite is at the ground level:
        if abs((y_offset + sprite_height) - (ground_y + sprite_height)) < tolerance:
            supported = True

        # If no support is found, start falling:
        if not supported:
            is_jumping = True
            # If you’re just starting to fall off a platform, begin with zero downward velocity.
            jump_velocity = 0

    # ---------------------------------------------------
    # Handle Jumping (initiating a jump when supported):
    # ---------------------------------------------------
    if key_state["space"] and not is_jumping:
        is_jumping = True
        jump_velocity = -15  # Negative value moves the sprite upward.
        key_state["space"] = False  # Clear the key to avoid retriggering.

    # ---------------------------------------------------
    # Handle Vertical Movement (Jump/Fall Physics):
    # ---------------------------------------------------
    if is_jumping:
        prev_y_offset = y_offset
        y_offset += jump_velocity
        jump_velocity += gravity

        # Only check for landing (collision) when falling downward
        if jump_velocity > 0:
            # Check collision with each platform
            for (plat_x, plat_y, plat_w, plat_h) in boxes:
                if x_offset + sprite_width > plat_x and x_offset < plat_x + plat_w:
                    # Did the sprite cross the platform's top between the previous frame and now?
                    if (prev_y_offset + sprite_width <= plat_y or prev_y_offset + sprite_height < plat_y) and \
                       (y_offset + sprite_height >= plat_y):
                        # Land on this platform:
                        y_offset = int(plat_y - sprite_height)
                        is_jumping = False
                        jump_velocity = 0
                        break

        # Check collision with the ground:
        if y_offset >= ground_y:
            y_offset = ground_y
            is_jumping = False
            jump_velocity = 0

    # ---------------------------------------------------
    # Draw the Platforms onto the Frame:
    # ---------------------------------------------------
    for (plat_x, plat_y, plat_w, plat_h) in boxes:
        pt1 = (int(plat_x), int(plat_y))
        pt2 = (int(plat_x + plat_w), int(plat_y + plat_h))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    # ---------------------------------------------------
    # Overlay the Sprite onto the Frame:
    # ---------------------------------------------------
    # Check that the sprite doesn't go off-screen vertically.
    if y_offset < 0:
        y_offset = 0
    if y_offset + sprite_height > frame_height:
        y_offset = frame_height - sprite_height

    if sprite.shape[2] == 4:
        # If the sprite has an alpha channel, blend it.
        sprite_bgr = sprite[:, :, :3]
        alpha_channel = sprite[:, :, 3] / 255.0  # Normalize alpha to 0-1.
        alpha = np.dstack([alpha_channel] * 3)
        roi = frame[y_offset:y_offset+sprite_height, x_offset:x_offset+sprite_width]
        blended = (alpha * sprite_bgr.astype(float) + (1 - alpha) * roi.astype(float)).astype(np.uint8)
        frame[y_offset:y_offset+sprite_height, x_offset:x_offset+sprite_width] = blended
    else:
        # Otherwise, just overlay the sprite.
        frame[y_offset:y_offset+sprite_height, x_offset:x_offset+sprite_width] = sprite

    # ---------------------------------------------------
    # Display the updated frame:
    # ---------------------------------------------------
    cv2.imshow("Camera with Jumping Sprite and Platforms", frame)

    # Exit if 'q' is pressed in the OpenCV window.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------
# Cleanup:
# ---------------------------------------------------
cap.release()
cv2.destroyAllWindows()
