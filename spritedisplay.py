import cv2
import numpy as np
import keyboard  # For smooth continuous key detection

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

# Scale the sprite to 20% of its original size
scale_factor = 0.2
new_width = int(sprite.shape[1] * scale_factor)
new_height = int(sprite.shape[0] * scale_factor)
sprite = cv2.resize(sprite, (new_width, new_height), interpolation=cv2.INTER_AREA)
sprite_height, sprite_width = sprite.shape[:2]

# ---------------------------------------------------
# Open the camera and set a lower resolution to reduce lag:
# ---------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit(1)

# Set a lower resolution (adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Grab an initial frame to determine frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the camera.")
    exit(1)
frame_height, frame_width = frame.shape[:2]

# ---------------------------------------------------
# Initial sprite position:
# ---------------------------------------------------
x_offset = 10  # Start near the left edge.
# Set the sprite so that its bottom aligns with the bottom of the frame.
ground_y = frame_height - sprite_height
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

    # Recalculate frame dimensions and ground level (dynamic in case the resolution changes)
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
    # Handle Jumping:
    # ---------------------------------------------------
    # If not jumping, ensure the sprite stays on the ground.
    if not is_jumping:
        y_offset = ground_y

    # Start a jump if SPACE is pressed and we're not already in a jump.
    if key_state["space"] and not is_jumping:
        is_jumping = True
        jump_velocity = -15  # Negative value moves the sprite upward.
        key_state["space"] = False  # Clear the space key to avoid retriggering.

    # If in a jump, update vertical position using simple physics.
    if is_jumping:
        y_offset += jump_velocity
        jump_velocity += gravity  # Gravity reduces upward velocity, then increases downward speed.
        # End the jump when the sprite reaches or passes the ground.
        if y_offset >= ground_y:
            y_offset = ground_y
            is_jumping = False
            jump_velocity = 0

    # ---------------------------------------------------
    # Overlay the Sprite onto the Frame:
    # ---------------------------------------------------
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
    cv2.imshow("Camera with Jumping Sprite", frame)

    # Exit if 'q' is pressed in the OpenCV window.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------
# Cleanup:
# ---------------------------------------------------
cap.release()
cv2.destroyAllWindows()
