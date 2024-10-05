import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
from collections import deque, Counter

# Initialize Mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize face mesh and hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Open input video (or webcam)
cap = cv2.VideoCapture(0)

# Emoji floating animation parameters
floating_emoji = None
floating_emoji_y = None
floating_emoji_x = None
floating_text = None
floating_start_time = None

# Detection history buffer (deque for the last 15 frames, assuming 30 fps for a 0.5-second buffer)
detection_history = deque(maxlen=15)

# Load a font that supports emojis (use any system font that has emoji support)
emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"  # Path for Windows (Segoe UI Emoji font)
emoji_font = ImageFont.truetype(emoji_font_path, 100)
text_font = ImageFont.truetype(emoji_font_path, 50)

# Function to reset floating emoji animation
def set_floating_emoji(text, emoji, x, y):
    global floating_emoji, floating_emoji_x, floating_emoji_y, floating_text, floating_start_time
    floating_emoji = emoji
    floating_text = text
    floating_emoji_x = x
    floating_emoji_y = y
    floating_start_time = time.time()

# Function to detect Thumbs Up
def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < wrist.y and index_finger_tip.y > thumb_tip.y

# Function to detect Heart gesture
def detect_heart(hand_landmarks_left, hand_landmarks_right):
    left_thumb_tip = hand_landmarks_left.landmark[mp_hands.HandLandmark.THUMB_TIP]
    right_thumb_tip = hand_landmarks_right.landmark[mp_hands.HandLandmark.THUMB_TIP]
    left_index_tip = hand_landmarks_left.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    right_index_tip = hand_landmarks_right.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_dist = np.linalg.norm(np.array([left_thumb_tip.x, left_thumb_tip.y]) - np.array([right_thumb_tip.x, right_thumb_tip.y]))
    index_dist = np.linalg.norm(np.array([left_index_tip.x, left_index_tip.y]) - np.array([right_index_tip.x, right_index_tip.y]))
    return thumb_dist < 0.05 and index_dist < 0.05

# Function to detect OK Hand
def detect_ok_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])) < 0.05

# Function to detect Victory Hand
def detect_victory_hand(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return index_finger_tip.y < ring_finger_tip.y and middle_finger_tip.y < pinky_finger_tip.y

# Function to check if all five fingers are extended on both hands (2x High Five)
def detect_high_five_both_hands(hand_landmarks_left, hand_landmarks_right):
    fingers_extended_left = [hand_landmarks_left.landmark[i].y < hand_landmarks_left.landmark[mp_hands.HandLandmark.WRIST].y for i in 
                            [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                             mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]]
    fingers_extended_right = [hand_landmarks_right.landmark[i].y < hand_landmarks_right.landmark[mp_hands.HandLandmark.WRIST].y for i in 
                             [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                              mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]]
    return all(fingers_extended_left) and all(fingers_extended_right)

# Improved facepalm detection: Stricter x and y proximity check to the forehead
def detect_facepalm(hand_landmarks, face_landmarks):
    forehead_y = (face_landmarks.landmark[9].y + face_landmarks.landmark[10].y) / 2
    forehead_x_center = (face_landmarks.landmark[9].x + face_landmarks.landmark[10].x) / 2

    # Get average position of all four fingers
    four_fingers_y = [hand_landmarks.landmark[i].y for i in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]]
    four_fingers_x = [hand_landmarks.landmark[i].x for i in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]]

    fingers_close_to_forehead = all(finger_y < forehead_y + 0.05 and abs(finger_x - forehead_x_center) < 0.1 for finger_y, finger_x in zip(four_fingers_y, four_fingers_x))
    return fingers_close_to_forehead

# Helper function to convert OpenCV image to PIL and draw emojis/text
def draw_pil_text_on_cv2(image, text, emoji, position, text_font, emoji_font):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text_position = (position[0] - 100, position[1] - 100)  # Adjust positioning
    emoji_position = (position[0] - 50, position[1] - 50)
    draw.text(text_position, text, font=text_font, fill=(255, 255, 255, 255))
    draw.text(emoji_position, emoji, font=emoji_font, fill=(255, 255, 255, 255))
    return np.array(pil_image)

# Process each frame in the video
start_time = time.time()  # Track frame rate
frame_count = 0  # Count frames for frame rate display
most_likely_gesture = None  # Initialize the most likely gesture to None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame to create a "selfie" view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as required by Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face and hand landmarks
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Create a blank image with the same dimensions as the input video
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype="uint8")

    # Draw face landmarks on the blank image
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=blank_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # Initialize current frame's gesture detection
    current_gesture = None

    if hand_results.multi_hand_landmarks:
        hands_detected = len(hand_results.multi_hand_landmarks)
        heart_detected = False

        # **Always draw hand landmarks, regardless of gesture detection**
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=blank_frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        # Detect if a "2x High Five" (both hands, all fingers extended) is happening
        if hands_detected == 2 and detect_high_five_both_hands(hand_results.multi_hand_landmarks[0], hand_results.multi_hand_landmarks[1]):
            # Skip detection for this frame if "2x High Five" detected, but hands are still drawn
            current_gesture = None
        else:
            # Detect Heart Gesture first (if both hands are present)
            if hands_detected == 2:
                if detect_heart(hand_results.multi_hand_landmarks[0], hand_results.multi_hand_landmarks[1]):
                    current_gesture = "Heart"

            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Skip other detections if Heart is detected
                if current_gesture == "Heart":
                    continue

                # Detect Facepalm (if face landmarks are also detected)
                if face_results.multi_face_landmarks and detect_facepalm(hand_landmarks, face_results.multi_face_landmarks[0]):
                    current_gesture = "Facepalm"
                    break  # Skip any other gesture detection

                # Detect Thumbs Up
                if detect_thumbs_up(hand_landmarks):
                    current_gesture = "Thumbs Up"
                    continue  # Skip Victory Hand detection if Thumbs Up is detected

                # Detect Victory Hand
                if detect_victory_hand(hand_landmarks):
                    current_gesture = "Victory Hand"

                # Detect OK Hand
                if detect_ok_hand(hand_landmarks):
                    current_gesture = "OK Hand"

    # Add detected gesture to the history buffer
    if current_gesture:
        detection_history.append(current_gesture)

    # Process the detection history
    if len(detection_history) > 0:
        # Count occurrences of each gesture in the buffer
        gesture_counter = Counter(detection_history)
        most_likely_gesture = gesture_counter.most_common(1)[0][0]

        # Set the floating emoji based on the most frequent gesture in the history
        if most_likely_gesture == "Heart":
            set_floating_emoji("Heart Detected", "❤️", frame.shape[1] // 2, frame.shape[0] // 2)
        elif most_likely_gesture == "Facepalm":
            set_floating_emoji("Facepalm Detected", "🤦", frame.shape[1] // 2, frame.shape[0] // 2)
        elif most_likely_gesture == "Thumbs Up":
            set_floating_emoji("Thumbs Up Detected", "👍", frame.shape[1] // 2, frame.shape[0] // 2)
        elif most_likely_gesture == "Victory Hand":
            set_floating_emoji("Victory Hand Detected", "✌️", frame.shape[1] // 2, frame.shape[0] // 2)
        elif most_likely_gesture == "OK Hand":
            set_floating_emoji("OK Hand Detected", "👌", frame.shape[1] // 2, frame.shape[0] // 2)

    # Clear the floating emoji if no gestures have been detected for 1 second
    if floating_start_time is not None and time.time() - floating_start_time > 1 and len(detection_history) == 0:
        floating_emoji = None
        floating_text = None

    # Resize the frame proportionally to fit the screen size without distortion
    screen_height, screen_width = 720, 1280
    h, w, _ = blank_frame.shape
    scale = min(screen_width / w, screen_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(blank_frame, (new_w, new_h))

    # Center the resized frame in a black background (letterboxing)
    output_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    x_offset = (screen_width - new_w) // 2
    y_offset = (screen_height - new_h) // 2
    output_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    # Floating emoji animation
    if floating_emoji and time.time() - floating_start_time < 2:  # Show emoji for 2 seconds
        floating_emoji_y -= 10  # Increased float speed
        output_frame = draw_pil_text_on_cv2(output_frame, floating_text, floating_emoji, (floating_emoji_x, floating_emoji_y), text_font, emoji_font)

    # Calculate and display frame rate
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        print(f"FPS: {fps:.2f}, Detection History: {detection_history}, Most Likely Gesture: {most_likely_gesture}")

    # Display the result in a window
    cv2.imshow("Gesture Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video objects
cap.release()
cv2.destroyAllWindows()
