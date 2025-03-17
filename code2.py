import cv2
import mediapipe as mp
import serial
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1
)

# Initialize serial connection with Arduino
ser = serial.Serial('COM4', 9600)  # Change to your Arduino port
time.sleep(2)  # Allow serial connection to establish

# Initialize video capture
cap = cv2.VideoCapture(0)
prev_fingers = -1
last_command_time = 0
command_delay = 0.3  # 300ms delay between commands

def calculate_fingers(hand_landmarks):
    fingers = 0
    hand_points = hand_landmarks.landmark
    
    # Thumb (check if it's extended outward)
    thumb_tip = hand_points[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_points[mp_hands.HandLandmark.THUMB_MCP]
    if thumb_tip.x < thumb_mcp.x - 0.05:  # Adjusted threshold for better detection
        fingers += 1

    # Other fingers (check if they're open)
    finger_joints = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    for tip, pip in finger_joints:
        if hand_points[tip].y < hand_points[pip].y - 0.05:  # Increased threshold
            fingers += 1

    return fingers

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Process image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_fingers = -1
    speed_percent = 0
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        current_fingers = calculate_fingers(hand_landmarks)
        
        # Map fingers to speed (0-5)
        speed_level = min(current_fingers, 5)
        speed_percent = int((speed_level / 5) * 100)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display status
        status_text = f'Fingers: {current_fingers} | Speed: {speed_percent}%'
        cv2.putText(image, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send command with debouncing
        if current_fingers != prev_fingers and (time.time() - last_command_time) > command_delay:
            ser.write(str(speed_level).encode())
            prev_fingers = current_fingers
            last_command_time = time.time()

    # Show turn off status when no hand detected
    else:
        cv2.putText(image, 'Fan: OFF', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if prev_fingers != -1 and (time.time() - last_command_time) > command_delay:
            ser.write(b'0')
            prev_fingers = -1
            last_command_time = time.time()

    cv2.imshow('Gesture Controlled Fan', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()