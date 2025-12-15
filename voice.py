import cv2
import mediapipe as mp
import pyautogui
import pyttsx3
import numpy as np
import time
import pygetwindow as gw

pyautogui.FAILSAFE = False  # Disable PyAutoGUI failsafe

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize tracking variables
last_click_time = time.time()
current_time = time.time()
click_in_progress = False
last_opened_window = None  # Track the last opened window

# Function to perform left click
def perform_left_click():
    global click_in_progress
    click_in_progress = True
    pyautogui.click()
    print("Left Click performed")
    click_in_progress = False

# Function to perform right click
def perform_right_click():
    global click_in_progress
    click_in_progress = True
    pyautogui.rightClick()
    print("Right Click performed")
    click_in_progress = False

# Function to perform double click and track opened app
def perform_double_click():
    global click_in_progress, last_opened_window
    click_in_progress = True
    pyautogui.doubleClick()
    print("Double Click performed")
    time.sleep(0.5)  # Short delay to allow window to activate
    active_window = gw.getActiveWindow()
    if active_window:
        last_opened_window = active_window
        app_name = active_window.title
        print(f"You clicked on: {app_name}")
        engine.say(f"Opening {app_name}")
        engine.runAndWait()
    click_in_progress = False

# Function to scroll
def perform_scroll(direction):
    pyautogui.scroll(50 if direction == "up" else -50)
    print(f"Scrolling {direction}")

def detect_and_speak_app():
    time.sleep(1)
    active_window = gw.getActiveWindow()
    if active_window:
        app_name = active_window.title
        print(f"You clicked on: {app_name}")
        engine.say(f"Opening {app_name}")
        engine.runAndWait()

# Function to close the last opened app
def close_last_opened_app():
    global last_opened_window
    if last_opened_window:
        try:
            app_name = last_opened_window.title
            print(f"Closing: {app_name}")
            engine.say(f"Closing {app_name}")
            engine.runAndWait()
            last_opened_window.close()
            last_opened_window = None
        except Exception as e:
            print("Could not close the application:", e)
            engine.say("Unable to close the application")
            engine.runAndWait()
    else:
        print("No app to close.")
        engine.say("No app to close")
        engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]
            thumb = landmarks[4]
            middle_finger = landmarks[12]
            pinky = landmarks[20]

            # Cursor movement
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Calculate distances
            thumb_index_distance = np.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)
            thumb_middle_distance = np.hypot(thumb.x - middle_finger.x, thumb.y - middle_finger.y)
            index_middle_distance = np.hypot(index_finger.x - middle_finger.x, index_finger.y - middle_finger.y)
            thumb_pinky_distance = np.hypot(thumb.x - pinky.x, thumb.y - pinky.y)

            # Gesture detection
            current_time = time.time()

            if thumb_index_distance < 0.05:
                if current_time - last_click_time < 0.3:
                    perform_double_click()
                else:
                    perform_left_click()
                last_click_time = current_time
            elif thumb_middle_distance < 0.05:
                perform_right_click()
            elif index_middle_distance < 0.05:
                perform_scroll("up" if index_finger.y < middle_finger.y else "down")
            elif thumb_pinky_distance < 0.05:
                close_last_opened_app()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
