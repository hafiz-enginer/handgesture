import streamlit as st
import cv2
import mediapipe as mp
import platform
import threading
import numpy as np
import time

# Set up Streamlit page
st.set_page_config(page_title="🖐️ Gesture Volume Control", layout="wide")
st.title("🖐️ Hand Gesture Based Volume Controller")
st.markdown("Click the button to **start detection** and control volume with hand gestures:")

# Volume control flag
enable_actions = False
try:
    if platform.system() == "Linux":
        import os
        if "DISPLAY" in os.environ:
            import pyautogui
            enable_actions = True
    else:
        import pyautogui
        enable_actions = True
except Exception as e:
    st.warning("pyautogui not available")

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# UI Buttons
start_button = st.button("▶️ Start Detection")
frame_placeholder = st.empty()
status_placeholder = st.empty()

# Get gesture
def get_custom_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    if thumb_tip < index_tip and middle_tip > index_tip:
        return "Volume Up"
    elif thumb_tip > index_tip and middle_tip < index_tip:
        return "Volume Down"
    elif thumb_tip < index_tip and middle_tip < index_tip:
        return "Mute"
    else:
        return "None"

# Camera loop (blocking inside start_button condition)
if start_button:
    cap = cv2.VideoCapture(0)

    st.markdown("**Press Stop (browser tab) to end detection.**")

    while True:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("❌ Webcam not detected.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_detected = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_custom_gesture(hand_landmarks.landmark)

                gesture_detected = gesture

                if enable_actions:
                    if gesture == "Volume Up":
                        pyautogui.press("volumeup")
                    elif gesture == "Volume Down":
                        pyautogui.press("volumedown")
                    elif gesture == "Mute":
                        pyautogui.press("volumemute")

                # Display gesture name on frame
                cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Display frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        status_placeholder.info(f"🖐️ Detected Gesture: **{gesture_detected}**")

        # Small delay to control frame rate
        time.sleep(0.03)
