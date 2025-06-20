import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2
import platform

# Volume control support (pyautogui) - only for local use
enable_actions = False
try:
    if platform.system() != "Linux":
        import pyautogui
        enable_actions = True
except Exception as e:
    enable_actions = False

# Streamlit UI setup
st.set_page_config(page_title="🖐️ Gesture Volume Control", layout="centered")
st.title("🖐️ Hand Gesture Volume Controller (WebRTC)")
st.markdown("Use your webcam to control volume using hand gestures.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# Video Processor for WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.last_gesture = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = get_custom_gesture(hand_landmarks.landmark)

                # Show gesture text
                cv2.putText(img, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

                # Optional: Volume key control (only local)
                if enable_actions and gesture != self.last_gesture:
                    if gesture == "Volume Up":
                        pyautogui.press("volumeup")
                    elif gesture == "Volume Down":
                        pyautogui.press("volumedown")
                    elif gesture == "Mute":
                        pyautogui.press("volumemute")
                    self.last_gesture = gesture

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam stream
webrtc_streamer(
    key="volume-control",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
