import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

st.title("Posenet Personal Trainer")

option = st.selectbox(
    "Choose your personal trainer AI model:",
    ("Random Forest", "Gradient Boosting"),
    index=None,
    placeholder="Select contact method...",
)

st.write("You selected:", option)

model_dict = {
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
}
if option is not None:
    with open(f"{model_dict[option]}.pkl", "rb") as f:
        model = pickle.load(f)


frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")
pt_message_placeholder = st.empty()
pt_prediction_placeholder = st.empty()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while option and cap.isOpened() and not stop_button_pressed:
        ret, image = cap.read()

        if not ret:
            st.write("Video capture ended")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Right hand
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

        # Left Hand
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

        # Pose Detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        frame_placeholder.image(image, channels="BGR")
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Concate rows
            row = pose_row

            # Make Detections
            X = pd.DataFrame([row])
            pose_class = model.predict(X)[0]
            pose_prob = model.predict_proba(X)[0]
            if "bad" in pose_class:
                pt_message_placeholder.write("Upright torso, do not lean your weight on one side")
            if "good" in pose_class:
                pt_message_placeholder.write("Good posture, keep it up!")
            pt_prediction_placeholder.write(f"Pose: {pose_class}, Probability: {round(pose_prob[np.argmax(pose_prob)], 2)}")

        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break


cap.release()
# # cv2.destroyAllWindows
