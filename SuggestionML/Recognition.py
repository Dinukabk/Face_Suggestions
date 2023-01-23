import cv2
import streamlit as st
import face_recognition
import pandas as pd
import numpy as np


WEBCAMNUM = 2
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]

st.title("Recognition")
FRAME_WINDOW = st.image([])

@st.cache
def load_known_data():
    DB = pd.read_csv(PATH_DATA)
    return (
        DB['name'].values, 
        DB[COLS_ENCODE].values
        )

def capture_face(video_capture):

    for i in range(3):
        video_capture.read()

    while(True):
        ret, frame = video_capture.read()
        FRAME_WINDOW.image(frame[:, :, ::-1])
     
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))

def recognize_frame(frame):

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
       
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]
        similarity = face_distance_to_conf(face_distances[best_match_index], 0.5)
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_DARK, 2)
        return name, similarity, frame[:, :, ::-1]

if __name__ == "__main__":
    while(True):
        known_face_names, known_face_encodings = load_known_data()
        video_capture = cv2.VideoCapture(WEBCAMNUM)
        frame = capture_face(video_capture)
        name, similarity, frame = recognize_frame(frame)
        FRAME_WINDOW.image(frame)
        if similarity > 0.75:
            label = f"**{name}**: *{similarity:.2%} likely*"
            st.markdown(label)
            break

    st.button('contunue......')