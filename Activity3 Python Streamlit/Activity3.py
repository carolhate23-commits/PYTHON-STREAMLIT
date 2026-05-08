import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")


    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    alert_objects = ["cell phone", "person", "cup"] 
    detected_alerts = []


    if results and results[0].boxes is not None:
        names = model.names
        class_ids = results[0].boxes.cls.tolist()
        
        for c in class_ids:
            label = names[int(c)]
            if label in alert_objects:
                detected_alerts.append(label)

      
        annotated_frame = results[0].plot()
    else:
       
        annotated_frame = img

    if detected_alerts:
        unique_alerts = ", ".join(set(detected_alerts))
        cv2.putText(
            annotated_frame,
            f"ALERT: {unique_alerts}",
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
