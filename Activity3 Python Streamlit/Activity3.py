import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # --- ALERT SYSTEM FOR SPECIFIC OBJECTS ---
    alert_objects = ["cell phone", "person", "cup"]  # you can edit this
    detected_alerts = []

    names = model.names
    class_ids = results[0].boxes.cls.tolist() if results[0].boxes is not None else []

    for c in class_ids:
        label = names[int(c)]
        if label in alert_objects:
            detected_alerts.append(label)

    # Annotate frame
    annotated_frame = results[0].plot()

    # --- DISPLAY ALERT ON VIDEO ---
    if detected_alerts:
        cv2.putText(
            annotated_frame,
            f"ALERT: {', '.join(set(detected_alerts))}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,  # smoother performance
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)