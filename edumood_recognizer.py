import cv2
from collections import Counter
from datetime import datetime
import uuid

import av
from deepface import DeepFace
import pandas as pd


class EduMoodSessionStats:
    """
 Stores emotion records for a single session.
Each record contains: recorded_at + the number of faces detected for each emotion
    """

    def __init__(self):
        self.records = []

    def add_record(self, record: dict):
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "recorded_at",
                    "happy",
                    "sad",
                    "angry",
                    "surprise",
                    "neutral",
                    "disgusted",
                    "fearful",
                ]
            )
        return pd.DataFrame(self.records)


class EduMoodRecognizer:
    """
 Reads video frames from the webcam (via streamlit-webrtc),
mirrors the image for a natural display,
and analyzes emotions every N frames using DeepFace
    """

    def __init__(self, session_stats: EduMoodSessionStats, analyze_every_n: int = 5):
        self.session_stats = session_stats
        self.analyze_every_n = analyze_every_n
        self.frame_count = 0
        self.last_annotated = None

        # Face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _analyze_frame(self, img_bgr):
        """
     Analyzes a single frame:
-Detects faces
-Extracts the dominant emotion for each face using DeepFace
-Draws bounding boxes and labels on the image
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        emotions = []

        for (x, y, w, h) in faces:
            face_roi = img_bgr[y : y + h, x : x + w]

            try:
                result = DeepFace.analyze(
                    face_roi, actions=["emotion"], enforce_detection=False
                )

                # DeepFace may return a list or a dictionary
                if isinstance(result, list):
                    dom = result[0].get("dominant_emotion", "").lower()
                else:
                    dom = result.get("dominant_emotion", "").lower()

                if dom:
                    emotions.append(dom)

                # draw rectangle and label
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    img_bgr,
                    dom,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            except Exception:
                # Ignore any frame-level errors (e.g., DeepFace failure)
                continue

        return img_bgr, emotions

    def recognize(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
     Called by streamlit-webrtc for each video frame.
Returns a processed frame annotated with bounding boxes and emotions.
        """
        self.frame_count += 1

        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Mirror the image for a natural display
        img = cv2.flip(img, 1)

        # Analyze only every Nth frame to reduce computational load
        if self.frame_count % self.analyze_every_n != 0:
            # If no previous annotated frame exists, return the raw image
            if self.last_annotated is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            # Reuse the last analyzed frame
            return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

        # Analyze the frame with DeepFace
        annotated, emotions = self._analyze_frame(img)
        self.last_annotated = annotated

        # Update session statistics if emotions were detected
        if emotions:
            counts = Counter([e.lower() for e in emotions])

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            record = {
                "recorded_at": now_str,
                "happy": counts.get("happy", 0),
                "sad": counts.get("sad", 0),
                "angry": counts.get("angry", 0),
                "surprise": counts.get("surprise", 0)
                + counts.get("surprised", 0),
                "neutral": counts.get("neutral", 0),
                "disgusted": counts.get("disgust", 0)
                + counts.get("disgusted", 0),
                "fearful": counts.get("fear", 0) + counts.get("fearful", 0),
            }

            self.session_stats.add_record(record)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
