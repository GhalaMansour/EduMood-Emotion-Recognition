#  EduMood – Students Emotion Recognition System

### *AI-Based Real-Time Classroom Emotion Analytics*

---

## 📌 1. Overview

**EduMood** is an AI-powered system designed to recognize and analyze students’ facial emotions in real time using a webcam.
The system captures video frames, detects faces, predicts emotions, stores aggregated session statistics, and visualizes the results through an interactive Streamlit dashboard.

This project is developed as part of the course:
**CSC — Design of Artificial Intelligence Systems**,
following the full **AI lifecycle** methodology.

---

## 📌 2. System Purpose

EduMood serves as an educational AI prototype that demonstrates:

* Real-time facial emotion recognition
* Classroom-level emotional analytics
* Data aggregation & visualization
* Integration of pre-trained models into a complete AI system

The system builds a structured analytic pipeline similar to Kevin Aguirre’s project, but adapted to our own architecture and academic requirements.

---

## 📌 3. High-Level System Architecture

```
┌────────────────────┐
│     Webcam Input    │
└─────────┬──────────┘
          │ Frames
          ▼
┌────────────────────┐
│  EduMoodRecognizer  │
│ ─ Face detection    │
│ ─ Emotion inference │
│ ─ Frame sampling    │
└─────────┬──────────┘
          │ Emotion record
          ▼
┌────────────────────┐
│ EduMoodSessionStats│
│ ─ Stores all rows  │
│ ─ Aggregates data  │
│ ─ Converts to DF   │
└─────────┬──────────┘
          │ Pandas DataFrame
          ▼
┌────────────────────┐
│     Streamlit UI   │
│ ─ Metrics          │
│ ─ Charts           │
│ ─ Tables           │
└────────────────────┘
```

---

## 📌 4. Technologies Used

| Component            | Purpose                               |
| -------------------- | ------------------------------------- |
| **Python 3.9+**      | Core development language             |
| **Streamlit**        | Interactive dashboard UI              |
| **streamlit-webrtc** | Real-time webcam streaming            |
| **DeepFace**         | Pre-trained emotion recognition model |
| **Pandas**           | Data storage & analysis               |
| **Altair**           | Visualization                         |
| **OpenCV**           | Frame processing                      |

---

## 📌 5. Emotion Model (DeepFace)

EduMood relies on **DeepFace’s built-in CNN expression model**, originally trained on **FER-2013**, to classify seven basic emotions:

* happy
* sad
* angry
* surprise
* neutral
* disgust
* fear

### Processing pipeline inside EduMood:

1. Detect a face
2. Crop the region of interest
3. Resize to 48×48
4. Pass through DeepFace model
5. Receive emotion probabilities
6. Select the highest-scoring label

> No retraining or fine-tuning was done — the system uses the model exactly as provided.

---

## 📌 6. Data Strategy

EduMood follows a structured data-handling approach:

* Uses a **pre-trained** model instead of custom training
* Generates a **session-level log** of all detected emotions
* Each processed frame corresponds to one row in the internal DataFrame
* Summaries are computed using aggregation (sum/mean)
* The system imitates the data-handling strategy seen in Kevin’s project

This ensures transparency and proper documentation for AI lifecycle reporting.

---

## 📌 7. Key Features

* 🔵 **Real-time emotion recognition**
* 📊 **Live dashboard analytics**
* 🧠 **Session accumulation** (every detection recorded)
* 🎚 **Frame sampling** (analyze every N frames to reduce latency)
* 🪞 **Mirror-mode** (camera flipped horizontally)
* 📈 **Bar charts, line charts, and KPI metrics**
* 🎯 **Lightweight design suitable for classrooms and demos**

---

## 📌 8. Project Structure

```
EduMood/
│
├── app.py                     # Streamlit dashboard
├── edumood_recognizer.py     # Recognition + session stats
├── requirements.txt
├── README.md                 # This documentation
├── LICENSE                   # License file
│
└── assets/ (optional)        # Images / logos
```

---

## 📌 9. Installation

```bash
pip install -r requirements.txt
```

---

## 📌 10. Running the Application

```bash
streamlit run app.py
```

---

## 📌 11. Credits & References

This project integrates and refers to the following:

* DeepFace library (Serengil et al.)
* FER-2013 dataset (ICML 2013)
* Conceptual inspiration from Kevin Aguirre’s **Facial Emotion Recognition App**
* Streamlit official documentation
* streamlit-webrtc official documentation

All external components are used under their respective licenses.

---

## 📌 12. License

The project is distributed for **academic and educational use only**.
Commercial use is not allowed unless a commercial license is obtained.
See the full terms in the **LICENSE** file.


