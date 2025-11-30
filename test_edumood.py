import cv2
from edumood_recognizer import EduMoodSessionStats, EduMoodRecognizer

def test_single_image_emotion():
    
    stats = EduMoodSessionStats()
    recognizer = EduMoodRecognizer(stats, analyze_every_n=1)

    
    img = cv2.imread("test_images/happy_student.jpg")
    assert img is not None, "Test image not found"

    
    annotated, emotions = recognizer._analyze_frame(img)

    
    assert isinstance(emotions, list)
