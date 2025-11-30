import cv2
from edumood_recognizer import EduMoodSessionStats, EduMoodRecognizer

def test_single_image_emotion():
    # 1) نجهز الـ session stats والـ recognizer
    stats = EduMoodSessionStats()
    recognizer = EduMoodRecognizer(stats, analyze_every_n=1)

    # 2) نقرأ صورة تجريبية من مجلد test_images
    img = cv2.imread("test_images/happy_student.jpg")
    assert img is not None, "Test image not found"

    # 3) نحلل الفريم باستخدام الكود الحقيقي
    annotated, emotions = recognizer._analyze_frame(img)

    # 4) نتأكد إن الكود اشتغل بدون ما يطيح
    assert isinstance(emotions, list)
