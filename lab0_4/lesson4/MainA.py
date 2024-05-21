import dlib
import pickle
import cv2
import os
import mediapipe as mp
import HandTrackingModule as htm

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
detector = htm.handDetector(detectionCon=0.75)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
def count_fingers(image):
    img = detector.findHands(image)
    img = cv2.flip(img, 1)
    lmList, bbox = detector.findPosition(img, draw=False)
    totalFingers = 0
    if lmList:
        fingersUp = detector.fingersUp()
        totalFingers = fingersUp.count(1)
    return totalFingers - 1


cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())
video_capture = cv2.VideoCapture(0)
finger_count = 0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
            names.append(name)
        else:
            names.append("Unknown")

        finger_count = count_fingers(frame)
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if name == "Unknown":
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            elif finger_count == 1:
                cv2.putText(frame, name.split(' ')[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            elif finger_count == 2:
                cv2.putText(frame, name.split(' ')[1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


        unknown_faces = []
        for ((x, y, w, h), name) in zip(faces, names):
            if names == "Unknown":
                unknown_faces.append((x, y, w, h))

        for (x, y, w, h) in unknown_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)


    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
