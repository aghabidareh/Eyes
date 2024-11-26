import cv2
import mediapipe as mp
import pygame
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()
alert_sound_path = "alert.wav"
try:
    alert_sound = pygame.mixer.Sound(alert_sound_path)
except pygame.error:
    print(f"فایل {alert_sound_path} پیدا نشد.")
    quit()

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks, eye_indices):
    vertical_1 = calculate_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    vertical_2 = calculate_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    horizontal = calculate_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25
DROWSINESS_DURATION = 3

eye_closed_time = 0
alert_played = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    height, width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            for index in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[index].x * width)
                y = int(landmarks[index].y * height)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            if avg_ear < EAR_THRESHOLD:
                if eye_closed_time == 0:
                    eye_closed_time = time.time()
                elif time.time() - eye_closed_time > DROWSINESS_DURATION and not alert_played:
                    alert_sound.play(loops=-1)
                    alert_played = True
            else:
                eye_closed_time = 0
                if alert_played:
                    alert_sound.stop()
                    alert_played = False

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()