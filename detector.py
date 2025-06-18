import cv2
import mediapipe as mp
import math
import time
import pygame



pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye Ration Calculation
def calculate_EAR(landmarks, eye_indices):
    vertical1 = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    vertical2 = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    horizontal = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# Eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

cap = cv2.VideoCapture(0)

score = 0
threshold = 0.25
alert_limit = 45
playing_alarm = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points]

        left_ear = calculate_EAR(landmarks, LEFT_EYE)
        right_ear = calculate_EAR(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < threshold:
            score = min(score + 0.7, 100)
        else:
            score = max(score - 0.5, 0)

        # Text
        cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
        cv2.putText(frame, f'Drowsy Score: {round(score)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)

        # Drowsiness triggered
        if score >= alert_limit:
            # Play alarm if not already playing
            if not playing_alarm:
                alarm_sound.play(-1)  # loop until stopped
                playing_alarm = True
        else:
            # Stop alarm and flashing
            if playing_alarm:
                alarm_sound.stop()
                playing_alarm = False

    else:
        # If no face, assume safe
        score = max(score - 1, 0)
        if playing_alarm:
            alarm_sound.stop()
            playing_alarm = False

    cv2.imshow('Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
