import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

# -----------------------------------------------
#               THRESHOLDS (EDIT HERE)
# -----------------------------------------------
EAR_THRESHOLD = 0.25       # Eyes closed threshold
EAR_CONSEC_FRAMES = 20     # How many frames before alert

YAW_THRESHOLD = 25         # Left-right head turn
PITCH_THRESHOLD = 16       # Up-down head tilt
ROLL_THRESHOLD = 25       # Side tilt (added)

ALARM_DELAY = 1.5          # Seconds between repeated alarms
# -----------------------------------------------

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drowsiness counters
ear_counter = 0
last_alarm_time = 0

def eye_aspect_ratio(landmarks, eye_points):
    p = []
    for idx in eye_points:
        p.append(landmarks[idx])

    # Distances
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])

    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm():
    global last_alarm_time
    current_time = time.time()

    if current_time - last_alarm_time > ALARM_DELAY:
        winsound.Beep(2500, 700)  # freq, duration
        last_alarm_time = current_time

# --------------------------
# START VIDEO CAPTURE
# --------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh_points = np.array(
            [
                (int(p.x * w), int(p.y * h), p.z)
                for p in results.multi_face_landmarks[0].landmark
            ]
        )

        # --------------------
        # EYE DROWSINESS CHECK
        # --------------------
        left_ear = eye_aspect_ratio(mesh_points, LEFT_EYE)
        right_ear = eye_aspect_ratio(mesh_points, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0

        if ear_counter > EAR_CONSEC_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            play_alarm()

        # --------------------
        # HEAD POSE ESTIMATION
        # --------------------
        # Key points: nose, eyes, mouth corners
        FACE_IDXS = [1, 33, 133, 362, 263, 61, 291]
        face_2d = []
        face_3d = []

        for idx in FACE_IDXS:
            x, y, z = mesh_points[idx]
            face_2d.append([x, y])
            face_3d.append([x, y, z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        cam_matrix = np.array([[w, 0, w/2],
                               [0, w, h/2],
                               [0, 0, 1]])

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        # HEAD TILT ALERTS
        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD or abs(roll) > ROLL_THRESHOLD:
            cv2.putText(frame, "HEAD POSE ALERT!", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            play_alarm()

        # Display angles
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()