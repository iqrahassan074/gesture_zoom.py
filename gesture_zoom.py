import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
zoom = 1.0

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist = distance(thumb_tip, index_tip)

            zoom = (dist - 0.02) * 10
            zoom = max(1.0, min(zoom, 3.0))

            cv2.putText(frame, f'Zoom: {zoom:.2f}x', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1 = max(center_x - new_w // 2, 0)
    y1 = max(center_y - new_h // 2, 0)
    x2 = min(center_x + new_w // 2, w)
    y2 = min(center_y + new_h // 2, h)

    cropped = frame[y1:y2, x1:x2]
    frame = cv2.resize(cropped, (w, h))

    cv2.imshow("Gesture Zoom Camera", frame)

    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()












