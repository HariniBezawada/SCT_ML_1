import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    if not success:
        print("Frame not received.")
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    fingers = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm_list = []

            h, w, c = img.shape

            for id, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lm_list.append((cx, cy))

            # Thumb
            if lm_list[4][0] > lm_list[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for i in range(1, 5):

                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total = fingers.count(1)

            gesture = "UNKNOWN"

            if total == 0:
                gesture = "FIST"

            elif total == 5:
                gesture = "PALM"

            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "ONE"

            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "TWO"

            elif fingers == [0, 1, 1, 1, 0]:
                gesture = "THREE"

            elif fingers == [0, 1, 1, 1, 1]:
                gesture = "FOUR"

            cv2.rectangle(img, (10, 10), (320, 80), (0, 255, 0), -1)

            cv2.putText(
                img,
                "Gesture: " + gesture,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()