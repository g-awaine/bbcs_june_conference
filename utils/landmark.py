import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Load your image
image_path = r"img2_eat_v2.2-e69b21.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Failed to load image. Check the path!")
    exit()

# Convert the image to RGB as MediaPipe expects
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get hand landmarks
results = hands.process(image_rgb)

if not results.multi_hand_landmarks:
    print("No hand landmarks detected")
else:
    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f"Hand {hand_idx + 1}:")
        for idx, lm in enumerate(hand_landmarks.landmark):
            print(f"  Landmark {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

hands.close()
