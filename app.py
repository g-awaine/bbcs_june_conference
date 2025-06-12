import cv2
import mediapipe as mp
import numpy as np

new_frame_height = 640
new_frame_width = 800

frame_count = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

text_list = []

dominant_hand = 'right'

def check_landmarks(landmarks, required_landmarks):
    try:
        return all(landmarks[lm_id].visibility > 0.5 for lm_id in required_landmarks)
    except IndexError:
        return False

def calculate_angle(point1, point2, point3):
    a = np.array(point1[:2])
    b = np.array(point2[:2])
    c = np.array(point3[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def good_morning(landmarks, wrist_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.1, min_elbow_angle=0, max_elbow_angle=15):
    wrist = landmarks[wrist_id]
    elbow = landmarks[elbow_id]
    shoulder = landmarks[shoulder_id]

    wrist_xyz = np.array([wrist.x, wrist.y, wrist.z])
    elbow_xyz = np.array([elbow.x, elbow.y, elbow.z])
    shoulder_xyz = np.array([shoulder.x, shoulder.y, shoulder.z])

    dist_wrist_to_shoulder = np.linalg.norm(wrist_xyz[:2] - shoulder_xyz[:2])

    elbow_angle = calculate_angle(shoulder_xyz, elbow_xyz, wrist_xyz)
    is_elbow_bent = min_elbow_angle < elbow_angle < max_elbow_angle

    vertical_diff = abs(wrist.y - chest_point[1])
    is_wrist_chest_level = vertical_diff < 0.2
    print("shoulder to wrist", dist_wrist_to_shoulder)
    print("vertical diff", vertical_diff)
    print("elbow_angle", elbow_angle)
    return dist_wrist_to_shoulder < dist_threshold and is_elbow_bent and is_wrist_chest_level

def is_pointing_to_chest(landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48):
    index_finger = landmarks[index_finger_id]
    elbow = landmarks[elbow_id]
    shoulder = landmarks[shoulder_id]

    index_finger_xyz = np.array([index_finger.x, index_finger.y, index_finger.z])
    elbow_xyz = np.array([elbow.x, elbow.y, elbow.z])
    shoulder_xyz = np.array([shoulder.x, shoulder.y, shoulder.z])

    dist_index_to_chest = np.linalg.norm(index_finger_xyz[:2] - chest_point[:2])

    is_forward = index_finger.z < elbow.z

    elbow_angle = calculate_angle(shoulder_xyz, elbow_xyz, index_finger_xyz)
    is_elbow_bent = min_elbow_angle < elbow_angle < max_elbow_angle

    is_index_finger_below_chest = index_finger.y > chest_point[1]
    
    return dist_index_to_chest < dist_threshold and is_forward and is_elbow_bent and is_index_finger_below_chest

def is_eating_gesture(hand_landmarks_list, face_nose_tip, face_proximity_threshold=0.3):
    # hand_landmarks_list will be a list of (x, y, z) tuples for all 21 hand landmarks
    # face_nose_tip will be an (x, y) tuple for the nose tip (no z for face detection)

    def distance_2d(a, b):
        # Calculate 2D distance for proximity check
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def distance_3d(a, b):
        # Calculate 3D distance for hand gesture specifics
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) ** 0.5

    # Check if we have enough landmarks for both hand and face
    if len(hand_landmarks_list) < 21 or face_nose_tip is None:
        return False

    wrist = hand_landmarks_list[0]
    index_tip = hand_landmarks_list[8]
    thumb_tip = hand_landmarks_list[4]
    middle_tip = hand_landmarks_list[12]
    
    thumb_index_dist = distance_3d(thumb_tip, index_tip)
    index_middle_dist = distance_3d(index_tip, middle_tip)
    wrist_to_index = distance_3d(wrist, index_tip)

    # Check if hand is near the face (using wrist as the reference for the hand, and only 2D for face)
    # We pass only x,y for wrist to distance_2d
    hand_to_face_dist = distance_2d((wrist[0], wrist[1]), face_nose_tip)
    is_near_face = hand_to_face_dist < face_proximity_threshold # Adjust this threshold as needed

    # Combined conditions for eating gesture
    return (
        thumb_index_dist < 0.05 and
        index_middle_dist < 0.05 and
        wrist_to_index < 0.12 and
        is_near_face
    )

class FeelGesture:
    def __init__(self):
        self.found_start_frame = False
        self.found_end_frame = False
        self.start_frame_count = 0

    def reinitialise(self):
        self.found_start_frame = False
        self.found_end_frame = False
        self.start_frame_count = 0
    
    def check_frame(self, frame_count, landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48):
        if not self.found_start_frame:
            self.found_start_frame = self.check_start_frame(self, landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48)
            if self.found_start_frame:
                self.start_frame_count = frame_count
                print("start", self.start_frame_count)
                return False

        elif frame_count - self.start_frame_count > 90:
            self.reinitialise()
            return False

        elif not self.found_end_frame:
            self.found_end_frame = self.check_end_frame(self, landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48)
            if self.found_end_frame:
                self.found_end_frame = True
                print("Have feel")
                return False
        else:
            self.reinitialise()
            return False

    def check_start_frame(self, landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48):
        pass

    def check_end_frame(self, landmarks, index_finger_id, elbow_id, shoulder_id, chest_point, dist_threshold=0.28, min_elbow_angle=20, max_elbow_angle=48):
        pass

class HappyGesture():
    def __init__(self):
        self.found_start_frame = False
        self.starting_frame_number = 0
        self.count = 0
        self.happy_detected = False

    def reinitialise(self):
        self.found_start_frame = False
        self.starting_frame_number = 0
        self.count = 0
        self.happy_detected = False
    
    def check_frame(self, frame_count, landmarks, wrist_id, elbow_id, shoulder_id, hip_id, chest_point, dist_threshold=0.28, min_shoulder_angle=30, max_shoulder_angle=50):
        is_happy = self.identify_happy_gesture(landmarks, wrist_id, elbow_id, shoulder_id, hip_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle)

        if self.happy_detected == True and is_happy == True:
            return True

        if not self.found_start_frame and is_happy:
            self.found_start_frame = True
            self.starting_frame_number = frame_count
            print("start", self.starting_frame_number)
    
        elif self.found_start_frame and is_happy:
            print("q")
            self.count += 1
            if self.count >= 20:
                print("happy detected")
                self.happy_detected = True
                return True

        elif self.found_start_frame and not is_happy:
            self.reinitialise()
            print("lost it", frame_count)

        return False

    def identify_happy_gesture(self, landmarks, wrist_id, elbow_id, shoulder_id, hip_id, chest_point, dist_threshold=0.28, min_shoulder_angle=30, max_shoulder_angle=50):
        wrist = landmarks[wrist_id]
        elbow = landmarks[elbow_id]
        shoulder = landmarks[shoulder_id]
        hip = landmarks[hip_id]

        wrist_xyz = np.array([wrist.x, wrist.y, wrist.z])
        elbow_xyz = np.array([elbow.x, elbow.y, elbow.z])
        shoulder_xyz = np.array([shoulder.x, shoulder.y, shoulder.z])
        hip_xyz = np.array([hip.x, hip.y, hip.z])

        dist_wrist_to_shoulder = np.linalg.norm(wrist_xyz[:2] - shoulder_xyz[:2])

        shoulder_angle = calculate_angle(hip_xyz, shoulder_xyz, elbow_xyz)
        is_shoulder_angled = min_shoulder_angle < shoulder_angle < max_shoulder_angle

        return dist_wrist_to_shoulder < dist_threshold and is_shoulder_angled
    

class BreakfastGesture():
    def __init__(self):
        self.phase = 0
        self.waiting_count = 0

    def reinitialise(self):
        self.phase = 0
        self.waiting_count = 0

    def check_frame(self, frame_count, landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold=0.04, min_shoulder_angle=0, max_shoulder_angle=20):
        is_up_position = self.identify_up_position(landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle)
        is_down_position = self.identify_down_position(landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle)

        # If the up position was found on the even phases increment phase to next phase i.e. the phase where he now lowers his hand to chin
        if is_up_position and (self.phase % 2 == 0):
            self.phase += 1
            self.waiting_count = 0
            print("now at phase:", self.phase)

        # If the down position was found on the odd phases increment phase to next phase i.e. the phase where he now lifts his hand to mouth
        elif is_down_position and (self.phase % 2 == 1):
            self.phase += 1
            self.waiting_count = 0
            print("now at phase:", self.phase)

        # Increment the waiting count while waiting for next phase 
        elif self.phase > 0:
            self.waiting_count += 1

        # If phase is 0 (didnt identify the first up position) then make waiting count 0
        elif self.phase == 0:
            self.waiting_count = 0


        # Check if waiting_count exceeds the waiting frames that is allowed
        if self.waiting_count > 60:
            # Assume the breakfast gesture was not successful
            print("Too long to continue", "phase: ", self.phase)
            self.reinitialise()

        # Check if the phase is 5 (meaning the full gesture was accomplished)
        if self.phase == 4:
            print("Breakfast was detected-----------------------------------")
            self.reinitialise()
            return True
        
        else:
            return False


    def identify_up_position(self, landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle):
        index_finger = landmarks[index_finger_id]
        elbow = landmarks[elbow_id]
        shoulder = landmarks[shoulder_id]
        hip = landmarks[hip_id]

        mouth_right = landmarks[mouth_right_id]
        mouth_left = landmarks[mouth_left_id]
        mouth_center = np.array([
            (mouth_right.x + mouth_left.x) / 2,
            (mouth_right.y + mouth_left.y) / 2,
            (mouth_right.z + mouth_left.z) / 2
        ])

        index_finger_xyz = np.array([index_finger.x, index_finger.y, index_finger.z])
        elbow_xyz = np.array([elbow.x, elbow.y, elbow.z])
        shoulder_xyz = np.array([shoulder.x, shoulder.y, shoulder.z])
        hip_xyz = np.array([hip.x, hip.y, hip.z])

        # Get distance from index finger to mouth centre
        dist_index_finger_to_mouth = np.linalg.norm(index_finger_xyz[:2] - mouth_center[:2])

        # Ensure shoulder is angled between the specified angles
        shoulder_angle = calculate_angle(hip_xyz, shoulder_xyz, elbow_xyz)
        is_shoulder_angled = min_shoulder_angle < shoulder_angle < max_shoulder_angle

        return dist_index_finger_to_mouth < dist_threshold and is_shoulder_angled

    def identify_down_position(self, landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle):
        index_finger = landmarks[index_finger_id]
        elbow = landmarks[elbow_id]
        shoulder = landmarks[shoulder_id]
        hip = landmarks[hip_id]

        mouth_right = landmarks[mouth_right_id]
        mouth_left = landmarks[mouth_left_id]
        mouth_center = np.array([
            (mouth_right.x + mouth_left.x) / 2,
            (mouth_right.y + mouth_left.y) / 2,
            (mouth_right.z + mouth_left.z) / 2
        ])

        index_finger_xyz = np.array([index_finger.x, index_finger.y, index_finger.z])
        elbow_xyz = np.array([elbow.x, elbow.y, elbow.z])
        shoulder_xyz = np.array([shoulder.x, shoulder.y, shoulder.z])
        hip_xyz = np.array([hip.x, hip.y, hip.z])

        # Get distance from index finger to chest centre
        dist_index_finger_to_chest = np.linalg.norm(index_finger_xyz[:2] - chest_point[:2])

        # Ensure shoulder is angled between the specified angles
        shoulder_angle = calculate_angle(hip_xyz, shoulder_xyz, elbow_xyz)
        is_shoulder_angled = min_shoulder_angle < shoulder_angle < max_shoulder_angle

        return dist_index_finger_to_chest < dist_threshold and is_shoulder_angled


class SatayGesture():
    def __init__(self):
        self.phase = 0
        self.waiting_count = 0

    def reinitialise(self):
        self.phase = 0
        self.waiting_count = 0

    def check_frame(self, frame_count, landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold=0.04, min_shoulder_angle=0, max_shoulder_angle=20):
        is_inner_position = self.identify_inner_position(landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle)
        is_outer_position = self.identify_outer_position(landmarks, index_finger_id, elbow_id, shoulder_id, hip_id, mouth_right_id, mouth_left_id, chest_point, dist_threshold, min_shoulder_angle, max_shoulder_angle)

        # If the up position was found on the even phases increment phase to next phase i.e. the phase where he now lowers his hand to chin
        if is_inner_position and (self.phase % 2 == 0):
            self.phase += 1
            self.waiting_count = 0
            print("now at phase:", self.phase)

        # If the down position was found on the odd phases increment phase to next phase i.e. the phase where he now lifts his hand to mouth
        elif is_outer_position and (self.phase % 2 == 1):
            self.phase += 1
            self.waiting_count = 0
            print("now at phase:", self.phase)

        # Increment the waiting count while waiting for next phase 
        elif self.phase > 0:
            self.waiting_count += 1

        # If phase is 0 (didnt identify the first up position) then make waiting count 0
        elif self.phase == 0:
            self.waiting_count = 0


        # Check if waiting_count exceeds the waiting frames that is allowed
        if self.waiting_count > 60:
            # Assume the breakfast gesture was not successful
            print("Too long to continue", "phase: ", self.phase)
            self.reinitialise()

        # Check if the phase is 5 (meaning the full gesture was accomplished)
        if self.phase == 3:
            print("Satay was detected-----------------------------------")
            self.reinitialise()
            return True
        
        else:
            return False
        

        

# Initialise HappyGesture object
happy_gesture = HappyGesture()

# Initialise BreakfastGesture object
breakfast_gesture = BreakfastGesture()

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()
else:
    pass

while True:
    ret, frame = cap.read()

    frame_count += 1

    if ret and frame_count % 1 == 0:
        pass
    else:
        continue

    processed_frame = cv2.resize(frame, (new_frame_width, new_frame_height))
    h, w, _ = processed_frame.shape

    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    # Process for pose and hands
    results = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)
    
    # Process for face
    results_face = face_detection.process(frame_rgb)

    left_pointing = False
    right_pointing = False
    is_good_morning = False
    is_happy = False
    is_eating = False # New variable for eating gesture
    is_breakfast = False
    face_nose_tip = None
    if results_face.detections:
        for detection in results_face.detections:
            # The nose tip is landmark 2 in MediaPipe Face Detection
            nose_tip_lm = detection.location_data.relative_keypoints[2]
            # ONLY use x, y for face_nose_tip as MediaPipe Face Detection does not provide z
            face_nose_tip = (nose_tip_lm.x, nose_tip_lm.y) 
            
            # Optionally draw face landmarks (for debugging)
            # mp_drawing.draw_detection(processed_frame, detection)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get shoulders and mouth
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

        chest_center = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2,
        ])
        mouth_center = np.array([
            (left_mouth.x + right_mouth.x) / 2,
            (left_mouth.y + right_mouth.y) / 2,
            (left_mouth.z + right_mouth.z) / 2,
        ])

        # Check pointing if the index, elbow and shoulder landmarks are available
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
        ]
        if check_landmarks(landmarks, required_landmarks):
            left_pointing = is_pointing_to_chest(
                landmarks,
                mp_pose.PoseLandmark.LEFT_INDEX,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                chest_center
            )
             
        # Check pointing if the index, elbow and shoulder landmarks are available
        required_landmarks = [
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        if check_landmarks(landmarks, required_landmarks):
            right_pointing = is_pointing_to_chest(
                landmarks,
                mp_pose.PoseLandmark.RIGHT_INDEX,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                chest_center
            )

        
        # required_landmarks = [
        #     mp_pose.PoseLandmark.RIGHT_WRIST,
        #     mp_pose.PoseLandmark.RIGHT_ELBOW,
        #     mp_pose.PoseLandmark.RIGHT_SHOULDER,
        # ]
        # if check_landmarks(landmarks, required_landmarks):
        #     is_good_morning = good_morning(
        #         landmarks,
        #         mp_pose.PoseLandmark.RIGHT_INDEX,
        #         mp_pose.PoseLandmark.RIGHT_ELBOW,
        #         mp_pose.PoseLandmark.RIGHT_SHOULDER,
        #         chest_center
        #     )

        required_landmarks = [
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        if check_landmarks(landmarks, required_landmarks):
            is_happy = happy_gesture.check_frame(
                frame_count,
                landmarks,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                chest_center
            )


        required_landmarks = [
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.MOUTH_RIGHT,
            mp_pose.PoseLandmark.MOUTH_LEFT
        ]
        if check_landmarks(landmarks, required_landmarks):
            is_breakfast = breakfast_gesture.check_frame(
                frame_count,
                landmarks,
                mp_pose.PoseLandmark.RIGHT_INDEX,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.MOUTH_RIGHT,
                mp_pose.PoseLandmark.MOUTH_LEFT,
                chest_center
            )

        # Draw pose
        mp_drawing.draw_landmarks(processed_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        chest_pixel = (int(chest_center[0] * w), int(chest_center[1] * h))
        mouth_pixel = (int(mouth_center[0] * w), int(mouth_center[1] * h))
        cv2.circle(processed_frame, chest_pixel, 2, (255, 255, 0))
        cv2.circle(processed_frame, mouth_pixel, 2, (255, 255, 0))

        if left_pointing or right_pointing:
            cv2.putText(processed_frame, "Pointing to Chest", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            text_list.append("I ")
        else:
            cv2.putText(processed_frame, "Not Pointing", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if is_good_morning:
            cv2.putText(processed_frame, "Morning", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            text_list.append("Good Morning ")
        else:
            cv2.putText(processed_frame, "Not Morning", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        if is_happy:
            cv2.putText(processed_frame, "Happy", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            text_list.append("Happy ")
        else:
            cv2.putText(processed_frame, "Not Happy", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            hand_points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Pass both hand_points and face_nose_tip to the eating gesture function
            # The is_eating_gesture function now handles 2D vs 3D distances internally
            if is_eating_gesture(hand_points, face_nose_tip):
                is_eating = True # Set the flag if eating gesture is detected

            mp_drawing.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    if is_breakfast:
        cv2.putText(processed_frame, "Breakfast", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        text_list.append("Breakfast ")
    else:
        cv2.putText(processed_frame, "Not Breakfast", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    if is_eating:
        cv2.putText(processed_frame, "Eating Gesture Detected", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    else:
        cv2.putText(processed_frame, "Not Eating", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)


    # Pass the raw text into the large language model

    cv2.imshow("Image", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
pose.close() # Close pose
face_detection.close() # Close face detection
cap.release()
cv2.destroyAllWindows()