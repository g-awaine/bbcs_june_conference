import cv2
import mediapipe as mp
import numpy as np

# Define the frame width and height
new_frame_height = 640
new_frame_width = 800

# Initialise frame count
frame_count = 0

# Initialise mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Initialise the text list containnig the text from the singaporean sign language
text_list = []

# Identify the dominant hand
dominant_hand = 'right'

# Checks if all the necessary landmarks are visible
def check_landmarks(landmarks, required_landmarks):
    try:
        return all(landmarks[lm_id].visibility > 0.5 for lm_id in required_landmarks)
    except IndexError:
        return False

# Functions to identify the positions of the body
def calculate_angle(point1, point2, point3):
    """
    Returns the angle at point2 formed by point1-point2-point3
    """
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

    # Get distance from wrist to shoulder
    dist_wrist_to_shoulder = np.linalg.norm(wrist_xyz[:2] - shoulder_xyz[:2])

    # Ensure elbow is angled between the specified angles
    elbow_angle = calculate_angle(shoulder_xyz, elbow_xyz, wrist_xyz)
    is_elbow_bent = min_elbow_angle < elbow_angle < max_elbow_angle

    # Identify if the wrist is around chest level
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

    # Get distance from index finger to chest
    dist_index_to_chest = np.linalg.norm(index_finger_xyz[:2] - chest_point[:2])

    # Identify if wrist is closer to chest
    is_forward = index_finger.z < elbow.z

    # Ensure elbow is angled between the specified angles
    elbow_angle = calculate_angle(shoulder_xyz, elbow_xyz, index_finger_xyz)
    is_elbow_bent = min_elbow_angle < elbow_angle < max_elbow_angle

    # Ensure the index finger is below the chest
    is_index_finger_below_chest = index_finger.y > chest_point[1]
    
    return dist_index_to_chest < dist_threshold and is_forward and is_elbow_bent and is_index_finger_below_chest

# DEPRECATED FOR NOW
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
            # Increment the count of frames where the gesture was active
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

        # Get distance from wrist to shoulder
        dist_wrist_to_shoulder = np.linalg.norm(wrist_xyz[:2] - shoulder_xyz[:2])

        # Ensure shoulder is angled between the specified angles
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
    # Proceed to main code
    pass

while True:
    # Capture a single frame
    ret, frame = cap.read()

    # Increment the frame count
    frame_count += 1

    # Check if the frame was captured
    if ret and frame_count % 1 == 0:
        # Proceed with the program if frame was captured and its a third frame 
        pass
    else:
        # Try to get a frame again
        continue

    # Process the frame
    processed_frame = cv2.resize(frame, (new_frame_width, new_frame_height))
    h, w, _ = processed_frame.shape

    # Also create a frame in rgb for mediapipe
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Initialise the pointers
    left_pointing = False
    right_pointing = False
    is_good_morning = False
    is_happy = False
    is_breakfast = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get shoulders and mouth
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]

        # Chest center as midpoint between shoulders
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

        # # Check pointing if the index, elbow and shoulder landmarks are available
        # required_landmarks = [
        #     mp_pose.PoseLandmark.LEFT_INDEX,
        #     mp_pose.PoseLandmark.LEFT_ELBOW,
        #     mp_pose.PoseLandmark.LEFT_SHOULDER,
        # ]
        # if check_landmarks(landmarks, required_landmarks):
        #     left_pointing = is_pointing_to_chest(
        #         landmarks,
        #         mp_pose.PoseLandmark.LEFT_INDEX,
        #         mp_pose.PoseLandmark.LEFT_ELBOW,
        #         mp_pose.PoseLandmark.LEFT_SHOULDER,
        #         chest_center
        #     )
             
        # # Check pointing if the index, elbow and shoulder landmarks are available
        # required_landmarks = [
        #     mp_pose.PoseLandmark.RIGHT_INDEX,
        #     mp_pose.PoseLandmark.RIGHT_ELBOW,
        #     mp_pose.PoseLandmark.RIGHT_SHOULDER,
        # ]
        # if check_landmarks(landmarks, required_landmarks):
        #     right_pointing = is_pointing_to_chest(
        #         landmarks,
        #         mp_pose.PoseLandmark.RIGHT_INDEX,
        #         mp_pose.PoseLandmark.RIGHT_ELBOW,
        #         mp_pose.PoseLandmark.RIGHT_SHOULDER,
        #         chest_center
        #     )

        
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

        # required_landmarks = [
        #     mp_pose.PoseLandmark.RIGHT_WRIST,
        #     mp_pose.PoseLandmark.RIGHT_ELBOW,
        #     mp_pose.PoseLandmark.RIGHT_SHOULDER,
        #     mp_pose.PoseLandmark.RIGHT_HIP,
        # ]
        # if check_landmarks(landmarks, required_landmarks):
        #     is_happy = happy_gesture.check_frame(
        #         frame_count,
        #         landmarks,
        #         mp_pose.PoseLandmark.RIGHT_WRIST,
        #         mp_pose.PoseLandmark.RIGHT_ELBOW,
        #         mp_pose.PoseLandmark.RIGHT_SHOULDER,
        #         mp_pose.PoseLandmark.RIGHT_HIP,
        #         chest_center
        #     )


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

        # Feedback
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


        if is_breakfast:
            cv2.putText(processed_frame, "Breakfast", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            text_list.append("Breakfast ")
        else:
            cv2.putText(processed_frame, "Not Breakfast", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    # Pass the raw text into the large language model

    
    # Display the frame after processing
    cv2.imshow("Image", processed_frame)



    # Pass the proper text into the Text-To-Speech model


    # End frame capture when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()