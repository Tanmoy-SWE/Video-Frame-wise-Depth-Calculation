import cv2
import time
import mediapipe as mp
import random

# Define the hand gestures for rock, paper, and scissors
GESTURE_ROCK = "rock"
GESTURE_PAPER = "paper"
GESTURE_SCISSORS = "scissors"

# Define the rules for the game
RULES = {
    GESTURE_ROCK: GESTURE_SCISSORS,
    GESTURE_PAPER: GESTURE_ROCK,
    GESTURE_SCISSORS: GESTURE_PAPER
}

# Grabbing the Holistic Model from Mediapipe and Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on the image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

# Focal length of the camera (adjust as per your camera)
focal_length = 1.40625


# Function to determine the gesture based on hand landmarks
def determine_gesture(hand_landmarks):
    # Get the landmarks for the index, middle, ring fingers, and the wrist
    index_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    ring_finger_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP]
    thumb_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]

    # Calculate the y-coordinates of the finger tips
    index_y = index_finger_tip.y
    middle_y = middle_finger_tip.y
    ring_y = ring_finger_tip.y
    thumb_y = thumb_finger_tip.y

    # Calculate the distance between the ring finger mcp and ring finger tip
    ring_distance = calculate_distance(ring_finger_mcp, ring_finger_tip)

    # Determine the gesture based on the criteria
    if thumb_y<middle_y:
        return GESTURE_ROCK
    elif ring_y < index_y:  # Adjust the threshold as needed
        return GESTURE_PAPER
    else:
        return GESTURE_SCISSORS


# Function to calculate the Euclidean distance between two landmarks
def calculate_distance(landmark1, landmark2):
    x1, y1, z1 = landmark1.x, landmark1.y, landmark1.z
    x2, y2, z2 = landmark2.x, landmark2.y, landmark2.z
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
    return distance

# Initializing the player's gesture
player_gesture = None

# Variables for countdown
countdown_start_time = 0
countdown_duration = 3
countdown_active = False
countdown_end_time = 0

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using the holistic model
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the right hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Accessing and calculating the depth of hand landmarks
    if results.right_hand_landmarks:
        # Determine the gesture based on hand landmarks
        player_gesture = determine_gesture(results.right_hand_landmarks)

        for landmark in results.right_hand_landmarks.landmark:
            # Get the normalized coordinates
            landmark_x = landmark.x
            landmark_y = landmark.y
            landmark_z = landmark.z

            # Calculate the depth (distance from the camera)
            depth = landmark_z * focal_length  # Adjust the scale as needed
            print("Right Hand Landmark - Depth:", depth)

    # Countdown timer logic
    if countdown_active:
        current_time = time.time()
        time_left = countdown_end_time - current_time

        # Display countdown on the screen
        cv2.putText(image, "Countdown: " + str(int(time_left)), (10, 220), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 2)

        # Check if countdown has ended
        if time_left <= 0:
            countdown_active = False

            # Determine the computer's gesture randomly
            computer_gesture = random.choice([GESTURE_ROCK, GESTURE_PAPER, GESTURE_SCISSORS])

            # Display the player's and computer's gestures
            cv2.putText(image, "Player: " + str(player_gesture), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Computer: " + str(computer_gesture), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 2)

            # Determine the result of the game
            result_text = ""
            if player_gesture and computer_gesture:
                if player_gesture == computer_gesture:
                    result_text = "It's a tie!"
                elif RULES[player_gesture] == computer_gesture:
                    result_text = "You win!"
                else:
                    result_text = "Computer wins!"

                # Display the result of the game
                cv2.putText(image, result_text, (10, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                # Display the resulting image
                cv2.imshow("Facial and Hand Landmarks", image)

                # Enter key 'q' to break the loop
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

capture.release()
cv2.destroyAllWindows()

