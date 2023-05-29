import cv2
import time
import mediapipe as mp

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

    # Drawing the facial landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
    )

    # Drawing the right hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Drawing the left hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # # Accessing and calculating the depth of hand landmarks
    # if results.right_hand_landmarks:
    #     for landmark in results.right_hand_landmarks.landmark:
    #         # Get the normalized coordinates
    #         landmark_x = landmark.x
    #         landmark_y = landmark.y
    #         landmark_z = landmark.z
    #
    #         # Calculate the depth (distance from the camera)
    #         depth = landmark_z * focal_length  # Adjust the scale as needed
    #         print("Right Hand Landmark - Depth:", depth)

    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            # Get the normalized coordinates
            landmark_x = landmark.x
            landmark_y = landmark.y
            landmark_z = landmark.z

            # Calculate the depth (distance from the camera)
            depth = landmark_z   # Adjust the scale as needed
            print("Left Hand Landmark - Depth:", depth)

    # #Accessing and calculating the depth of face landmarks
    # if results.face_landmarks:
    #     for idx, landmark in enumerate(results.face_landmarks.landmark):
    #         # Get the normalized coordinates
    #         landmark_x = landmark.x
    #         landmark_y = landmark.y
    #         landmark_z = landmark.z
    #
    #         # Calculate the depth (distance from the camera)
    #         depth = landmark_z * focal_length  # Adjust the scale as needed
    #         print("Face Landmark", idx, "- Depth:", depth)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
