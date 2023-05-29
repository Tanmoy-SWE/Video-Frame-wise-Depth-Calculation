import cv2
import time
import mediapipe as mp

# Grabbing the Holistic Model from Mediapipe and Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # Capture frame by frame
    ret, frame = capture.read()

    # Resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the frame from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using the holistic model
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
    )

    # Drawing Right hand Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Left hand Landmarks
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

    # Calculate and display the depth of each hand landmark
    if results.right_hand_landmarks:
        for landmark in mp_holistic.HandLandmark:
            # Get the normalized coordinates of the landmark
            normalized_landmark = results.right_hand_landmarks.landmark[landmark]
            # Get the depth (z-coordinate) of the landmark
            depth = normalized_landmark.z
            # Convert the depth to the scale of the frame dimensions
            depth_pixel = int(depth * frame.shape[0])
            # Display the depth value on the image
            # Display the depth value on the image
            cv2.putText(image, f"R-{landmark.name}: {depth_pixel}", (10, 100 + 30 * landmark.value),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)

            # Calculate and display the depth of each facial landmark
        if results.face_landmarks:
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                # Get the normalized coordinates of the landmark
                normalized_landmark = results.face_landmarks.landmark[idx]
                # Get the depth (z-coordinate) of the landmark
                depth = normalized_landmark.z
                # Convert the depth to the scale of the frame dimensions
                depth_pixel = int(depth * frame.shape[0])
                # Display the depth value on the image
                cv2.putText(image, f"F-{idx}: {depth_pixel}", (10, 400 + 30 * idx),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

            # Display the resulting image
        cv2.imshow("Facial and Hand Landmarks", image)

        # Enter 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()
